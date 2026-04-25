"""
AnthropicBatchBackend: lockstep batch backend for Anthropic Message Batches API.

Drop-in replacement for AnthropicBackend, but every ``.call()`` enqueues the
request and blocks on a ``Future`` until the next batch round completes. A
single coordinator thread flushes the pending queue when every *active worker*
is simultaneously blocked inside ``.call()`` — i.e., every live claim has one
pending LLM request and is ready to advance. The coordinator then submits
those N requests as one batch, waits for completion, and resolves every
Future in lockstep so all workers resume together.

Workers register their claim via the ``worker_session()`` context manager so
the coordinator knows the denominator for the "all blocked" check. Between
claims (thread-pool task boundaries) a thread is idle and not counted.

Pricing: Anthropic's Message Batches API charges ~50% of the sync Messages
API, with a 24-hour SLA (typical completion is minutes to an hour).
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
import uuid
from concurrent.futures import Future

from llm_backend import LLMBackend, LLMCall


class AnthropicBatchBackend(LLMBackend):
    """Lockstep batch-API backend.

    Thread-safe. Instantiate once, share across worker threads (or an
    ``executor`` managing them). Call ``shutdown()`` after every worker
    has exited to stop the coordinator cleanly.
    """

    #: Seconds between batch.retrieve() polls.
    POLL_INTERVAL = 10.0
    #: Condition-variable wake-up period (seconds). Used as a safety net —
    #: notify_all() is the primary signal.
    CV_TIMEOUT = 30.0
    #: Hard cap for a single batch round (matches Anthropic's SLA).
    BATCH_DEADLINE = 24 * 3600.0
    #: After the lockstep trigger fires, wait this long for stragglers
    #: before flushing. Without this, the coordinator can trip on the
    #: first worker to call during thread-pool ramp-up (worker 1 is
    #: active+blocked before workers 2+ have even entered their session).
    #: A flush during settling cancels if anything changes — so we only
    #: fire once state is truly stable.
    SETTLING_DELAY = 0.5

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        verbose: bool = True,
    ):
        super().__init__()
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

        # --- shared state (guarded by _cv) ---
        self._cv = threading.Condition()
        self._pending: dict[str, dict] = {}    # custom_id -> request params
        self._futures: dict[str, Future] = {}  # custom_id -> caller Future
        self._active: int = 0                  # worker_session() depth (claims in flight)
        self._blocked: int = 0                 # workers currently inside .call()
        self._stop: bool = False
        self._batch_rounds: int = 0

        # call_log is shared across threads — protect appends.
        self._call_log_lock = threading.Lock()

        self._client = None
        self._coord = threading.Thread(
            target=self._coordinator, name="batch-coord", daemon=True
        )
        self._coord.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def worker_session(self):
        """A claim running inside this context is counted as an *active*
        worker. The coordinator only flushes once every active worker is
        blocked inside ``.call()`` (i.e. ``blocked == active``). Enter at
        the start of each claim; the ``finally`` decrements on exit —
        both on success and on exception."""
        with self._cv:
            self._active += 1
            self._cv.notify_all()
        try:
            yield
        finally:
            with self._cv:
                self._active -= 1
                self._cv.notify_all()

    def call(self, role: str, system_prompt: str, user_prompt: str) -> str:
        """Enqueue a batch request and block until its Future is resolved.

        Must be called from inside ``worker_session()`` (enforced by the
        eval driver — the backend itself does not check, because a single
        stray call would only stall the coordinator, not corrupt data).
        """
        cid = f"{role}-{uuid.uuid4().hex[:16]}"
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        from llm_backend import AnthropicBackend
        if AnthropicBackend._supports_temperature(self.model):
            params["temperature"] = self.temperature
        fut: Future = Future()
        start = time.time()
        with self._cv:
            self._pending[cid] = params
            self._futures[cid] = fut
            self._blocked += 1
            self._cv.notify_all()
        try:
            text = fut.result()  # blocks until coordinator resolves
        finally:
            with self._cv:
                self._blocked -= 1
                # no need to notify_all — decrementing blocked while the
                # coordinator is waiting for blocked==active only ever
                # makes the condition *less* likely to flip. The next
                # notify happens when the next worker enqueues.
        elapsed = int((time.time() - start) * 1000)
        with self._call_log_lock:
            self.call_log.append(LLMCall(
                role=role, model=self.model,
                system_prompt=system_prompt[:200],
                user_prompt=user_prompt[:200],
                response=text[:500],
                elapsed_ms=elapsed,
                timestamp=time.time(),
            ))
        return text

    def shutdown(self, timeout: float = 30.0) -> None:
        """Signal the coordinator to exit. Call after every worker has
        left its ``worker_session`` — otherwise pending futures will
        never resolve."""
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._coord.join(timeout=timeout)

    def get_stats(self) -> dict:
        base = super().get_stats()
        base["batch_rounds"] = self._batch_rounds
        return base

    # ------------------------------------------------------------------
    # Coordinator
    # ------------------------------------------------------------------

    def _coordinator(self) -> None:
        while True:
            with self._cv:
                while True:
                    if self._stop and not self._pending:
                        return
                    # Lockstep trigger: every active worker has enqueued a
                    # request and is now blocked inside .call(). When that
                    # holds, pause for SETTLING_DELAY to let stragglers
                    # (e.g., thread-pool workers that haven't yet entered
                    # worker_session during startup) join the round. If
                    # state changes during the wait, re-evaluate.
                    if (self._pending and self._active > 0
                            and self._blocked == self._active):
                        snap = (self._active, self._blocked, len(self._pending))
                        self._cv.wait(timeout=self.SETTLING_DELAY)
                        post = (self._active, self._blocked, len(self._pending))
                        if post != snap:
                            continue  # something changed — re-check
                        # Stable for SETTLING_DELAY — flush.
                        batch_params = dict(self._pending)
                        batch_futures = dict(self._futures)
                        self._pending.clear()
                        self._futures.clear()
                        break
                    # Shutdown escape hatch: if stop was requested and
                    # there is pending work but no active workers,
                    # something is wrong (workers crashed without
                    # draining their futures) — flush anyway.
                    if self._stop and self._pending:
                        batch_params = dict(self._pending)
                        batch_futures = dict(self._futures)
                        self._pending.clear()
                        self._futures.clear()
                        break
                    self._cv.wait(timeout=self.CV_TIMEOUT)

            # Submit + dispatch outside the lock so new workers can enqueue
            # during the polling window (though flushes are lockstep — no
            # new workers will enqueue until the current round resolves).
            try:
                self._submit_round(batch_params, batch_futures)
            except Exception as e:
                for fut in batch_futures.values():
                    if not fut.done():
                        fut.set_exception(e)

    def _submit_round(
        self,
        params_by_cid: dict[str, dict],
        futures_by_cid: dict[str, Future],
    ) -> None:
        """Submit one batch, poll to completion, resolve every Future."""
        self._batch_rounds += 1
        round_id = self._batch_rounds
        n = len(params_by_cid)
        client = self._get_client()

        payload = [{"custom_id": cid, "params": params}
                   for cid, params in params_by_cid.items()]

        if self.verbose:
            print(f"[batch round {round_id}] submitting {n} requests...",
                  file=sys.stderr, flush=True)
        t0 = time.time()
        batch = client.messages.batches.create(requests=payload)
        batch_id = batch.id

        # Poll until ended (typical: minutes; SLA: 24h)
        deadline = t0 + self.BATCH_DEADLINE
        while True:
            if time.time() > deadline:
                raise TimeoutError(
                    f"batch {batch_id} exceeded {self.BATCH_DEADLINE}s SLA")
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                break
            time.sleep(self.POLL_INTERVAL)

        elapsed = time.time() - t0
        if self.verbose:
            counts = getattr(batch, "request_counts", None)
            print(f"[batch round {round_id}] ended in {elapsed:.0f}s "
                  f"(counts={counts})", file=sys.stderr, flush=True)

        # Dispatch results to per-request futures.
        delivered = 0
        for result in client.messages.batches.results(batch_id):
            cid = getattr(result, "custom_id", None)
            fut = futures_by_cid.get(cid) if cid else None
            if fut is None:
                continue
            rt = result.result.type
            if rt == "succeeded":
                try:
                    msg = result.result.message
                    text = "".join(
                        b.text for b in msg.content if getattr(b, "type", "") == "text"
                    )
                    if not text and msg.content:
                        text = getattr(msg.content[0], "text", "")
                    fut.set_result(text)
                    delivered += 1
                except Exception as e:
                    fut.set_exception(
                        RuntimeError(f"batch result parse error for {cid}: {e}"))
            else:
                err = getattr(result.result, "error", None)
                fut.set_exception(
                    RuntimeError(f"batch item {cid} {rt}: {err}"))

        # Fail any orphans (shouldn't happen with well-formed custom_ids).
        for cid, fut in futures_by_cid.items():
            if not fut.done():
                fut.set_exception(
                    RuntimeError(f"batch item {cid}: no result returned"))

        if self.verbose:
            print(f"[batch round {round_id}] delivered {delivered}/{n} results",
                  file=sys.stderr, flush=True)

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install: pip install anthropic"
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client
