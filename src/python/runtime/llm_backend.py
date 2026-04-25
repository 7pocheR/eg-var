"""
LLM Backend: Pluggable interface for LLM calls.

Supports: Gemini (google-genai), mock (for testing).
Each call is stateless — the backend creates a fresh request per call.

Usage:
    backend = GeminiBackend(api_key="...", model="gemini-2.5-flash")
    response = backend.call("analyst", system_prompt, user_prompt)
"""
from __future__ import annotations
import contextlib
import json
import os
import time
from dataclasses import dataclass, field


@dataclass
class LLMCall:
    """Record of a single LLM call for auditing."""
    role: str
    model: str
    system_prompt: str
    user_prompt: str
    response: str
    elapsed_ms: int
    timestamp: float


class LLMBackend:
    """Base class for LLM backends."""

    def __init__(self):
        self.call_log: list[LLMCall] = []

    def call(self, role: str, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

    @contextlib.contextmanager
    def worker_session(self):
        """No-op for sync backends; batch backends override this to
        track how many claims are currently in flight (the denominator
        for the "all blocked" lockstep trigger). Wrap each claim's work
        in this context so both sync and batch modes use the same code
        path."""
        yield

    def shutdown(self) -> None:
        """No-op for sync backends; batch backends override to stop
        their coordinator thread cleanly."""
        return

    def get_stats(self) -> dict:
        return {
            "total_calls": len(self.call_log),
            "total_ms": sum(c.elapsed_ms for c in self.call_log),
            "by_role": {
                role: len([c for c in self.call_log if c.role == role])
                for role in set(c.role for c in self.call_log)
            },
        }


class GeminiBackend(LLMBackend):
    """Google Gemini API backend."""

    def __init__(self, api_key: str | None = None,
                 model: str = "gemini-2.5-flash",
                 temperature: float = 0.0,
                 max_retries: int = 3,
                 retry_delay: float = 2.0):
        super().__init__()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "google-genai package required. Install: pip install google-genai"
                )
        return self._client

    def call(self, role: str, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        from google.genai import types

        start = time.time()

        for attempt in range(self.max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.temperature,
                    ),
                )
                text = response.text
                elapsed = int((time.time() - start) * 1000)

                self.call_log.append(LLMCall(
                    role=role, model=self.model,
                    system_prompt=system_prompt[:200],
                    user_prompt=user_prompt[:200],
                    response=text[:500],
                    elapsed_ms=elapsed,
                    timestamp=time.time(),
                ))
                return text

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"Gemini API failed after {self.max_retries} retries: {e}")


class MockBackend(LLMBackend):
    """Mock backend for testing pipeline without API calls."""

    def __init__(self):
        super().__init__()

    def call(self, role: str, system_prompt: str, user_prompt: str) -> str:
        start = time.time()

        if role == "analyst":
            response = self._mock_analyst(user_prompt)
        elif role == "formalizer":
            response = self._mock_formalizer(user_prompt)
        elif role == "solver":
            response = self._mock_solver(user_prompt)
        elif role == "evidence_formalizer":
            response = self._mock_evidence(user_prompt)
        elif role == "renderer":
            response = self._mock_renderer(user_prompt)
        else:
            response = '{"error": "unknown role"}'

        elapsed = int((time.time() - start) * 1000)
        self.call_log.append(LLMCall(
            role=role, model="mock",
            system_prompt=system_prompt[:200],
            user_prompt=user_prompt[:200],
            response=response[:500],
            elapsed_ms=elapsed,
            timestamp=time.time(),
        ))
        return response

    def _mock_analyst(self, prompt: str) -> str:
        return "Mock reading guide. Key column is the first string column. Numeric columns contain counts."

    def _mock_formalizer(self, prompt: str) -> str:
        # Extract claim
        claim = ""
        if "CLAIM TO FORMALIZE:" in prompt:
            claim = prompt.split("CLAIM TO FORMALIZE:")[1].strip().split("\n")[0].lower()

        # Basic pattern matching for common TabFact claim types
        if "more" in claim and "than" in claim:
            words = claim.split()
            return json.dumps({
                "goal": "HasProperty(entity1, property, v1) AND HasProperty(entity2, property, v2) AND ValueGt(v1, v2)",
                "goal_type": "comparison",
                "entities_needed": [{"name": "entity1", "lookup_key": "entity1", "role": "lhs"},
                                    {"name": "entity2", "lookup_key": "entity2", "role": "rhs"}],
                "properties_needed": [{"name": "property", "maps_to_column": "unknown"}],
                "evidence_plan": "Look up values for both entities, compare.",
                "confidence": "low",
            })

        return json.dumps({
            "goal": "", "goal_type": "other",
            "entities_needed": [], "properties_needed": [],
            "evidence_plan": "", "confidence": "low",
        })

    def _mock_solver(self, prompt: str) -> str:
        return json.dumps({
            "lookups": [],
            "verdict": "ABSTAIN",
            "proof_sketch": "Mock solver — no lookups.",
            "kernel_checks": [],
        })

    def _mock_evidence(self, prompt: str) -> str:
        return json.dumps({"formal_evidence": "mock", "value_type": "str", "parsed_value": "mock"})

    def _mock_renderer(self, prompt: str) -> str:
        return "Mock answer."


class MistralBackend(LLMBackend):
    """Mistral API backend (includes Leanstral)."""

    def __init__(self, api_key: str | None = None,
                 model: str = "labs-leanstral-2603",
                 temperature: float = 0.0,
                 max_retries: int = 3,
                 retry_delay: float = 2.0):
        super().__init__()
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None

    def call(self, role: str, system_prompt: str, user_prompt: str) -> str:
        import urllib.request
        start = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        req_body = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            req_body["temperature"] = self.temperature

        payload = json.dumps(req_body).encode()

        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    "https://api.mistral.ai/v1/chat/completions",
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                )
                resp = urllib.request.urlopen(req, timeout=120)
                data = json.loads(resp.read())
                text = data["choices"][0]["message"]["content"]
                elapsed = int((time.time() - start) * 1000)

                self.call_log.append(LLMCall(
                    role=role, model=self.model,
                    system_prompt=system_prompt[:200],
                    user_prompt=user_prompt[:200],
                    response=text[:500],
                    elapsed_ms=elapsed,
                    timestamp=time.time(),
                ))
                return text

            except urllib.error.HTTPError as e:
                body = e.read().decode()[:200] if hasattr(e, 'read') else str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"Mistral API failed after {self.max_retries} retries: {e.code} {body}")
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"Mistral API failed after {self.max_retries} retries: {e}")


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend."""

    def __init__(self, api_key: str | None = None,
                 model: str = "claude-sonnet-4-6",
                 temperature: float = 0.0,
                 max_tokens: int = 8192,
                 max_retries: int = 3,
                 retry_delay: float = 2.0):
        super().__init__()
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install: pip install anthropic"
                )
        return self._client

    @staticmethod
    def _supports_temperature(model: str) -> bool:
        """Anthropic's extended-thinking / reasoning models (Opus 4.7+)
        deprecated the ``temperature`` parameter. Return False for those
        so we omit the param entirely; True for Sonnet 4.x and earlier."""
        m = model.lower()
        # Known reasoning models that reject temperature. Add to this
        # list as Anthropic ships more.
        if m.startswith("claude-opus-4-7"):
            return False
        return True

    def call(self, role: str, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        start = time.time()

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if self._supports_temperature(self.model):
            kwargs["temperature"] = self.temperature

        for attempt in range(self.max_retries):
            try:
                message = client.messages.create(**kwargs)
                text = message.content[0].text
                elapsed = int((time.time() - start) * 1000)

                self.call_log.append(LLMCall(
                    role=role, model=self.model,
                    system_prompt=system_prompt[:200],
                    user_prompt=user_prompt[:200],
                    response=text[:500],
                    elapsed_ms=elapsed,
                    timestamp=time.time(),
                ))
                return text

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"Anthropic API failed after {self.max_retries} retries: {e}")


def create_backend(provider: str = "gemini", *, batch: bool = False, **kwargs) -> LLMBackend:
    """Factory for creating LLM backends.

    ``batch=True`` selects the Anthropic Message Batches API backend
    (50% cost, async, 24h SLA). Only supported for Anthropic.
    """
    if batch:
        if provider not in ("anthropic", "claude"):
            raise ValueError(
                f"batch=True is only supported for provider=anthropic/claude, "
                f"got provider={provider!r}"
            )
        from batch_backend import AnthropicBatchBackend
        return AnthropicBatchBackend(**kwargs)
    if provider == "gemini":
        return GeminiBackend(**kwargs)
    elif provider == "anthropic" or provider == "claude":
        return AnthropicBackend(**kwargs)
    elif provider == "mistral" or provider == "leanstral":
        return MistralBackend(**kwargs)
    elif provider == "mock":
        return MockBackend()
    else:
        raise ValueError(f"Unknown provider: {provider}")
