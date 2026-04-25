"""
TableBench Evaluation — FC + NR subset.

Adapts TableBench format (question → value answer) to EG-VAR pipeline
(claim → VERIFIED/REFUTED). The adapter converts each question+answer
into a verifiable claim: "the answer to [question] is [answer]".

The pipeline verifies whether the claim is VERIFIED by the table data.
If VERIFIED + kernel-checked → the answer is correct.
If REFUTED → the answer is wrong (should not happen if gold is correct).
If ABSTAIN → could not verify.

Usage:
  export ANTHROPIC_API_KEY=your_key
  python eval_tablebench.py --provider claude --model claude-sonnet-4-20250514 --n 50

  # Or with Gemini:
  export GEMINI_API_KEY=your_key
  python eval_tablebench.py --provider gemini --model gemini-2.5-pro --n 50
"""
from __future__ import annotations
import argparse
import json
import sys
import threading
import time
from pathlib import Path
from copy import deepcopy

from table_adapter import TableData, schema
from pipeline import run_pipeline, _call_analyst
from dictionary import init_core_dictionary, DictEntry, Alias
from llm_backend import create_backend, LLMBackend

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "tablebench"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results"


def load_tablebench_fc_nr(n: int | None = None, start: int = 0) -> list[dict]:
    """Load TableBench FC+NR subset."""
    with open(DATA_DIR / "fc_nr_subset.json") as f:
        data = json.load(f)
    if n is not None:
        data = data[start:start + n]
    else:
        data = data[start:]
    return data


def tablebench_to_tabledata(item: dict) -> TableData:
    """Convert TableBench table format to our TableData format.

    Canonicalizes column names (lower, strip, collapse whitespace
    including literal \\n / \\t) so the fixture's space-normalized
    forms match table headers when the source has embedded newlines.
    Fails fast on collisions — two raw headers that canonicalize to
    the same string would make lookups ambiguous.
    """
    from table_adapter import _canonicalize_col
    table = item["table"]
    raw_columns = table["columns"]
    columns = [_canonicalize_col(c) for c in raw_columns]
    seen: dict[str, int] = {}
    for i, c in enumerate(columns):
        if c in seen:
            raise ValueError(
                f"column collision after canonicalization in table "
                f"{item.get('id', 'unknown')!r}: "
                f"{raw_columns[seen[c]]!r} and {raw_columns[i]!r} both "
                f"canonicalize to {c!r}"
            )
        seen[c] = i
    rows = []
    for row in table["data"]:
        # Ensure row has same length as columns
        padded = row + [""] * (len(columns) - len(row))
        rows.append([str(v).strip() for v in padded[:len(columns)]])

    return TableData(
        table_id=item.get("id", "unknown"),
        caption=f"TableBench {item.get('qtype', '')} / {item.get('qsubtype', '')}",
        columns=columns,
        rows=rows,
    )


def extract_answer(evidence: list[dict], goal_type: str) -> str:
    """Extract the answer from evidence (certified by kernel).

    The answer depends on what predicates are in the evidence:
    - MeanWhere → the value IS the answer
    - SumWhere alone → the sum IS the answer
    - SumWhere + CountWhere → answer = sum/count (average)
    - CountWhere alone → the count IS the answer
    - MaxWhere/MinWhere → the value IS the answer
    """
    from fractions import Fraction

    mean_val = None
    sum_val = None
    count_val = None
    other_val = None

    for ev in evidence:
        wprop = ev.get("wprop", "")
        raw = ev.get("raw_value", "")
        try:
            frac = Fraction(raw)
        except (ValueError, TypeError):
            continue

        if "MeanWhere" in wprop:
            mean_val = frac
        elif "SumWhere" in wprop:
            sum_val = frac
        elif "CountWhere" in wprop:
            count_val = frac
        else:
            other_val = frac

    # MeanWhere is direct answer
    if mean_val is not None:
        return str(float(mean_val))

    # SumWhere + CountWhere = average
    if sum_val is not None and count_val is not None:
        return str(float(sum_val / count_val))

    # SumWhere alone = total
    if sum_val is not None:
        return str(float(sum_val))

    # CountWhere alone = count
    if count_val is not None:
        return str(float(count_val))

    # Fallback
    if other_val is not None:
        return str(float(other_val))

    return ""


def question_to_claim(question: str, answer: str = "") -> str:
    """Convert a question into a claim for the formalizer.

    The gold answer is NOT passed to the formalizer. The solver must discover
    the answer via Sigma-bound witnesses. Evaluation compares the discovered
    answer with the gold externally.
    """
    q = question.strip().rstrip("?").lower()
    return q


def run_baseline(question: str, table_item: dict, backend: LLMBackend) -> dict:
    """Bare LLM baseline: directly answer the question from the table."""
    table = table_item["table"]
    cols = " | ".join(table["columns"])
    rows_text = "\n".join(" | ".join(str(v) for v in row) for row in table["data"][:50])

    prompt = (
        f"Answer this question based on the table. Put your final answer inside \\boxed{{}}.\n\n"
        f"Table:\n{cols}\n{rows_text}\n\n"
        f"Question: {question}"
    )
    try:
        raw = backend.call("baseline", "You answer table questions. Output ONLY the answer value.", prompt)
        return {"answer": raw.strip(), "error": None}
    except Exception as e:
        return {"answer": "", "error": str(e)}


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison. Extract \\boxed{} content if present."""
    import re
    a = str(answer).strip()

    # Extract \boxed{...} content
    boxed = re.findall(r'\\boxed\{([^}]*)\}', a)
    if boxed:
        a = boxed[-1].strip()

    a = a.lower()
    # Remove common prefixes
    for prefix in ["the answer is ", "answer: ", "final answer: "]:
        if a.startswith(prefix):
            a = a[len(prefix):]
    a = a.strip().strip(".")
    return a


def answers_match(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold (flexible matching)."""
    p = normalize_answer(predicted)
    g = normalize_answer(gold)

    # Empty prediction never matches
    if not p:
        return False

    # Exact match
    if p == g:
        return True

    # Numeric match: round predicted to gold's decimal places
    # Handle fraction format (e.g., "463/9" from Sigma witness)
    from fractions import Fraction
    try:
        pn = float(Fraction(p.replace(",", "").replace("%", "")))
        gn = float(Fraction(g.replace(",", "").replace("%", "")))
        # Determine gold's decimal places
        gold_str = g.replace(",", "")
        dp = len(gold_str.split(".")[-1]) if "." in gold_str else 0
        # Round predicted to same dp (round-half-up, matching benchmark convention)
        import math
        if dp > 0:
            scale = 10 ** dp
            pn_rounded = math.floor(pn * scale + 0.5) / scale
        else:
            pn_rounded = math.floor(pn + 0.5)
        if pn_rounded == gn:
            return True
        # Fallback: within 1%
        if abs(pn - gn) < 0.01 * max(abs(gn), 1):
            return True
    except (ValueError, TypeError):
        pass

    # Substring match (for multi-part answers)
    if g in p or p in g:
        return True

    return False


def _normalize_model(model: str) -> str:
    return model.replace(".", "_").replace("-", "_")


def _build_analyst_cache_entry(table, backend_call):
    """Call the analyst once for a table; produce (table, guide, base_dict)
    for run_pipeline (variant A). Extracted from the outer loop so it can
    be reused by both the sync and batch code paths."""
    s = schema(table)
    guide, analyst_entries = _call_analyst(table, s, backend_call)
    d = init_core_dictionary()
    for entry_data in analyst_entries:
        try:
            aliases = [Alias(
                surface_form=a.get("surface_form", ""),
                context=a.get("context", ""),
                source=a.get("source", "analyst_proposed"),
                grade_impact=a.get("grade_impact", "no_cap"),
            ) for a in entry_data.get("aliases", [])]
            d.add_entry(DictEntry(
                name=entry_data.get("name", ""),
                type=entry_data.get("type", "Property"),
                description=entry_data.get("description", ""),
                formalization=entry_data.get("formalization", ""),
                aliases=aliases,
            ))
        except Exception as e:
            print(f"  [analyst entry FAILED] {entry_data.get('name','?')}: {e}",
                  file=sys.stderr)
    return (table, guide, d)


def _process_claim(
    *,
    i: int,
    item: dict,
    backend,
    args,
    gold_formalization,
    table_cache: dict | None,
    table_cache_lock=None,
) -> dict:
    """Per-claim body. Returns a single result dict.

    In sync mode, ``table_cache`` is the single-threaded cache passed from
    the outer loop. In batch mode, ``table_cache`` is None — each worker
    runs its own analyst call (Tier 1 fixtures have unique table_ids per
    claim, so no redundancy; for future TabFact we can add a pre-pass).

    Raises on catastrophic pipeline error; the caller is expected to catch
    and format an ERROR result."""
    from copy import deepcopy  # local to avoid circular imports
    question = item["question"]
    gold_answer = str(item["answer"])
    qtype = item["qtype"]
    qsubtype = item["qsubtype"]

    table = tablebench_to_tabledata(item)
    idx = args.start + i

    bl_answer_truncated = ""
    bl_correct = False
    if not args.skip_baseline:
        bl = run_baseline(question, item, backend)
        bl_correct = answers_match(bl["answer"], gold_answer)
        bl_answer_truncated = bl["answer"][:80]

    if args.baseline_only:
        return {
            "idx": idx, "question": question[:80],
            "gold": gold_answer, "qtype": qtype, "qsubtype": qsubtype,
            "baseline_answer": bl_answer_truncated, "baseline_correct": bl_correct,
        }

    claim = question_to_claim(question, gold_answer)

    try:
        if args.variant == "A":
            table_id = item.get("id", "unknown")
            if table_cache is not None:
                # sync path: serialized mutation
                if table_id not in table_cache:
                    table_cache[table_id] = _build_analyst_cache_entry(table, backend.call)
                _table, guide, base_dict = table_cache[table_id]
            else:
                # batch path: each worker runs its own analyst
                _table, guide, base_dict = _build_analyst_cache_entry(table, backend.call)
            pr = run_pipeline(
                claim, table, backend=backend,
                reading_guide=guide, dictionary=deepcopy(base_dict),
                n_formalizers=1, common_sense=False,
                variant="A",
                question=question,
                gold_formalization=gold_formalization,
            )
        else:
            pr = run_pipeline(
                claim, table, backend=backend,
                n_formalizers=1, common_sense=False,
                variant=args.variant,
                question=question,
                gold_formalization=gold_formalization,
            )
    except Exception as e:
        return {
            "idx": idx, "question": question[:80],
            "gold": gold_answer, "qtype": qtype, "qsubtype": qsubtype,
            "predicted": "ERROR", "error": str(e)[:300],
            "baseline_answer": bl_answer_truncated, "baseline_correct": bl_correct,
        }

    return {
        "idx": idx, "question": question[:80],
        "gold": gold_answer, "qtype": qtype, "qsubtype": qsubtype,
        "egvar_status": pr.status, "egvar_grade": pr.grade,
        "egvar_steps": len(pr.steps),
        "egvar_correct": pr.status == "VERIFIED",
        "egvar_answer": str(pr.answer),
        "egvar_details": pr.details,
        "baseline_answer": bl_answer_truncated,
        "baseline_correct": bl_correct,
    }


def _tally(r: dict, acc: dict) -> str:
    """Update summary counters from a result dict. Returns a short display
    tag for logging."""
    if "baseline_correct" in r and not acc.get("_no_baseline"):
        if r.get("baseline_correct"):
            acc["baseline_correct"] = acc.get("baseline_correct", 0) + 1
    if r.get("egvar_status") == "VERIFIED":
        acc["verified"] = acc.get("verified", 0) + 1
        gold = r.get("gold", "")
        if answers_match(str(r.get("egvar_answer", "")), gold):
            acc["correct_verified"] = acc.get("correct_verified", 0) + 1
            return "✓"
        return "✗(wrong)"
    if r.get("egvar_status") == "REFUTED":
        acc["verified"] = acc.get("verified", 0) + 1
        return "✗"
    if r.get("predicted") == "ERROR":
        acc["error"] = acc.get("error", 0) + 1
        return "!err"
    if "egvar_status" in r:
        acc["abstain"] = acc.get("abstain", 0) + 1
        return "○"
    return ""


def result_filename(
    *, egvar: bool, variant: str, model: str, total: int
) -> str:
    """Compute the result JSON filename.

    EG-VAR results carry a per-variant filename component
    (``tablebench_egvar_{variant}_{model}_{n}.json``) so running
    different pipeline variants on the same claim set does not
    overwrite each other. Baseline results stay variant-agnostic:
    ``tablebench_baseline_{model}_{n}.json``.
    """
    model_norm = _normalize_model(model)
    if egvar:
        return f"tablebench_egvar_{variant}_{model_norm}_{total}.json"
    return f"tablebench_baseline_{model_norm}_{total}.json"


def main():
    parser = argparse.ArgumentParser(description="EG-VAR TableBench Evaluation")
    parser.add_argument("--provider", default="claude", choices=["gemini", "claude", "anthropic", "mock"])
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--baseline-only", action="store_true", help="Run bare LLM baseline only")
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip the bare LLM baseline (useful for variant smoke/scale runs where "
             "baseline is run separately once with --baseline-only)",
    )
    parser.add_argument(
        "--variant", default="A", choices=["A", "B", "F"],
        help="Pipeline variant: A=analyst-driven (default), "
             "B=masked table, F=masked table + refinement retry.",
    )
    parser.add_argument("--subtypes", nargs="*", default=None, help="Filter by qsubtype")
    parser.add_argument(
        "--gold-formalization", type=str, default=None,
        help=(
            "Path to a JSON fixture list of (table_id, question, "
            "lean_goal_type) triplets. When a claim matches (by exact "
            "table_id AND question), the LLM formalizer is bypassed and "
            "the triplet's lean_goal_type is used instead. Strict: errors "
            "if any selected claim has no matching entry."
        ),
    )
    parser.add_argument(
        "--batch", action="store_true",
        help=(
            "Use Anthropic's Message Batches API (50%% cost, 24h SLA, "
            "typical minutes-to-hour). Claims run concurrently in a "
            "ThreadPoolExecutor; every time all live claims are blocked "
            "on an LLM call, the coordinator flushes one batch, waits, "
            "then wakes all workers. Anthropic provider only."
        ),
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=20,
        help=(
            "Max live claims in batch mode (default 20). Local work "
            "(tool execution, Lean typecheck) runs in parallel across "
            "these workers, so don't set higher than your CPU comfortably "
            "handles."
        ),
    )
    parser.add_argument(
        "--fixture-filter", action="store_true",
        help=(
            "Restrict the claim set to entries present in "
            "--gold-formalization. Useful for running only the curated "
            "Tier 1 subset against the full FC/NR dataset. Ignores "
            "--start/--n when enabled (those apply AFTER filtering, so "
            "'--n 12' takes the first 12 filtered claims)."
        ),
    )
    parser.add_argument(
        "--stride", type=int, default=None,
        help=(
            "If set (e.g., 10), take every Nth claim after loading / "
            "filtering. Combined with --fixture-filter --stride 10, "
            "picks 12 representative claims from a 120-entry fixture."
        ),
    )
    args = parser.parse_args()

    if args.baseline_only and args.skip_baseline:
        parser.error("--baseline-only and --skip-baseline are mutually exclusive")
    if args.baseline_only and args.gold_formalization:
        parser.error(
            "--baseline-only and --gold-formalization are mutually exclusive"
        )
    if args.batch and args.provider not in ("claude", "anthropic"):
        parser.error("--batch requires --provider claude (Anthropic Batches API)")
    if args.batch and not args.skip_baseline and not args.baseline_only:
        parser.error(
            "--batch requires --skip-baseline (the bare LLM baseline is "
            "a serial path; batching it is out of scope for this MVP)"
        )

    kwargs = {"model": args.model}
    if args.temperature is not None and args.provider != "mock":
        kwargs["temperature"] = args.temperature
    backend = create_backend(provider=args.provider, batch=args.batch, **kwargs)

    if args.fixture_filter:
        # Fixture-filter loads the full set then filters to fixture members,
        # then applies --stride (if any), then --start/--n.
        if not args.gold_formalization:
            parser.error("--fixture-filter requires --gold-formalization")
        all_claims = load_tablebench_fc_nr(None, 0)
        with open(args.gold_formalization) as gf:
            fixture_entries = json.load(gf)
        fixture_keys = {(e["table_id"], e["question"]) for e in fixture_entries}
        claims = [c for c in all_claims
                  if (c.get("id", "unknown"), c["question"]) in fixture_keys]
        if args.stride:
            claims = claims[::args.stride]
        claims = claims[args.start : args.start + args.n] if args.n else claims[args.start:]
    else:
        claims = load_tablebench_fc_nr(args.n, args.start)
        if args.stride:
            claims = claims[::args.stride]
    if args.subtypes:
        claims = [c for c in claims if c["qsubtype"] in args.subtypes]

    # Load + validate the gold formalization fixture (if any).
    # Strict coverage check runs after --subtypes filtering and
    # before any LLM call so missing entries surface upfront.
    gold_formalization = None
    if args.gold_formalization:
        from pipeline import validate_gold_fixture
        with open(args.gold_formalization) as gf:
            gold_formalization = json.load(gf)
        try:
            validate_gold_fixture(gold_formalization)
        except ValueError as e:
            parser.error(f"gold formalization fixture invalid: {e}")
        print(f"Gold formalization fixture: {args.gold_formalization} "
              f"({len(gold_formalization)} entries)")

        if not args.baseline_only:
            # Precompute entry set for O(1) coverage lookup
            have = {(e["table_id"], e["question"]) for e in gold_formalization}
            missing = [
                (args.start + i, item.get("id", "unknown"), item["question"])
                for i, item in enumerate(claims)
                if (item.get("id", "unknown"), item["question"]) not in have
            ]
            if missing:
                sys.stderr.write(
                    f"\n[gold formalization] {len(missing)} selected "
                    f"claim(s) have no fixture entry:\n"
                )
                for idx, tid, q in missing[:10]:
                    sys.stderr.write(f"  idx={idx} table_id={tid} q={q[:60]!r}\n")
                if len(missing) > 10:
                    sys.stderr.write(f"  ... and {len(missing) - 10} more\n")
                sys.stderr.write("Add these to the fixture or drop --gold-formalization.\n")
                sys.exit(2)

    print(f"Loaded {len(claims)} TableBench FC+NR claims (indices {args.start}-{args.start + len(claims) - 1})")
    print(f"Provider: {args.provider}, Model: {args.model}, Variant: {args.variant}")
    if args.skip_baseline and not args.baseline_only:
        print("  (--skip-baseline: bare LLM baseline will NOT be run)")
    if args.gold_formalization:
        print("  (--gold-formalization active: LLM formalizer bypassed)")

    # Per-claim JSONL sidecar (crash recovery / progress-streaming).
    # Written incrementally: one line per completed claim. The full
    # JSON at end-of-run remains the canonical summary.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = RESULTS_DIR / result_filename(
        egvar=not args.baseline_only,
        variant=args.variant,
        model=args.model,
        total=len(claims),
    ).replace(".json", ".jsonl")
    jsonl_lock = threading.Lock()
    jsonl_file = jsonl_path.open("w")

    acc: dict = {"verified": 0, "correct_verified": 0, "abstain": 0,
                 "baseline_correct": 0, "error": 0}
    results: list[dict] = []
    start_time = time.time()

    def _emit_result(i: int, item: dict, r: dict) -> None:
        """Record + log + flush one claim's result. Safe under lock."""
        tag = _tally(r, acc)
        results.append(r)
        with jsonl_lock:
            jsonl_file.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            jsonl_file.flush()
        status = r.get("egvar_status", "")
        grade = r.get("egvar_grade", "")
        steps = r.get("egvar_steps", 0)
        q = r.get("question", item.get("question", ""))[:60]
        qs = r.get("qsubtype", "")
        done = sum(1 for _ in results)  # len(results)
        print(f"[{done}/{len(claims)}] [{qs}] {q}", file=sys.stderr)
        if "error" in r:
            print(f"  ERROR: {r['error']}", file=sys.stderr)
        elif args.baseline_only:
            bl = "✓" if r.get("baseline_correct") else "✗"
            print(f"  Baseline: {bl} pred='{r.get('baseline_answer', '')[:40]}'",
                  file=sys.stderr)
        else:
            if not args.skip_baseline:
                bl = "✓" if r.get("baseline_correct") else "✗"
                print(f"  Baseline: {bl} pred='{r.get('baseline_answer', '')[:40]}'",
                      file=sys.stderr)
            print(f"  EG-VAR[{args.variant}]: {tag} status={status} grade={grade} "
                  f"steps={steps} gold={r.get('gold', '')} "
                  f"answer={r.get('egvar_answer', '')[:40]}",
                  file=sys.stderr)

    if args.batch:
        # Concurrent claim processing under the batch backend's
        # lockstep coordinator. Each worker wraps its run in
        # worker_session() so the coordinator knows the denominator.
        import concurrent.futures
        print(f"  (--batch active: {args.max_concurrent} concurrent claims via "
              f"Anthropic Message Batches API)", file=sys.stderr)

        def worker(i_item):
            i, item = i_item
            with backend.worker_session():
                try:
                    return i, item, _process_claim(
                        i=i, item=item, backend=backend, args=args,
                        gold_formalization=gold_formalization,
                        table_cache=None,  # no shared cache in batch mode
                    )
                except Exception as e:
                    return i, item, {
                        "idx": args.start + i,
                        "question": item.get("question", "")[:80],
                        "gold": str(item.get("answer", "")),
                        "qtype": item.get("qtype", ""),
                        "qsubtype": item.get("qsubtype", ""),
                        "predicted": "ERROR", "error": str(e)[:300],
                    }

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_concurrent
        ) as pool:
            futs = [pool.submit(worker, (i, it)) for i, it in enumerate(claims)]
            for f in concurrent.futures.as_completed(futs):
                i, item, r = f.result()
                _emit_result(i, item, r)
        backend.shutdown()
    else:
        table_cache: dict = {}
        for i, item in enumerate(claims):
            try:
                r = _process_claim(
                    i=i, item=item, backend=backend, args=args,
                    gold_formalization=gold_formalization,
                    table_cache=table_cache,
                )
            except Exception as e:
                r = {
                    "idx": args.start + i,
                    "question": item.get("question", "")[:80],
                    "gold": str(item.get("answer", "")),
                    "qtype": item.get("qtype", ""),
                    "qsubtype": item.get("qsubtype", ""),
                    "predicted": "ERROR", "error": str(e)[:300],
                }
            _emit_result(i, item, r)

    jsonl_file.close()
    elapsed = time.time() - start_time
    verified = acc["verified"]
    correct_verified = acc["correct_verified"]
    abstain = acc["abstain"]
    baseline_correct = acc["baseline_correct"]

    # Summary
    total = len(claims)
    print(f"\n{'='*60}")
    variant_tag = "" if args.baseline_only else f" variant={args.variant}"
    print(f"TableBench FC+NR Results ({args.model}{variant_tag})")
    print(f"{'='*60}")
    print(f"  Total claims: {total}")
    if total > 0 and not args.skip_baseline:
        print(f"  Baseline correct: {baseline_correct}/{total} ({baseline_correct/total:.0%})")
    if not args.baseline_only:
        wrong_verified = verified - correct_verified
        print(f"  EG-VAR verified: {verified}/{total}")
        print(f"  EG-VAR correct@verified: {correct_verified}/{verified}" if verified > 0 else "")
        print(f"  EG-VAR wrong@verified: {wrong_verified}/{verified}" if verified > 0 else "")
        print(f"  EG-VAR abstain: {abstain}/{total}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  LLM calls: {backend.get_stats()['total_calls']}")

    output = {
        "results": results,
        "model": args.model,
        "elapsed": elapsed,
        "total": total,
        "start": args.start,
        "n": args.n,
        "skip_baseline": args.skip_baseline,
    }
    if args.gold_formalization:
        output["gold_formalization"] = {
            "fixture_path": args.gold_formalization,
            "num_entries": len(gold_formalization) if gold_formalization else 0,
        }
    if not args.skip_baseline:
        output["baseline_correct"] = baseline_correct
    if not args.baseline_only:
        output["variant"] = args.variant
        output["verified"] = verified
        output["correct_verified"] = correct_verified
        output["abstain"] = abstain

    out_path = RESULTS_DIR / result_filename(
        egvar=not args.baseline_only,
        variant=args.variant,
        model=args.model,
        total=total,
    )
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
