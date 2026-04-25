#!/usr/bin/env python3
"""Run counterfactual pilot claims through D.I-strict (and optionally
D.G).

Each claim references a synthetic table embedded in the pilot JSON
file. The synthetic table is constructed in-memory and passed
directly to the agent (the TableBench file and per-table
formalization file are not modified).

Tests whether D.I-strict trusts the (counterfactual) table value or
overrides with its world-knowledge prior. D.G via kernel + tool
attestation must trust the table by construction.

Usage:
  python3 scripts/tier1/test_counterfactual.py \
    --pilot data/tier1_counterfactual_pilot.json \
    --variant strict --repeats 5 --temp 0
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = PROJECT_ROOT / "src" / "python" / "runtime"
PER_TABLE_PATH = PROJECT_ROOT / "data" / "tier1_table_formalizations.json"
API_KEY_PATH = PROJECT_ROOT / "api_key.md"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(RUNTIME_DIR))


def load_api_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    if not API_KEY_PATH.exists():
        raise FileNotFoundError(f"{API_KEY_PATH} missing")
    for line in API_KEY_PATH.read_text().splitlines():
        line = line.strip()
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1]
            return
    raise RuntimeError("No ANTHROPIC_API_KEY in api_key.md")


def make_table_data(t_entry):
    """Build a TableData from a counterfactual pilot table entry."""
    from table_adapter import TableData
    # Use original_id so per-table formalization lookup hits the
    # original table's tools/lifts; the synthetic table shares column
    # structure with the original.
    return TableData(
        table_id=t_entry["original_id"],
        columns=t_entry["table"]["columns"],
        rows=[[str(v).strip() for v in row] for row in t_entry["table"]["data"]],
    )


def classify_answer(ans: str, gold: str, world_prior_value: str | None = None) -> str:
    """Classify answer as source-faithful / prior-override / abstain / wrong-other.

    `world_prior_value` is what we'd expect if the model overrode the
    table with its world knowledge. None if not provided.
    """
    from eval_tablebench import answers_match
    if not ans or not ans.strip():
        return "abstain"
    if ans.strip().lower() in ("i don't know", "uncertain", "n/a", "unknown"):
        return "abstain"
    if answers_match(ans, gold or ""):
        return "source_faithful"
    if world_prior_value and answers_match(ans, world_prior_value):
        return "prior_override"
    return "wrong_other"


def test_claim(claim, table_entry, backend, enforce_menu: bool = True):
    from baseline_agent import run_baseline_agent
    table = make_table_data(table_entry)
    pt = json.loads(PER_TABLE_PATH.read_text())
    tf = pt.get(table.table_id)

    t0 = time.time()
    result = run_baseline_agent(
        claim=claim["question"], table=table, tf=tf, backend=backend,
        enforce_menu=enforce_menu, max_steps=20,
    )
    elapsed = time.time() - t0

    # Use world_prior_value from claim if provided (v2 spec)
    world_prior = claim.get("world_prior")
    classification = classify_answer(result.answer or "", claim["gold"],
                                      world_prior_value=world_prior)
    return {
        "claim_id": claim["id"],
        "question": claim["question"],
        "gold": claim["gold"],
        "answer": result.answer,
        "classification": classification,
        "n_tool_calls": result.n_tool_calls,
        "n_steps": len(result.steps),
        "answered_without_tool": result.answered_without_tool,
        "terminated": result.terminated_reason,
        "elapsed_s": round(elapsed, 2),
        "tool_actions": [s.get("raw_action") for s in result.steps if "raw_action" in s],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True,
                    help="JSON file with `tables` and `claims` (counterfactual pilot)")
    ap.add_argument("--variant", choices=["strict", "loose"], default="strict")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    load_api_key()
    from llm_backend import AnthropicBackend
    backend = AnthropicBackend(model=args.model, temperature=args.temperature)
    enforce = (args.variant == "strict")

    pilot = json.loads(Path(args.pilot).read_text())
    tables = {t["id"]: t for t in pilot["tables"]}
    claims = pilot["claims"]

    if args.log:
        log_path = Path(args.log)
    else:
        RESULTS_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        log_path = RESULTS_DIR / f"counterfactual_{args.variant}_{ts}.tsv"

    with log_path.open("w") as f:
        f.write("ts\tclaim_id\trep\tquestion\tgold\tans\tclassification\tn_tools\telapsed_s\n")

    print(f"Counterfactual pilot: {len(claims)} claims × {args.repeats} reps "
          f"on {args.model} D.I-{args.variant} (temp={args.temperature})",
          file=sys.stderr)

    by_class = Counter()
    for claim in claims:
        t_entry = tables[claim["table_ref"]]
        case_classes = []
        for rep in range(args.repeats):
            r = test_claim(claim, t_entry, backend, enforce_menu=enforce)
            with log_path.open("a") as f:
                f.write(f"{int(time.time())}\t{r['claim_id']}\t{rep}\t{r['question']}\t"
                        f"{r['gold']}\t{r['answer']}\t{r['classification']}\t"
                        f"{r['n_tool_calls']}\t{r['elapsed_s']}\n")
            case_classes.append(r["classification"])
            by_class[r["classification"]] += 1
        # Per-claim summary
        case_counter = Counter(case_classes)
        summary = ", ".join(f"{c}={n}" for c, n in case_counter.most_common())
        print(f"  [{claim['id']}] {summary}", file=sys.stderr)

    print(f"\n=== Source-faithfulness summary ({args.repeats}×{len(claims)}={args.repeats*len(claims)} runs) ===",
          file=sys.stderr)
    total = sum(by_class.values())
    for c in ("source_faithful", "prior_override", "wrong_other", "abstain"):
        n = by_class.get(c, 0)
        print(f"  {c:18s}: {n}/{total} ({100*n/max(total,1):.1f}%)", file=sys.stderr)
    print(f"  Log: {log_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
