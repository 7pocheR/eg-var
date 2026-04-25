#!/usr/bin/env python3
"""D.G — Real-LLM headline eval over the full Tier 1 120-claim fixture.

For each fixture entry:
  - Load TableBench table by table_id
  - Build the per-table tool menu from the table-formalizations file
  - Run the live pipeline with the configured model as the untrusted
    solver
  - Pass the full fixture as `gold_formalization` so the formalizer
    uses the curator's `lean_goal_type` (fixed-goal mode); the gold
    tool call and proof body remain validator-only
  - Capture status, grade, answer, steps, elapsed, error

Output: results/dg_eval_<timestamp>.jsonl with one line per entry.

Resume: if --out points to an existing file, skip entries already in
it.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = PROJECT_ROOT / "src" / "python" / "runtime"
TABLEBENCH_PATH = PROJECT_ROOT / "data" / "tablebench" / "TableBench.jsonl"
FIXTURE_PATH = PROJECT_ROOT / "data" / "tier1_gold_formalizations.json"
API_KEY_PATH = PROJECT_ROOT / "api_key.md"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(RUNTIME_DIR))


def load_api_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    if not API_KEY_PATH.exists():
        raise FileNotFoundError(f"{API_KEY_PATH} missing and ANTHROPIC_API_KEY not set")
    for line in API_KEY_PATH.read_text().splitlines():
        line = line.strip()
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1]
            return
    raise RuntimeError(f"No ANTHROPIC_API_KEY entry in {API_KEY_PATH}")


def load_table(table_id: str):
    """Load a TableBench entry as TableData."""
    from table_adapter import TableData
    with TABLEBENCH_PATH.open() as f:
        for line in f:
            if table_id not in line:
                continue
            e = json.loads(line)
            if e.get("id") != table_id:
                continue
            return TableData(
                table_id=table_id,
                columns=e["table"]["columns"],
                rows=[[str(v).strip() for v in row] for row in e["table"]["data"]],
            ), str(e.get("answer", ""))
    return None, None


def primary_predicate(goal: str) -> str:
    """Extract outermost L2 predicate name from a lean_goal_type."""
    PREDS = ("PProd", "MeanWhere", "SumWhere", "MaxWhere", "MinWhere",
             "ArgmaxWhere", "ArgminWhere", "CountWhere", "CountDistinctWhere",
             "TopKMeanWhere", "TopKSumWhere", "BottomKMeanWhere",
             "BottomKSumWhere", "HasProperty", "Relation", "EntitySum")
    for p in PREDS:
        if p in goal:
            return p
    return "OTHER"


def load_done(out_path: Path) -> set[int]:
    """Return set of fixture indices already evaluated in out_path."""
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open() as f:
        for line in f:
            try:
                e = json.loads(line)
                if "idx" in e:
                    done.add(int(e["idx"]))
            except json.JSONDecodeError:
                continue
    return done


def run_one(idx: int, entry: dict, fixture: list, backend,
            *, enforce_menu: bool = False, persist_steps: bool = False) -> dict:
    """Run pipeline for one fixture entry; return result dict.

    enforce_menu: when True, per-table menu enforcement gates stats /
      lookup / lookup_batch actions via
      tool_dispatch.is_action_in_per_table_menu.
    persist_steps: when True, include the full `steps` list in the
      returned dict. Tool-call steps carry `raw_action` (full solver
      JSON, including lookup_batch.lookups) and structured `outcome`;
      non-tool steps omit those fields.
    """
    import pipeline
    import re
    table_id = entry["table_id"]
    question = entry["question"]
    table, gold_answer = load_table(table_id)
    if table is None:
        return {"idx": idx, "table_id": table_id, "status": "ERROR",
                "error": "table not found"}

    t0 = time.time()
    try:
        from eval_tablebench import answers_match
        result = pipeline.run_pipeline(
            claim=question, table=table, backend=backend,
            n_formalizers=1, variant="A",
            gold_formalization=fixture, question=question,
            enforce_menu=enforce_menu,
        )
        elapsed = time.time() - t0
        # Resolve `obs<N>_val` references to the actual lean_value via
        # the evidence list (mirrors validate_fixture.py:281-289).
        ans = result.answer
        if ans and isinstance(ans, str):
            m = re.match(r'^obs(\d+)_val$', ans)
            if m:
                ev_idx = int(m.group(1))
                evidence_list = getattr(result, 'evidence', None) or []
                if 0 <= ev_idx < len(evidence_list):
                    lv = evidence_list[ev_idx].get("lean_value", "")
                    ans = pipeline._extract_sigma_witness(f"⟨{lv},?_⟩")
        # Use fuzzy compare from existing eval module
        ans_match = answers_match(ans or "", gold_answer or "") if (ans and gold_answer) else False
        out = {
            "idx": idx, "table_id": table_id, "question": question,
            "gold_answer": gold_answer,
            "status": result.status, "grade": str(result.grade),
            "answer": result.answer,
            "answer_resolved": ans,
            "answer_match": ans_match,
            "n_steps": len(result.steps),
            "elapsed_s": round(elapsed, 2),
            "predicate": primary_predicate(entry.get("lean_goal_type", "")),
            "has_pprod": "PProd" in entry.get("lean_goal_type", ""),
            "enforce_menu": enforce_menu,
        }
        if persist_steps:
            # Pass-through verbatim: tool-call steps carry raw_action +
            # outcome; non-tool steps omit them.
            out["steps"] = result.steps
        return out
    except Exception as e:
        elapsed = time.time() - t0
        return {"idx": idx, "table_id": table_id, "status": "EXCEPTION",
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(elapsed, 2),
                "enforce_menu": enforce_menu}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Run only the first N (post-start) entries.")
    ap.add_argument("--start", type=int, default=0,
                    help="Start at fixture index I.")
    ap.add_argument("--filter", default="",
                    help="Run only entries whose primary predicate contains this substring.")
    ap.add_argument("--out", default=None,
                    help="Output JSONL path (default: results/dg_eval_<ts>.jsonl).")
    ap.add_argument("--model", default="claude-sonnet-4-6",
                    help="Anthropic model id (default: claude-sonnet-4-6).")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true",
                    help="Skip indices already present in --out.")
    ap.add_argument("--enforce-menu", action="store_true",
                    help="Pass enforce_menu=True to pipeline.run_pipeline. "
                         "Off-menu stats/lookup/lookup_batch actions are rejected by "
                         "tool_dispatch.is_action_in_per_table_menu before reaching "
                         "the adapter.")
    ap.add_argument("--persist-steps", action="store_true",
                    help="Include the full `steps` list in each output line. "
                         "Tool-call steps carry raw_action (full JSON, including "
                         "lookup_batch.lookups) and structured outcome; non-tool "
                         "steps omit those fields.")
    args = ap.parse_args()

    load_api_key()
    from llm_backend import AnthropicBackend

    fixture = json.loads(FIXTURE_PATH.read_text())
    backend = AnthropicBackend(model=args.model, temperature=args.temperature)

    if args.out:
        out_path = Path(args.out)
    else:
        RESULTS_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = RESULTS_DIR / f"dg_eval_{ts}.jsonl"

    done = load_done(out_path) if args.resume else set()

    selected = []
    for i, entry in enumerate(fixture):
        if i < args.start:
            continue
        if args.resume and i in done:
            continue
        if args.filter and args.filter.lower() not in primary_predicate(
                entry.get("lean_goal_type", "")).lower():
            continue
        selected.append((i, entry))
        if args.limit is not None and len(selected) >= args.limit:
            break

    print(f"Running {len(selected)} entries → {out_path}", file=sys.stderr)
    n_ok = n_fail = n_err = 0
    n_verified = 0
    n_grade_verified = 0

    for j, (i, entry) in enumerate(selected):
        result = run_one(i, entry, fixture, backend,
                         enforce_menu=args.enforce_menu,
                         persist_steps=args.persist_steps)
        with out_path.open("a") as f:
            f.write(json.dumps(result) + "\n")
        status = result.get("status", "?")
        ans = result.get("answer", "?")
        gold = result.get("gold_answer", "?")
        match = result.get("answer_match", False)
        elapsed = result.get("elapsed_s", 0)
        marker = "✓" if match else "✗"
        print(f"  [{j+1:3d}/{len(selected)}] idx={i:3d} {marker} "
              f"status={status} grade={result.get('grade','?')} "
              f"ans={str(ans)[:30]!r} gold={str(gold)[:30]!r} "
              f"({elapsed:.1f}s)", file=sys.stderr)
        if status == "VERIFIED":
            n_verified += 1
        if str(result.get("grade", "")).endswith("Verified"):
            n_grade_verified += 1
        if match:
            n_ok += 1
        elif status == "EXCEPTION":
            n_err += 1
        else:
            n_fail += 1

    n = len(selected)
    if n == 0:
        print(f"\n=== D.G eval summary: NO ENTRIES SELECTED ===", file=sys.stderr)
        print(f"  output: {out_path}", file=sys.stderr)
        return 0
    print(f"\n=== D.G eval summary ({n} entries) ===", file=sys.stderr)
    print(f"  status=VERIFIED:        {n_verified} ({n_verified/n*100:.1f}%)",
          file=sys.stderr)
    print(f"  grade=Verified:         {n_grade_verified} ({n_grade_verified/n*100:.1f}%)",
          file=sys.stderr)
    print(f"  answer matches gold:    {n_ok} ({n_ok/n*100:.1f}%)", file=sys.stderr)
    print(f"  answer ≠ gold:          {n_fail}", file=sys.stderr)
    print(f"  exceptions:             {n_err}", file=sys.stderr)
    print(f"  output:                 {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
