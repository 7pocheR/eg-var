#!/usr/bin/env python3
"""Baseline runner for D.I-strict / D.I-loose.

Mirrors run_dg_eval.py argument surface; --variant selects the rung:
  --variant strict - per-table menu enforcement ON (apples-to-apples
                     comparison vs D.G under enforcement).
  --variant loose  - full runtime tool surface.

Output: results/di_<variant>_<ts>.jsonl with one line per fixture
entry. Schema mirrors run_dg_eval.py plus baseline diagnostic buckets
(n_off_menu_calls, n_invalid_json, n_unknown_actions, n_tool_errors,
answered_without_tool, last_observation, final_action,
terminated_reason).
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
PER_TABLE_PATH = PROJECT_ROOT / "data" / "tier1_table_formalizations.json"
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


def load_table_and_gold(table_id: str):
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
    PREDS = ("PProd", "MeanWhere", "SumWhere", "MaxWhere", "MinWhere",
             "ArgmaxWhere", "ArgminWhere", "CountWhere", "CountDistinctWhere",
             "TopKMeanWhere", "TopKSumWhere", "BottomKMeanWhere",
             "BottomKSumWhere", "HasProperty", "Relation", "EntitySum")
    for p in PREDS:
        if p in goal:
            return p
    return "OTHER"


def load_done(out_path: Path) -> set[int]:
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


def run_one(idx: int, entry: dict, pt: dict, backend, *, enforce_menu: bool,
            persist_steps: bool = True) -> dict:
    """Run baseline_agent for one fixture entry."""
    from baseline_agent import run_baseline_agent
    from eval_tablebench import answers_match

    table_id = entry["table_id"]
    question = entry["question"]
    table, gold_answer = load_table_and_gold(table_id)
    if table is None:
        return {"idx": idx, "table_id": table_id, "status": "ERROR",
                "error": "table not found"}

    tf = pt.get(table_id)
    try:
        result = run_baseline_agent(
            claim=question, table=table, tf=tf, backend=backend,
            enforce_menu=enforce_menu,
        )
    except Exception as e:
        return {"idx": idx, "table_id": table_id, "status": "EXCEPTION",
                "error": f"{type(e).__name__}: {e}"}

    ans = result.answer
    ans_match = answers_match(ans or "", gold_answer or "") if (ans and gold_answer) else False
    # Schema parity with run_dg_eval.py: map baseline-specific
    # terminated_reason to a status string from
    # {ANSWERED, GAVE_UP, EXCEPTION} that mirrors D.G's status field.
    # grade is None (no kernel verdict); answer_resolved mirrors
    # answer (no Sigma-witness resolution needed for baselines).
    if result.error:
        status = "EXCEPTION"
    elif result.terminated_reason == "answered":
        status = "ANSWERED"
    else:
        status = "GAVE_UP"
    out = {
        "idx": idx, "table_id": table_id, "question": question,
        "gold_answer": gold_answer,
        "status": status,
        "grade": None,                 # baseline has no kernel grade
        "answer": result.answer,
        "answer_resolved": result.answer,  # no Sigma-witness layer
        "answer_match": ans_match,
        "n_steps": len(result.steps),
        "n_tool_calls": result.n_tool_calls,
        "elapsed_s": result.elapsed_s,
        "predicate": primary_predicate(entry.get("lean_goal_type", "")),
        "has_pprod": "PProd" in entry.get("lean_goal_type", ""),
        "enforce_menu": result.enforce_menu,
        "terminated_reason": result.terminated_reason,
        # Diagnostic buckets
        "n_off_menu_calls": result.n_off_menu_calls,
        "n_invalid_json": result.n_invalid_json,
        "n_unknown_actions": result.n_unknown_actions,
        "n_tool_errors": result.n_tool_errors,
        "answered_without_tool": result.answered_without_tool,
        "last_observation": result.last_observation,
        "final_action": result.final_action,
        "error": result.error,
    }
    if persist_steps:
        out["steps"] = result.steps
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["strict", "loose"], required=True,
                    help="strict: per-table menu enforcement ON. loose: full runtime tool surface.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Run only the first N entries (post-start).")
    ap.add_argument("--start", type=int, default=0,
                    help="Start at fixture index I.")
    ap.add_argument("--filter", default="",
                    help="Run only entries whose primary predicate contains this substring.")
    ap.add_argument("--out", default=None,
                    help="Output JSONL path (default: results/di_<variant>_<ts>.jsonl).")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true",
                    help="Skip indices already present in --out.")
    ap.add_argument("--no-persist-steps", action="store_true",
                    help="Disable per-step trace persistence (smaller output, less diagnostic).")
    args = ap.parse_args()

    load_api_key()
    from llm_backend import AnthropicBackend

    fixture = json.loads(FIXTURE_PATH.read_text())
    pt = json.loads(PER_TABLE_PATH.read_text())
    backend = AnthropicBackend(model=args.model, temperature=args.temperature)
    enforce_menu = (args.variant == "strict")
    persist_steps = not args.no_persist_steps

    if args.out:
        out_path = Path(args.out)
    else:
        RESULTS_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = RESULTS_DIR / f"di_{args.variant}_{ts}.jsonl"

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

    print(f"D.I-{args.variant} baseline: {len(selected)} entries → {out_path}",
          file=sys.stderr)
    print(f"  enforce_menu={enforce_menu}, persist_steps={persist_steps}", file=sys.stderr)
    t0 = time.time()
    n_match = 0
    n_off_menu_total = 0
    n_no_tool = 0

    for j, (i, entry) in enumerate(selected):
        result = run_one(i, entry, pt, backend,
                         enforce_menu=enforce_menu,
                         persist_steps=persist_steps)
        with out_path.open("a") as f:
            f.write(json.dumps(result) + "\n")
        ans = result.get("answer", "?")
        gold = result.get("gold_answer", "?")
        match = result.get("answer_match", False)
        elapsed = result.get("elapsed_s", 0)
        n_tools = result.get("n_tool_calls", 0)
        marker = "✓" if match else "✗"
        if match:
            n_match += 1
        n_off_menu_total += result.get("n_off_menu_calls", 0)
        if result.get("answered_without_tool"):
            n_no_tool += 1
        term = result.get("terminated_reason", "?")[:6]
        print(f"  [{j+1:3d}/{len(selected)}] idx={i:3d} {marker} "
              f"ans={str(ans)[:25]!r:30s} gold={str(gold)[:20]!r} "
              f"tools={n_tools} term={term} ({elapsed:.1f}s)", file=sys.stderr)

    total = time.time() - t0
    n = len(selected)
    print(f"\n=== D.I-{args.variant} headline ({n} entries, {total:.1f}s wall) ===",
          file=sys.stderr)
    print(f"  answer_match:           {n_match}/{n} ({100*n_match/max(n,1):.1f}%)",
          file=sys.stderr)
    print(f"  off_menu rejections:    {n_off_menu_total} (total)",
          file=sys.stderr)
    print(f"  answered_without_tool:  {n_no_tool}/{n} ({100*n_no_tool/max(n,1):.1f}%)",
          file=sys.stderr)
    print(f"  output:                 {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
