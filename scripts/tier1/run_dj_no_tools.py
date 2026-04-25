#!/usr/bin/env python3
"""D.J - Weak baseline: LLM with table only, NO tools.

Bottom rung of the baseline ladder. Each claim gets a single-shot
prompt: question + table + 'answer in one line'. The model has no
tools, no kernel, no formalization; it must compute manually from the
visible table.

The headline gap from D.J -> D.I-loose -> D.I-strict -> D.G isolates
the contributions of tools, per-table commitments, and kernel.

Output: results/dj_<ts>.jsonl with one line per fixture entry. Schema
mirrors run_di_baseline.py minus the tool-related fields.
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


SYS_PROMPT = """You are a question-answering agent for tabular data. \
Read the table and answer the question.

ANSWER FORMAT (load-bearing — the judge is strict):
Output ONLY the bare answer:
  - For numeric answers: just the number, no commas, no units.
    e.g. `4351` (NOT `"4,351 thousand"`).
  - For entity/text answers: just the entity name as it appears in the table.
    e.g. `oman` (NOT `"Oman has the highest..."`).
  - For date/time answers: the date or time as it appears in the table.

Do NOT wrap the answer in a sentence. Do NOT add explanations.
Surrounding prose will be marked WRONG even when your underlying answer
is right. Output ONLY the answer string, nothing else."""


def build_user_prompt(claim: str, table_text: str) -> str:
    return (
        f"QUESTION: {claim}\n\n"
        f"TABLE:\n{table_text}\n\n"
        f"Answer:"
    )


def run_one(idx: int, entry: dict, backend) -> dict:
    """Single-shot prompt — no tools, no loop."""
    from pipeline import _render_table
    from eval_tablebench import answers_match

    table_id = entry["table_id"]
    question = entry["question"]
    table, gold_answer = load_table_and_gold(table_id)
    if table is None:
        return {"idx": idx, "table_id": table_id, "status": "ERROR",
                "error": "table not found"}

    table_text = _render_table(table)
    user = build_user_prompt(question, table_text)

    t0 = time.time()
    try:
        answer = backend.call("solver", SYS_PROMPT, user).strip()
    except Exception as e:
        return {"idx": idx, "table_id": table_id, "status": "EXCEPTION",
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.time() - t0, 2)}
    elapsed = round(time.time() - t0, 2)

    ans_match = answers_match(answer or "", gold_answer or "") if (answer and gold_answer) else False
    return {
        "idx": idx, "table_id": table_id, "question": question,
        "gold_answer": gold_answer,
        "answer": answer,
        "answer_match": ans_match,
        "elapsed_s": elapsed,
        "predicate": primary_predicate(entry.get("lean_goal_type", "")),
        "has_pprod": "PProd" in entry.get("lean_goal_type", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--filter", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
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
        out_path = RESULTS_DIR / f"dj_{ts}.jsonl"

    done = set()
    if args.resume and out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if "idx" in e:
                        done.add(int(e["idx"]))
                except json.JSONDecodeError:
                    continue

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

    print(f"D.J no-tools baseline: {len(selected)} entries → {out_path}", file=sys.stderr)
    t0 = time.time()
    n_match = 0
    for j, (i, entry) in enumerate(selected):
        result = run_one(i, entry, backend)
        with out_path.open("a") as f:
            f.write(json.dumps(result) + "\n")
        ans = result.get("answer", "?")
        gold = result.get("gold_answer", "?")
        match = result.get("answer_match", False)
        elapsed = result.get("elapsed_s", 0)
        marker = "✓" if match else "✗"
        if match:
            n_match += 1
        print(f"  [{j+1:3d}/{len(selected)}] idx={i:3d} {marker} "
              f"ans={str(ans)[:25]!r:30s} gold={str(gold)[:20]!r} "
              f"({elapsed:.1f}s)", file=sys.stderr)

    total = time.time() - t0
    n = len(selected)
    print(f"\n=== D.J headline ({n} entries, {total:.1f}s wall) ===", file=sys.stderr)
    print(f"  answer_match: {n_match}/{n} ({100*n_match/max(n,1):.1f}%)", file=sys.stderr)
    print(f"  output:       {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
