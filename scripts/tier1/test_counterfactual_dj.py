#!/usr/bin/env python3
"""D.J - Single-shot table-only baseline on a counterfactual pilot.

Companion to test_counterfactual.py (D.I) and
test_counterfactual_dg.py (D.G). Renders the synthetic table inline
and asks the model to answer in one line. No tools, no kernel.
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
API_KEY_PATH = PROJECT_ROOT / "api_key.md"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(RUNTIME_DIR))


def load_api_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    if API_KEY_PATH.exists():
        for line in API_KEY_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1]
                return
    raise RuntimeError("ANTHROPIC_API_KEY missing")


def render_table(t):
    cols = t["table"]["columns"]
    rows = t["table"]["data"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def classify(ans, gold, world_prior):
    from eval_tablebench import answers_match
    if not ans or not ans.strip():
        return "abstain"
    if ans.strip().lower() in ("i don't know", "uncertain", "n/a", "unknown"):
        return "abstain"
    if answers_match(ans, gold or ""):
        return "source_faithful"
    if world_prior and answers_match(ans, world_prior):
        return "prior_override"
    return "wrong_other"


def run_one(claim, table_entry, backend):
    table_md = render_table(table_entry)
    prompt = (
        f"Below is a table. Answer the question about the table in ONE short line.\n"
        f"Output ONLY the bare answer (a number, name, yes/no, or short phrase). "
        f"Do not explain.\n\n"
        f"TABLE:\n{table_md}\n\n"
        f"QUESTION: {claim['question']}\n\n"
        f"ANSWER:"
    )
    t0 = time.time()
    try:
        text = backend.call(role="dj", system_prompt="", user_prompt=prompt)
        ans = (text or "").strip().split("\n")[0].strip()
        elapsed = time.time() - t0
    except Exception as e:
        return {
            "claim_id": claim["id"], "answer": None, "classification": f"exception: {type(e).__name__}",
            "elapsed_s": round(time.time() - t0, 2),
        }
    return {
        "claim_id": claim["id"],
        "question": claim["question"],
        "gold": claim["gold"],
        "answer": ans,
        "classification": classify(ans, claim["gold"], claim.get("world_prior")),
        "elapsed_s": round(elapsed, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    load_api_key()
    from llm_backend import AnthropicBackend
    backend = AnthropicBackend(model=args.model, temperature=args.temperature)

    pilot = json.loads(Path(args.pilot).read_text())
    tables = {t["id"]: t for t in pilot["tables"]}
    claims = pilot["claims"]

    if args.log:
        log_path = Path(args.log)
    else:
        RESULTS_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        log_path = RESULTS_DIR / f"counterfactual_dj_{ts}.tsv"

    with log_path.open("w") as f:
        f.write("ts\tclaim_id\trep\tquestion\tgold\tans\tclassification\telapsed_s\n")

    print(f"D.J counterfactual: {len(claims)} claims × {args.repeats} reps "
          f"on {args.model} (temp={args.temperature})", file=sys.stderr)

    by_class = Counter()
    for c in claims:
        t_entry = tables[c["table_ref"]]
        case_classes = []
        for rep in range(args.repeats):
            r = run_one(c, t_entry, backend)
            with log_path.open("a") as f:
                f.write(f"{int(time.time())}\t{r['claim_id']}\t{rep}\t{r.get('question','')}\t"
                        f"{r.get('gold','')}\t{r.get('answer','')}\t{r['classification']}\t"
                        f"{r['elapsed_s']}\n")
            case_classes.append(r["classification"])
            by_class[r["classification"]] += 1
        case_counter = Counter(case_classes)
        summary = ", ".join(f"{c}={n}" for c, n in case_counter.most_common())
        print(f"  [{c}] {summary}", file=sys.stderr) if False else \
            print(f"  [{r['claim_id'].rsplit('_',1)[0]}] {summary}", file=sys.stderr)

    print(f"\n=== Source-faithfulness summary "
          f"({args.repeats}×{len(claims)}={args.repeats*len(claims)} runs) ===",
          file=sys.stderr)
    total = sum(by_class.values())
    for c, n in by_class.most_common():
        print(f"  {c:18s}: {n}/{total} ({100*n/max(total,1):.1f}%)", file=sys.stderr)
    print(f"  Log: {log_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
