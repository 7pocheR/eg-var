#!/usr/bin/env python3
"""Per-claim CSV diff across the baseline ladder.

Reads the 4 result JSONLs (D.G, D.I-strict, D.I-loose, D.J) and
produces a single per-claim diff CSV with one row per fixture index,
columns covering each rung's outcome.

Usage:
  python3 scripts/tier1/build_baseline_ladder_csv.py \
    --dg results/dg_eval_enforced.jsonl \
    --di-strict results/di_strict.jsonl \
    --di-loose results/di_loose.jsonl \
    --dj results/dj.jsonl \
    --out results/baseline_ladder_per_claim.csv

Any --rung argument may be omitted; the corresponding columns are empty.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"


def load_jsonl(path: str | None) -> dict:
    """Return {idx: row} for a JSONL result file, or empty dict if path is None."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"WARN: {p} not found; skipping", file=sys.stderr)
        return {}
    out = {}
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                if "idx" in e:
                    out[int(e["idx"])] = e
            except json.JSONDecodeError as ex:
                print(f"WARN: malformed line in {p}: {ex}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dg", default=None,
                    help="D.G (EG-VAR with kernel) JSONL.")
    ap.add_argument("--di-strict", default=None,
                    help="D.I-strict (kernel-off, menu enforced) JSONL.")
    ap.add_argument("--di-loose", default=None,
                    help="D.I-loose (kernel-off, full surface) JSONL.")
    ap.add_argument("--dj", default=None,
                    help="D.J (no tools) JSONL.")
    ap.add_argument("--fixture", default=str(PROJECT_ROOT / "data" / "tier1_gold_formalizations.json"),
                    help="Fixture JSON for question + lean_goal_type metadata.")
    ap.add_argument("--out", default=str(RESULTS_DIR / "baseline_ladder_per_claim.csv"),
                    help="Output CSV path.")
    args = ap.parse_args()

    fixture = json.loads(Path(args.fixture).read_text())
    dg = load_jsonl(args.dg)
    di_strict = load_jsonl(args.di_strict)
    di_loose = load_jsonl(args.di_loose)
    dj = load_jsonl(args.dj)

    # Determine which idxs to include — union of all loaded sets, sorted.
    idxs = sorted(set(dg) | set(di_strict) | set(di_loose) | set(dj) |
                  set(range(len(fixture))))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Predicate from fixture (for slicing analysis)
    PREDS = ("PProd", "MeanWhere", "SumWhere", "MaxWhere", "MinWhere",
             "ArgmaxWhere", "ArgminWhere", "CountWhere", "CountDistinctWhere",
             "TopKMeanWhere", "TopKSumWhere", "BottomKMeanWhere",
             "BottomKSumWhere", "HasProperty", "Relation", "EntitySum")

    def primary_predicate(goal: str) -> str:
        for p in PREDS:
            if p in goal:
                return p
        return "OTHER"

    fieldnames = [
        "idx", "table_id", "question", "gold_answer", "predicate", "has_pprod",
        # D.G (EG-VAR)
        "dg_status", "dg_grade", "dg_answer", "dg_match", "dg_n_steps", "dg_elapsed_s",
        # D.I-strict
        "di_strict_answer", "di_strict_match", "di_strict_n_tool_calls",
        "di_strict_n_off_menu_calls", "di_strict_terminated",
        "di_strict_answered_without_tool",
        # D.I-loose
        "di_loose_answer", "di_loose_match", "di_loose_n_tool_calls",
        "di_loose_terminated", "di_loose_answered_without_tool",
        # D.J
        "dj_answer", "dj_match",
        # Headline diff (EG-VAR-only, baseline-only, etc.)
        "diff_class",
    ]

    rows = []
    counts: Counter = Counter()
    for idx in idxs:
        if idx >= len(fixture):
            continue
        e = fixture[idx]
        tid = e["table_id"]
        question = e["question"]
        # Try to recover gold_answer from any rung that has it
        gold = (dg.get(idx, {}).get("gold_answer")
                or di_strict.get(idx, {}).get("gold_answer")
                or di_loose.get(idx, {}).get("gold_answer")
                or dj.get(idx, {}).get("gold_answer")
                or "")
        predicate = primary_predicate(e.get("lean_goal_type", ""))
        has_pprod = "PProd" in e.get("lean_goal_type", "")

        d_dg = dg.get(idx, {})
        d_dis = di_strict.get(idx, {})
        d_dil = di_loose.get(idx, {})
        d_dj = dj.get(idx, {})

        m_dg = d_dg.get("answer_match", None)
        m_dis = d_dis.get("answer_match", None)
        m_dil = d_dil.get("answer_match", None)
        m_dj = d_dj.get("answer_match", None)

        # Headline diff class
        if m_dg and not m_dis and not m_dil and not m_dj:
            cls = "DG_ONLY_WIN"
        elif m_dg and m_dis and not m_dj:
            cls = "DG_DIS_WIN"
        elif m_dj and not m_dg:
            cls = "DJ_WIN_DG_LOST"  # rare
        elif all(v is True for v in (m_dg, m_dis, m_dil, m_dj) if v is not None):
            cls = "ALL_CORRECT"
        elif all(v is False or v is None for v in (m_dg, m_dis, m_dil, m_dj)):
            cls = "ALL_FAIL"
        elif m_dis and not m_dil:
            cls = "DIS_BEATS_LOOSE"
        elif m_dil and not m_dis:
            cls = "LOOSE_BEATS_DIS"
        else:
            cls = "MIXED"
        counts[cls] += 1

        rows.append({
            "idx": idx, "table_id": tid, "question": question,
            "gold_answer": gold, "predicate": predicate, "has_pprod": has_pprod,
            "dg_status": d_dg.get("status", ""),
            "dg_grade": d_dg.get("grade", ""),
            "dg_answer": d_dg.get("answer", ""),
            "dg_match": m_dg if m_dg is not None else "",
            "dg_n_steps": d_dg.get("n_steps", ""),
            "dg_elapsed_s": d_dg.get("elapsed_s", ""),
            "di_strict_answer": d_dis.get("answer", ""),
            "di_strict_match": m_dis if m_dis is not None else "",
            "di_strict_n_tool_calls": d_dis.get("n_tool_calls", ""),
            "di_strict_n_off_menu_calls": d_dis.get("n_off_menu_calls", ""),
            "di_strict_terminated": d_dis.get("terminated_reason", ""),
            "di_strict_answered_without_tool": d_dis.get("answered_without_tool", ""),
            "di_loose_answer": d_dil.get("answer", ""),
            "di_loose_match": m_dil if m_dil is not None else "",
            "di_loose_n_tool_calls": d_dil.get("n_tool_calls", ""),
            "di_loose_terminated": d_dil.get("terminated_reason", ""),
            "di_loose_answered_without_tool": d_dil.get("answered_without_tool", ""),
            "dj_answer": d_dj.get("answer", ""),
            "dj_match": m_dj if m_dj is not None else "",
            "diff_class": cls,
        })

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Top-line summary
    n = len(rows)
    print(f"\n=== Baseline ladder per-claim CSV ===", file=sys.stderr)
    print(f"  Output: {out_path}  ({n} rows)", file=sys.stderr)
    print(f"  Sources:", file=sys.stderr)
    print(f"    D.G:        {args.dg or '(none)'} ({len(dg)} entries)", file=sys.stderr)
    print(f"    D.I-strict: {args.di_strict or '(none)'} ({len(di_strict)} entries)", file=sys.stderr)
    print(f"    D.I-loose:  {args.di_loose or '(none)'} ({len(di_loose)} entries)", file=sys.stderr)
    print(f"    D.J:        {args.dj or '(none)'} ({len(dj)} entries)", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"  diff_class distribution:", file=sys.stderr)
    for cls, c in counts.most_common():
        print(f"    {cls}: {c}", file=sys.stderr)

    # Per-rung answer_match tallies (only for rungs with data)
    print(f"", file=sys.stderr)
    print(f"  answer_match per rung:", file=sys.stderr)
    for label, src in [("D.G", dg), ("D.I-strict", di_strict),
                       ("D.I-loose", di_loose), ("D.J", dj)]:
        if not src:
            continue
        nm = sum(1 for v in src.values() if v.get("answer_match"))
        ns = len(src)
        print(f"    {label}: {nm}/{ns} ({100*nm/max(ns,1):.1f}%)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
