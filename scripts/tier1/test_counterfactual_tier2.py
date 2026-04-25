#!/usr/bin/env python3
"""Tier 2 evaluation on counterfactual claims.

Runs the full EG-VAR pipeline (analyst -> formalizer -> solver ->
kernel) without gold formalization, on counterfactual tables from
the pilot fixtures. Measures whether the source-faithfulness
property of D.G extends to the deployment-time LLM-formalizer
regime.

Shares ``make_table_data``, ``classify_dg``, and the TSV logging
convention with ``test_counterfactual_dg.py`` so results are
directly comparable.
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
sys.path.insert(0, str(RUNTIME_DIR))

# Import helpers from the D.G sibling runner for table-build + classification
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tier1"))
from test_counterfactual_dg import (  # type: ignore
    make_table_data, classify_dg, load_api_key
)


def shape_of(claim_id: str) -> str:
    parts = claim_id.split("_")
    return parts[-1] if parts else "?"


def find_shape_d_postproc_entry(claim_id: str, shape: str, pilot: dict) -> dict | None:
    """Look up shape-D post-processor metadata if this is a shape-D
    claim.

    Tier 2 doesn't load the D.G fixture, but shape-D claims still
    need the (subject_entity, direction_if_subject_wins/loses)
    metadata to post-process the winning-entity witness into a
    direction word. Read from the D.G fixture file co-located with
    the pilot; if unavailable, shape-D classification falls back to
    'verified_other'.
    """
    if shape != "D":
        return None
    # Guess the D.G fixture path from the pilot's panel marker
    pilot_path = pilot.get("_pilot_path")
    if not pilot_path:
        return None
    name = os.path.basename(pilot_path)
    if name.endswith("_v4.json"):
        dg_path = PROJECT_ROOT / "data" / "tier1_counterfactual_dg_fixtures_v4.json"
    elif name.endswith("_v45.json"):
        dg_path = PROJECT_ROOT / "data" / "tier1_counterfactual_dg_fixtures_v45.json"
    else:
        return None
    if not dg_path.exists():
        return None
    d = json.load(open(dg_path))
    for e in d.get("fixtures", []):
        if e.get("_claim_id") == claim_id:
            return e
    return None


def run_one(claim_entry: dict, table_entry: dict, backend, model: str,
            temperature: float, rep: int, pilot: dict) -> dict:
    """Run one Tier 2 evaluation on a counterfactual claim."""
    from pipeline import run_pipeline
    table = make_table_data(table_entry)
    question = claim_entry["question"]
    gold = claim_entry["gold"]
    world_prior = claim_entry.get("world_prior")
    claim_id = claim_entry.get("id", "")
    shape = shape_of(claim_id)

    t0 = time.time()
    try:
        result = run_pipeline(
            claim=question, table=table, backend=backend,
            n_formalizers=1, variant="A",
            # No gold_formalization: LLM formalizer runs end-to-end.
            enforce_menu=True,
            menu_aware=True,  # per-table menu in formalizer prompt
        )
        elapsed = time.time() - t0

        # Resolve obs<N>_val pattern.
        import re
        ans = result.answer
        if ans and isinstance(ans, str):
            m = re.match(r'^obs(\d+)_val$', ans)
            if m:
                ev_idx = int(m.group(1))
                evidence_list = getattr(result, 'evidence', None) or []
                if 0 <= ev_idx < len(evidence_list):
                    import pipeline
                    lv = evidence_list[ev_idx].get("lean_value", "")
                    ans = pipeline._extract_sigma_witness(f"\u27e8{lv},?_\u27e9")

        # Build classifier input; shape-D post-processor metadata
        # comes from the D.G fixture.
        fix_entry = find_shape_d_postproc_entry(claim_id, shape, pilot)

        class _R:
            pass
        R = _R()
        R.status = result.status
        R.grade = result.grade
        R.answer = ans
        classification = classify_dg(R, gold, world_prior, fix_entry=fix_entry)

        return {
            "claim_id": claim_id, "rep": rep, "shape": shape,
            "question": question, "gold": gold, "world_prior": world_prior,
            "answer_raw": result.answer, "answer_resolved": ans,
            "status": result.status, "grade": str(result.grade),
            "classification": classification,
            "n_steps": len(result.steps), "elapsed_s": round(elapsed, 2),
        }
    except Exception as e:
        return {
            "claim_id": claim_id, "rep": rep, "shape": shape,
            "question": question, "gold": gold, "world_prior": world_prior,
            "answer_raw": None, "answer_resolved": None,
            "status": "ERROR", "grade": "", "classification": "error",
            "n_steps": 0, "elapsed_s": round(time.time() - t0, 2),
            "error": str(e),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--shapes", default="B,D",
                    help="Comma-separated shape letters to run (e.g., 'B,D').")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    shapes_wanted = set(s.strip().upper() for s in args.shapes.split(",") if s.strip())

    load_api_key()
    from llm_backend import AnthropicBackend
    backend = AnthropicBackend(model=args.model, temperature=args.temperature)

    pilot = json.load(open(args.pilot))
    pilot["_pilot_path"] = args.pilot  # for shape-D fixture path guessing
    tables_by_id = {t["id"]: t for t in pilot["tables"]}
    claims = [c for c in pilot["claims"] if shape_of(c["id"]) in shapes_wanted]
    print(f"Running Tier 2 on {len(claims)} claims "
          f"(shapes {sorted(shapes_wanted)}, model={args.model}, "
          f"repeats={args.repeats}) -> {args.log}", file=sys.stderr)

    results = []
    with open(args.log, "w") as f:
        f.write("ts\tclaim_id\trep\tshape\tquestion\tgold\tworld_prior\t"
                "answer_raw\tanswer_resolved\tstatus\tgrade\tclassification\t"
                "n_steps\telapsed_s\n")
        for ci, c in enumerate(claims):
            table = tables_by_id.get(c["table_ref"])
            if not table:
                print(f"  WARN: no table for {c['id']}", file=sys.stderr)
                continue
            for rep in range(args.repeats):
                r = run_one(c, table, backend, args.model,
                            args.temperature, rep, pilot)
                results.append(r)
                f.write(f"{int(time.time())}\t{r['claim_id']}\t{rep}\t{r['shape']}\t"
                        f"{r['question']}\t{r['gold']}\t{r.get('world_prior','')}\t"
                        f"{r.get('answer_raw','')}\t{r.get('answer_resolved','')}\t"
                        f"{r['status']}\t{r['grade']}\t{r['classification']}\t"
                        f"{r['n_steps']}\t{r['elapsed_s']}\n")
                f.flush()
                print(f"  [{r['claim_id']} rep={rep}] "
                      f"gold={r['gold']!r}  ans={r.get('answer_resolved','')!r}  "
                      f"verdict={r['classification']}  ({r['elapsed_s']}s)",
                      file=sys.stderr)

    counts = Counter(r["classification"] for r in results)
    print(f"\n=== summary ===", file=sys.stderr)
    for k, v in counts.most_common():
        print(f"  {k}: {v}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
