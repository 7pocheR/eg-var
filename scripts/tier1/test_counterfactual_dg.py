#!/usr/bin/env python3
"""Run counterfactual claims through D.G (EG-VAR with kernel).

Companion to scripts/tier1/test_counterfactual.py (which runs D.I /
baseline LLM-with-tools). D.G via kernel attestation must stay
source-faithful: the proof body asserts a literal answer string and
the kernel only typechecks if the tool actually attests that answer
for the synthetic counterfactual table.

The D.G fixture provides lean_goal_type / gold_tool_call /
gold_proof_body for each claim, keyed by (table_id, question). The
synthetic table is constructed from the pilot JSON and passed
in-memory.
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
    """Build a TableData from a counterfactual pilot table entry.

    Uses original_id so per-table tool resolution and lift checking
    lands on the curator's interpretive commitment for the source
    table; the synthetic rows replace the original data so the
    kernel attests only what the in-memory table shows.

    Canonicalizes column names so downstream runtime-emitted axioms
    use the canonical form (string-rfl against raw TableBench column
    names like ``"july 1 , 2013 projection"`` would otherwise fail).
    """
    from table_adapter import TableData, _canonicalize_col
    return TableData(
        table_id=t_entry["original_id"],
        columns=[_canonicalize_col(c) for c in t_entry["table"]["columns"]],
        rows=[[str(v).strip() for v in row] for row in t_entry["table"]["data"]],
    )


def shape_d_direction_word(witness: str, fix_entry: dict) -> str | None:
    """For shape-D fixture entries, map a kernel-verified
    winning-entity witness to a direction word based on
    subject_entity / direction_if_* fields.

    Shape-D claims ("Does X have a larger or smaller P than Y?")
    share the shape-B formalization: ArgmaxWhere over (X or Y)
    returns the entity with the larger P. If the witness matches
    subject_entity, the subject has the larger value (answer is
    direction_if_subject_wins); otherwise direction_if_subject_loses.

    Returns None if fixture entry is not shape-D.
    """
    if fix_entry.get('_shape') != 'D':
        return None
    subject = (fix_entry.get('subject_entity') or '').strip().lower()
    if not subject:
        return None
    d_win = fix_entry.get('direction_if_subject_wins')
    d_lose = fix_entry.get('direction_if_subject_loses')
    if not (d_win and d_lose):
        return None
    w = (witness or '').strip().lower()
    if not w:
        return None
    # Fuzzy match: witness contains subject OR subject contains witness
    if subject in w or w in subject:
        return d_win
    return d_lose


def classify_dg(result, gold: str, world_prior_value: str | None,
                fix_entry: dict | None = None) -> str:
    """Classify D.G outcome.

    D.G can be VERIFIED with answer matching gold (source_faithful),
    VERIFIED with answer matching world_prior (prior_override -
    impossible by construction; indicates kernel breach), no
    verified proof, or other failure.

    For shape-D entries the raw Sigma witness is the winning-entity
    name; it is post-processed to a direction word
    (larger/smaller/more/...) before matching against ``gold``. The
    kernel's verification of the ArgmaxWhere goal remains the
    load-bearing check; the post-processor is a deterministic map
    from witness to the claim's chosen lexical answer.
    """
    from eval_tablebench import answers_match
    status = (getattr(result, "status", "") or "").upper()
    grade = str(getattr(result, "grade", "")).lower()
    ans = result.answer or ""

    if status in ("VERIFIED",) or "verified" in grade:
        # Shape-D post-processing: map winner entity → direction word
        if fix_entry is not None:
            mapped = shape_d_direction_word(ans, fix_entry)
            if mapped is not None:
                ans_for_match = mapped
            else:
                ans_for_match = ans
        else:
            ans_for_match = ans
        if answers_match(ans_for_match, gold):
            return "source_faithful"
        if world_prior_value and answers_match(ans_for_match, world_prior_value):
            return "prior_override_KERNEL_BREACH"
        return "verified_other"
    return "no_verified_proof"


def find_v3_claim_for_fixture(fix_entry: dict, v3_claims: list) -> dict | None:
    """Match a D.G fixture entry back to its v3 claim (for gold lookup)."""
    q = fix_entry["question"].strip()
    for c in v3_claims:
        if c["question"].strip() == q:
            return c
    return None


def find_table_for_v3_claim(v3_claim: dict, tables_by_id: dict) -> dict | None:
    return tables_by_id.get(v3_claim["table_ref"])


def run_one(fix_entry: dict, v3_claim: dict, table_entry: dict, fixture_list: list,
            backend, log_path: Path, rep: int) -> dict:
    """Run one D.G evaluation. Returns result dict with classification."""
    from pipeline import run_pipeline
    table = make_table_data(table_entry)
    question = fix_entry["question"]
    gold = v3_claim["gold"]
    world_prior = v3_claim.get("world_prior")

    t0 = time.time()
    try:
        result = run_pipeline(
            claim=question, table=table, backend=backend,
            n_formalizers=1, variant="A",
            gold_formalization=fixture_list, question=question,
            enforce_menu=False,
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
                    ans = pipeline._extract_sigma_witness(f"⟨{lv},?_⟩")

        # Build a synthetic result-like for the classifier
        class _R:
            pass
        R = _R()
        R.status = result.status
        R.grade = result.grade
        R.answer = ans
        classification = classify_dg(R, gold, world_prior, fix_entry=fix_entry)

        return {
            "claim_id": v3_claim["id"],
            "rep": rep,
            "question": question,
            "gold": gold,
            "world_prior": world_prior,
            "answer_raw": result.answer,
            "answer_resolved": ans,
            "status": result.status,
            "grade": str(result.grade),
            "classification": classification,
            "n_steps": len(result.steps),
            "elapsed_s": round(elapsed, 2),
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "claim_id": v3_claim["id"], "rep": rep, "question": question,
            "gold": gold, "world_prior": world_prior,
            "answer_raw": None, "answer_resolved": None,
            "status": "EXCEPTION", "grade": "n/a",
            "classification": f"exception: {type(e).__name__}",
            "n_steps": 0, "elapsed_s": round(elapsed, 2),
            "error": f"{type(e).__name__}: {e}"[:200],
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True,
                    help="v3 pilot JSON (synthetic tables + claims)")
    ap.add_argument("--dg-fixture", required=True,
                    help="Companion D.G fixture (lean artifacts per claim)")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    load_api_key()
    from llm_backend import AnthropicBackend
    backend = AnthropicBackend(model=args.model, temperature=args.temperature)

    pilot = json.loads(Path(args.pilot).read_text())
    tables_by_id = {t["id"]: t for t in pilot["tables"]}
    v3_claims = pilot["claims"]

    dg_data = json.loads(Path(args.dg_fixture).read_text())
    fixture_list = dg_data["fixtures"]

    if args.log:
        log_path = Path(args.log)
    else:
        RESULTS_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        log_path = RESULTS_DIR / f"counterfactual_dg_{ts}.tsv"

    with log_path.open("w") as f:
        f.write("ts\tclaim_id\trep\tquestion\tgold\tworld_prior\t"
                "answer_resolved\tstatus\tgrade\tclassification\tn_steps\telapsed_s\n")

    print(f"D.G counterfactual: {len(fixture_list)} claims × {args.repeats} reps "
          f"on {args.model} (temp={args.temperature})", file=sys.stderr)

    by_class = Counter()
    results_log = []
    for fix in fixture_list:
        v3 = find_v3_claim_for_fixture(fix, v3_claims)
        if v3 is None:
            print(f"  [WARN] no v3 claim matches fixture question: "
                  f"{fix['question'][:60]!r}", file=sys.stderr)
            continue
        t_entry = tables_by_id.get(v3["table_ref"])
        if t_entry is None:
            print(f"  [WARN] no table for {v3['table_ref']}", file=sys.stderr)
            continue

        case_classes = []
        for rep in range(args.repeats):
            r = run_one(fix, v3, t_entry, fixture_list, backend, log_path, rep)
            with log_path.open("a") as f:
                f.write(f"{int(time.time())}\t{r['claim_id']}\t{rep}\t"
                        f"{r['question']}\t{r['gold']}\t{r.get('world_prior','')}\t"
                        f"{r['answer_resolved']}\t{r['status']}\t{r['grade']}\t"
                        f"{r['classification']}\t{r['n_steps']}\t{r['elapsed_s']}\n")
            case_classes.append(r["classification"])
            by_class[r["classification"]] += 1
            results_log.append(r)
        case_counter = Counter(case_classes)
        summary = ", ".join(f"{c}={n}" for c, n in case_counter.most_common())
        print(f"  [{v3['id']}] gold={v3['gold']!r}  →  {summary}", file=sys.stderr)

    print(f"\n=== D.G source-faithfulness summary "
          f"({args.repeats}×{len(fixture_list)}={args.repeats*len(fixture_list)} runs) ===",
          file=sys.stderr)
    total = sum(by_class.values())
    for c, n in by_class.most_common():
        print(f"  {c:32s}: {n}/{total} ({100*n/max(total,1):.1f}%)", file=sys.stderr)
    print(f"  Log: {log_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
