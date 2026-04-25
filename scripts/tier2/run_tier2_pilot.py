#!/usr/bin/env python3
"""Tier 2 - formalizer pilot (single LLM formalizer, no gold).

Subset (frozen, seed=42; stratified by op x Cond.all/conditional):
  count_where 8 cond | mean 4 all + 4 cond | sum 3 all + 3 cond
  argmax 2 all + 2 cond | max 1 all + 1 cond | min 1 all
  argmin 1 all = 30

Per-claim taxonomy:
  B1: type-eq + accepted + answer-match    -> CORRECT
  B2: type-eq + accepted + answer-mismatch -> framework regression
  B3: type-eq + rejected                   -> solver gave up
  B4: type-neq + accepted + answer-match   -> accidental hit
  B5: type-neq + accepted + answer-mismatch-> silent wrong
  B6: type-neq + rejected                  -> honest abstain
  B7: formalizer abstained early           -> early honest abstain

Headline metrics:
  silent_wrong_rate     = B5 / N
  honest_failure_rate   = (B1 + B3 + B6 + B7) / N
  answer_match_overall  = (B1 + B4) / N
  answer_match_accepted = (B1 + B4) / (B1 + B2 + B4 + B5)
  abstain_rate          = (B3 + B6 + B7) / N
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
import collections
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = PROJECT_ROOT / "src" / "python" / "runtime"
TABLEBENCH_PATH = PROJECT_ROOT / "data" / "tablebench" / "TableBench.jsonl"
FIXTURE_PATH = PROJECT_ROOT / "data" / "tier1_gold_formalizations.json"
API_KEY_PATH = PROJECT_ROOT / "api_key.md"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(RUNTIME_DIR))


# Stratification: op x (Cond.all | conditional).
SUBSET_TARGETS = {
    ("count", "cond"): 8,
    ("mean", "all"): 4, ("mean", "cond"): 4,
    ("sum", "all"): 3, ("sum", "cond"): 3,
    ("argmax", "all"): 2, ("argmax", "cond"): 2,
    ("max", "all"): 1, ("max", "cond"): 1,
    ("min", "all"): 1,
    ("argmin", "all"): 1,
}
SUBSET_SEED = 42


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


def classify_op_cond(entry: dict) -> tuple[str, str]:
    """Map fixture entry -> (operation, 'all'|'cond') for stratification."""
    s = entry.get("lean_goal_type", "")
    op = "other"
    for cand in ("CountWhere", "MeanWhere", "SumWhere", "ArgmaxWhere",
                 "MaxWhere", "MinWhere", "ArgminWhere",
                 "TopKMeanWhere", "TopKSumWhere", "TopKSumByWhere"):
        if cand in s:
            op = cand.replace("Where", "").replace("By", "_by").lower()
            break
    cond = "all" if "Cond.all" in s else "cond"
    return op, cond


def build_subset(fixture: list) -> list[int]:
    """Return frozen list of fixture indices for the Tier 2 30-claim pilot."""
    rng = random.Random(SUBSET_SEED)
    by_strat = collections.defaultdict(list)
    for i, f in enumerate(fixture):
        op, cond = classify_op_cond(f)
        by_strat[(op, cond)].append(i)
    indices = []
    for stratum, n in SUBSET_TARGETS.items():
        pool = sorted(by_strat.get(stratum, []))
        if len(pool) < n:
            print(f"WARN: stratum {stratum} has {len(pool)} entries, want {n}",
                  file=sys.stderr)
            n = len(pool)
        indices.extend(rng.sample(pool, n))
    return sorted(indices)


def classify_outcome(*, type_eq: bool, kernel_accepted: bool,
                     answer_match: bool, formalizer_aborted: bool) -> str:
    """Map (type_eq, kernel_accepted, answer_match) to bucket B1..B7.

    B7 takes precedence: formalizer never produced a goal.
    """
    if formalizer_aborted:
        return "B7"  # early honest abstain
    if type_eq and kernel_accepted and answer_match:
        return "B1"  # CORRECT
    if type_eq and kernel_accepted and not answer_match:
        return "B2"  # framework regression (tool bug)
    if type_eq and not kernel_accepted:
        return "B3"  # solver gave up on the right type
    if (not type_eq) and kernel_accepted and answer_match:
        return "B4"  # accidental hit
    if (not type_eq) and kernel_accepted and not answer_match:
        return "B5"  # SILENT WRONG (load-bearing)
    if (not type_eq) and not kernel_accepted:
        return "B6"  # honest abstain (wrong type rejected)
    return "B?"  # unreachable


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


def run_one(idx: int, entry: dict, backend, *, persist_steps: bool = False,
            n_formalizers: int = 1, variant: str = "A",
            prompt_diversity: bool = False, formalizer_backend=None,
            anomaly_aware: bool = False,
            ambiguity_auditor: bool = False,
            contrastive: bool = False,
            menu_aware: bool = False) -> dict:
    """Run pipeline WITHOUT gold_formalization; classify outcome.

    When persist_steps=True the per-claim record carries the full
    solver `steps` list (tool calls, kernel checks, feedback) so the
    B3 diagnosis path can see what the solver actually attempted.
    """
    import re
    import pipeline
    from eval_tablebench import answers_match

    table_id = entry["table_id"]
    question = entry["question"]
    gold_lean_type = entry.get("lean_goal_type", "")

    table, gold_answer = load_table(table_id)
    if table is None:
        return {"idx": idx, "table_id": table_id, "status": "ERROR",
                "error": "table not found", "bucket": "B?"}

    t0 = time.time()
    try:
        # If formalizer_backend differs from backend, build a routing
        # llm_call that uses formalizer_backend for the 'formalizer'
        # role only; other roles stay on the default backend.
        if formalizer_backend is not None and formalizer_backend is not backend:
            def routed_llm_call(role: str, system_prompt: str, user_prompt: str) -> str:
                if role == "formalizer":
                    return formalizer_backend.call(role, system_prompt, user_prompt)
                return backend.call(role, system_prompt, user_prompt)
            result = pipeline.run_pipeline(
                claim=question, table=table, llm_call=routed_llm_call,
                n_formalizers=n_formalizers, variant=variant,
                gold_formalization=None,
                question=question,
                enforce_menu=True,
                prompt_diversity=prompt_diversity,
                anomaly_aware=anomaly_aware,
                ambiguity_auditor=ambiguity_auditor,
                contrastive=contrastive,
                menu_aware=menu_aware,
            )
        else:
            # Tier 2 toggle: gold_formalization=None => LLM
            # formalizer runs end-to-end.
            result = pipeline.run_pipeline(
                claim=question, table=table, backend=backend,
                n_formalizers=n_formalizers, variant=variant,
                gold_formalization=None,
                question=question,
                enforce_menu=True,
                prompt_diversity=prompt_diversity,
                anomaly_aware=anomaly_aware,
                ambiguity_auditor=ambiguity_auditor,
                contrastive=contrastive,
                menu_aware=menu_aware,
            )
        elapsed = time.time() - t0

        # Resolve obs<N>_val references.
        ans = result.answer
        if ans and isinstance(ans, str):
            m = re.match(r"^obs(\d+)_val$", ans)
            if m:
                ev_idx = int(m.group(1))
                evidence_list = getattr(result, "evidence", None) or []
                if 0 <= ev_idx < len(evidence_list):
                    lv = evidence_list[ev_idx].get("lean_value", "")
                    ans = pipeline._extract_sigma_witness(f"⟨{lv},?_⟩")

        # Extract LLM-proposed lean_goal_type from formalizer_outputs.
        formalizer_outputs = (result.details or {}).get("formalizer_outputs", []) or []
        valid = [f for f in formalizer_outputs if f.get("goal")]
        formalizer_aborted = (len(valid) == 0)
        llm_lean_type = (valid[0].get("lean_goal_type", "") if valid else "")
        # All formalizer lean_goal_types + their prompt variants for
        # post-hoc consensus / b-score analysis.
        all_lean_types = [
            {"prompt_variant": f.get("prompt_variant", "standard"),
             "lean_goal_type": f.get("lean_goal_type", ""),
             "ambiguous_terms": f.get("ambiguous_terms", []),
             "confidence": f.get("confidence", "")}
            for f in formalizer_outputs
        ]

        # Type equality via kernel rfl. Normalize guillemets
        # `«name»` to `"name"` on both sides so the kernel sees
        # apples-to-apples string literals.
        type_eq = False
        if llm_lean_type and gold_lean_type:
            llm_norm = pipeline._lean_strip_guillemets(llm_lean_type)
            gold_norm = pipeline._lean_strip_guillemets(gold_lean_type)
            type_eq = pipeline._goals_equal(llm_norm, gold_norm)

        kernel_accepted = (result.status == "VERIFIED")
        ans_match = (answers_match(ans or "", gold_answer or "")
                     if (ans and gold_answer) else False)

        bucket = classify_outcome(
            type_eq=type_eq,
            kernel_accepted=kernel_accepted,
            answer_match=ans_match,
            formalizer_aborted=formalizer_aborted,
        )

        out = {
            "idx": idx, "table_id": table_id, "question": question,
            "gold_answer": gold_answer,
            "gold_lean_type": gold_lean_type,
            "llm_lean_type": llm_lean_type,
            "type_eq": type_eq,
            "status": result.status,
            "grade": str(result.grade),
            "kernel_accepted": kernel_accepted,
            "answer": result.answer,
            "answer_resolved": ans,
            "answer_match": ans_match,
            "formalizer_aborted": formalizer_aborted,
            "bucket": bucket,
            "n_steps": len(result.steps),
            "n_formalizer_attempts": len(formalizer_outputs),
            "all_lean_types": all_lean_types,
            "elapsed_s": round(elapsed, 2),
            "enforce_menu": True,
            "prompt_diversity": prompt_diversity,
        }
        if persist_steps:
            out["steps"] = result.steps
        return out
    except Exception as e:
        elapsed = time.time() - t0
        return {"idx": idx, "table_id": table_id, "status": "EXCEPTION",
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(elapsed, 2),
                "bucket": "B?", "enforce_menu": True}


def summarize(results: list[dict]) -> dict:
    """Compute headline metrics from per-claim results."""
    n = len(results)
    if n == 0:
        return {"n": 0}
    counts = collections.Counter(r.get("bucket", "B?") for r in results)
    b = lambda k: counts.get(k, 0)
    silent_wrong = b("B5")
    honest_failure = b("B1") + b("B3") + b("B6") + b("B7")
    answer_match_overall = b("B1") + b("B4")
    accepted = b("B1") + b("B2") + b("B4") + b("B5")
    answer_match_accepted_num = b("B1") + b("B4")
    abstain = b("B3") + b("B6") + b("B7")
    return {
        "n": n,
        "buckets": dict(counts),
        "silent_wrong_rate": silent_wrong / n,
        "honest_failure_rate": honest_failure / n,
        "answer_match_overall": answer_match_overall / n,
        "answer_match_accepted": (answer_match_accepted_num / accepted
                                   if accepted else 0.0),
        "abstain_rate": abstain / n,
        "accepted_n": accepted,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--persist-steps", action="store_true",
                    help="Include full solver steps list per claim "
                         "(B3 diagnosis path).")
    ap.add_argument("--indices", default=None,
                    help="Comma-separated fixture indices to run "
                         "(overrides the frozen 30-claim subset).")
    ap.add_argument("--n-formalizers", type=int, default=1,
                    help="Number of independent claim formalizer "
                         "calls. n>1 captures all formalizer outputs "
                         "in details['formalizer_outputs'] for "
                         "inspection / consensus analysis.")
    ap.add_argument("--formalizer-model", default=None,
                    help="If set and != --model, route the "
                         "'formalizer' role only to this model "
                         "(other roles stay on --model).")
    ap.add_argument("--variant", default="A", choices=["A", "B", "F"],
                    help="Pipeline variant: A=Analyst+Formalizer+"
                         "Solver, B=Masked-table (no analyst), "
                         "F=Variant B + refinement retry on tool "
                         "failure.")
    ap.add_argument("--prompt-diversity", action="store_true",
                    help="Cycle the formalizer through "
                         "narrow/broad/literal interpretation "
                         "policies. Requires --n-formalizers=3.")
    ap.add_argument("--anomaly-aware", action="store_true",
                    help="Inject per-table structural metadata "
                         "(summary rows, identity column, gating "
                         "notes) into the formalizer prompt.")
    ap.add_argument("--ambiguity-auditor", action="store_true",
                    help="Run ambiguity auditor (claim-only, blind "
                         "to table/menu) before formalizer; flagged "
                         "phrases thread into formalizer prompt and "
                         "force has_ambiguity=True for grade "
                         "downcast.")
    ap.add_argument("--contrastive", action="store_true",
                    help="Formalizer also emits "
                         "alternative_formalization (runner-up + "
                         "why rejected). Transparency-only; does "
                         "not affect grade.")
    ap.add_argument("--max-solver-steps", type=int, default=None,
                    help="Override pipeline.MAX_SOLVER_STEPS "
                         "(default 20).")
    ap.add_argument("--menu", action="store_true",
                    help="Per-table sound-operations menu "
                         "injection. Adds a 'PER-TABLE MENU' block "
                         "to the formalizer prompt listing tools "
                         "for this table.")
    args = ap.parse_args()

    if args.max_solver_steps is not None:
        import pipeline as _pipe
        _pipe.MAX_SOLVER_STEPS = args.max_solver_steps
        print(f"MAX_SOLVER_STEPS overridden to {args.max_solver_steps}",
              file=sys.stderr)

    load_api_key()
    from llm_backend import AnthropicBackend

    fixture = json.loads(FIXTURE_PATH.read_text())
    backend = AnthropicBackend(model=args.model, temperature=args.temperature)
    formalizer_backend = None
    if args.formalizer_model and args.formalizer_model != args.model:
        formalizer_backend = AnthropicBackend(model=args.formalizer_model,
                                               temperature=args.temperature)
        print(f"Routing formalizer role -> {args.formalizer_model}; other roles -> {args.model}",
              file=sys.stderr)

    if args.indices:
        indices = sorted({int(x.strip()) for x in args.indices.split(",")
                          if x.strip()})
        print(f"Tier 2 custom indices ({len(indices)}): {indices}",
              file=sys.stderr)
    else:
        indices = build_subset(fixture)
        print(f"Tier 2 frozen subset: {len(indices)} indices = {indices}",
              file=sys.stderr)

    if args.out:
        out_path = Path(args.out)
    else:
        RESULTS_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = RESULTS_DIR / f"tier2_pilot_{ts}.jsonl"

    done = load_done(out_path) if args.resume else set()
    selected = [(i, fixture[i]) for i in indices if i not in done]
    if args.limit is not None:
        selected = selected[: args.limit]

    print(f"Running {len(selected)} entries -> {out_path}", file=sys.stderr)

    if args.prompt_diversity and args.n_formalizers != 3:
        sys.exit(f"--prompt-diversity requires --n-formalizers=3 "
                 f"(got {args.n_formalizers})")

    results = []
    for j, (i, entry) in enumerate(selected):
        result = run_one(i, entry, backend, persist_steps=args.persist_steps,
                         n_formalizers=args.n_formalizers, variant=args.variant,
                         prompt_diversity=args.prompt_diversity,
                         formalizer_backend=formalizer_backend,
                         anomaly_aware=args.anomaly_aware,
                         ambiguity_auditor=args.ambiguity_auditor,
                         contrastive=args.contrastive,
                         menu_aware=args.menu)
        with out_path.open("a") as f:
            f.write(json.dumps(result) + "\n")
        results.append(result)
        marker = result.get("bucket", "?")
        ans = result.get("answer_resolved", result.get("answer", "?"))
        gold = result.get("gold_answer", "?")
        elapsed = result.get("elapsed_s", 0)
        print(f"  [{j+1:3d}/{len(selected)}] idx={i:3d} bucket={marker:3s} "
              f"type_eq={result.get('type_eq', False)} "
              f"accepted={result.get('kernel_accepted', False)} "
              f"ans_match={result.get('answer_match', False)} "
              f"({elapsed:.1f}s)", file=sys.stderr)

    # Always re-load full file for resume-aware summary.
    if args.resume and out_path.exists():
        results = []
        with out_path.open() as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    summary = summarize(results)
    print(f"\n=== Tier 2 pilot summary (n={summary['n']}) ===", file=sys.stderr)
    print(f"  buckets:                {summary.get('buckets', {})}", file=sys.stderr)
    print(f"  silent_wrong_rate:      {summary.get('silent_wrong_rate', 0):.3f}  "
          f"({summary.get('buckets', {}).get('B5', 0)}/{summary['n']})",
          file=sys.stderr)
    print(f"  honest_failure_rate:    {summary.get('honest_failure_rate', 0):.3f}",
          file=sys.stderr)
    print(f"  answer_match_overall:   {summary.get('answer_match_overall', 0):.3f}",
          file=sys.stderr)
    print(f"  answer_match_accepted:  {summary.get('answer_match_accepted', 0):.3f}  "
          f"(n_accepted={summary.get('accepted_n', 0)})", file=sys.stderr)
    print(f"  abstain_rate:           {summary.get('abstain_rate', 0):.3f}",
          file=sys.stderr)
    print(f"  output:                 {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
