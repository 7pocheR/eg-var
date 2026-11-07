# EG-VAR Code Supplement

Companion code for [Evidence-Grounded Verified Agentic Reasoning: A Path Toward
Eliminating LLM Hallucination in Empirical Inference via Tool-Attested Kernel
Proofs](https://arxiv.org/abs/2607.12650), by Junyu Ren.

The public repository is [7pocheR/eg-var](https://github.com/7pocheR/eg-var).
Appendix G of the paper cites the frozen source snapshot at commit
[`88160519ce03dbd824f62799c1d3e1a02794a6e8`](https://github.com/7pocheR/eg-var/tree/88160519ce03dbd824f62799c1d3e1a02794a6e8).
Use this repository for code access; the former anonymous review mirror is no
longer required.

## Layout

```
src/
  lean/                        # Lean 4 library (kernel prelude)
    EGVar/                     # kernel prelude, world ontology, tactics
    EGVar.lean, Main.lean, lakefile.toml
  python/runtime/              # runtime (pipeline, formalizer, verifier)
data/
  tier1_gold_formalizations.json     # 120 gold goal types (Tier 1)
  tier1_table_formalizations.json    # audited per-source lifts
  tier1_tool_library.json
  tier1_counterfactual_pilot_v4.json   # extreme-flip panel claims
  tier1_counterfactual_pilot_v45.json  # subtle-flip panel claims
  tier1_counterfactual_dg_fixtures_v4.json   # gold for kernel-checked rung (v4)
  tier1_counterfactual_dg_fixtures_v45.json  # gold for kernel-checked rung (v45)
  tablebench/                  # TableBench FC+NR claims (used in E1, E3)
results/replay_fixtures/       # 11 generated Lean proofs (replay demo)
scripts/
  tier1/                       # Tier 1 + Tier 1.5 runners
  tier2/run_tier2_pilot.py     # Tier 2 runner
  replay.py
```

## Setup

```
# 1. Provision API key
export ANTHROPIC_API_KEY="<YOUR_ANTHROPIC_API_KEY>"

# 2. Python path — the runtime package is not installed; use PYTHONPATH
export PYTHONPATH=src/python/runtime

# 3. Build the Lean library once
cd src/lean && lake build && cd -
# expected: "Build completed successfully (22 jobs)."
```

Requirements: Lean 4 (elan / lake, any stable `v4.x`), Python 3.11+, and an
Anthropic API key with access to `claude-sonnet-4-6`,
`claude-haiku-4-5-20251001`, and `claude-opus-4-7`.

All commands below are one-shot runnable from the repository root after
the three setup steps.

---

## E1 — Tier 1 baseline ladder (§4.1, Table 3)

4-rung ladder on n=120 TableBench FC+NR claims. Sonnet reported in the
main text; Opus added for the kernel-checked rung per Appendix L.

```
# Sonnet — all 4 rungs
python3 scripts/tier1/run_dg_eval.py      --limit 120 --model claude-sonnet-4-6 --out results/tier1_dg_sonnet.jsonl
python3 scripts/tier1/run_di_baseline.py  --variant strict --limit 120 --model claude-sonnet-4-6 --out results/tier1_di_strict_sonnet.jsonl
python3 scripts/tier1/run_di_baseline.py  --variant loose  --limit 120 --model claude-sonnet-4-6 --out results/tier1_di_loose_sonnet.jsonl
python3 scripts/tier1/run_dj_no_tools.py  --limit 120 --model claude-sonnet-4-6 --out results/tier1_dj_sonnet.jsonl

# Opus — kernel-checked rung only (Appendix L)
python3 scripts/tier1/run_dg_eval.py --limit 120 --model claude-opus-4-7 --out results/tier1_dg_opus.jsonl

# Post-process: per-claim ladder CSV (Appendix B)
python3 scripts/tier1/build_baseline_ladder_csv.py
```

**Rung → runner mapping**: `run_dg_eval.py` = kernel-checked;
`run_di_baseline.py --variant strict` = Tools-curated (per-source menu);
`run_di_baseline.py --variant loose` = Tools-open (free tools);
`run_dj_no_tools.py` = Table-only (no tools).

Paper numbers: kernel-checked rung 120/120, Tools-curated 114/120, Tools-open 111/120,
Table-only 107/120.

---

## E2 — Tier 1.5 counterfactual stress panels (§4.2, Table 4)

Two panels (extreme-flip v4, subtle-flip v45) × 4 rungs × 2 models
(Sonnet + Haiku). Each cell reports source-faithful count on
shape-B (binary-compare) + shape-D (direction) claims.

```
# === Extreme-flip panel (v4) ===
# Sonnet
python3 scripts/tier1/test_counterfactual_dg.py   --pilot data/tier1_counterfactual_pilot_v4.json  --dg-fixture data/tier1_counterfactual_dg_fixtures_v4.json  --model claude-sonnet-4-6 --log results/cf_v4_dg_sonnet.tsv
python3 scripts/tier1/test_counterfactual.py      --pilot data/tier1_counterfactual_pilot_v4.json  --variant strict --model claude-sonnet-4-6 --log results/cf_v4_di_strict_sonnet.tsv
python3 scripts/tier1/test_counterfactual.py      --pilot data/tier1_counterfactual_pilot_v4.json  --variant loose  --model claude-sonnet-4-6 --log results/cf_v4_di_loose_sonnet.tsv
python3 scripts/tier1/test_counterfactual_dj.py   --pilot data/tier1_counterfactual_pilot_v4.json  --model claude-sonnet-4-6 --log results/cf_v4_dj_sonnet.tsv

# Haiku
python3 scripts/tier1/test_counterfactual_dg.py   --pilot data/tier1_counterfactual_pilot_v4.json  --dg-fixture data/tier1_counterfactual_dg_fixtures_v4.json  --model claude-haiku-4-5-20251001 --log results/cf_v4_dg_haiku.tsv
python3 scripts/tier1/test_counterfactual.py      --pilot data/tier1_counterfactual_pilot_v4.json  --variant strict --model claude-haiku-4-5-20251001 --log results/cf_v4_di_strict_haiku.tsv
python3 scripts/tier1/test_counterfactual.py      --pilot data/tier1_counterfactual_pilot_v4.json  --variant loose  --model claude-haiku-4-5-20251001 --log results/cf_v4_di_loose_haiku.tsv
python3 scripts/tier1/test_counterfactual_dj.py   --pilot data/tier1_counterfactual_pilot_v4.json  --model claude-haiku-4-5-20251001 --log results/cf_v4_dj_haiku.tsv

# === Subtle-flip panel (v45) ===
# Sonnet
python3 scripts/tier1/test_counterfactual_dg.py   --pilot data/tier1_counterfactual_pilot_v45.json --dg-fixture data/tier1_counterfactual_dg_fixtures_v45.json --model claude-sonnet-4-6 --log results/cf_v45_dg_sonnet.tsv
python3 scripts/tier1/test_counterfactual.py      --pilot data/tier1_counterfactual_pilot_v45.json --variant strict --model claude-sonnet-4-6 --log results/cf_v45_di_strict_sonnet.tsv
python3 scripts/tier1/test_counterfactual.py      --pilot data/tier1_counterfactual_pilot_v45.json --variant loose  --model claude-sonnet-4-6 --log results/cf_v45_di_loose_sonnet.tsv
python3 scripts/tier1/test_counterfactual_dj.py   --pilot data/tier1_counterfactual_pilot_v45.json --model claude-sonnet-4-6 --log results/cf_v45_dj_sonnet.tsv

# Haiku
python3 scripts/tier1/test_counterfactual_dg.py   --pilot data/tier1_counterfactual_pilot_v45.json --dg-fixture data/tier1_counterfactual_dg_fixtures_v45.json --model claude-haiku-4-5-20251001 --log results/cf_v45_dg_haiku.tsv
python3 scripts/tier1/test_counterfactual.py      --pilot data/tier1_counterfactual_pilot_v45.json --variant strict --model claude-haiku-4-5-20251001 --log results/cf_v45_di_strict_haiku.tsv
python3 scripts/tier1/test_counterfactual.py      --pilot data/tier1_counterfactual_pilot_v45.json --variant loose  --model claude-haiku-4-5-20251001 --log results/cf_v45_di_loose_haiku.tsv
python3 scripts/tier1/test_counterfactual_dj.py   --pilot data/tier1_counterfactual_pilot_v45.json --model claude-haiku-4-5-20251001 --log results/cf_v45_dj_haiku.tsv
```

`test_counterfactual_dg.py` defaults to `--repeats 3`; `--log` receives
TSV output with one row per (claim, rep) cell.

Paper numbers: kernel-checked rung 20/20 on every cell; Sonnet Tools-curated drops to
16–20 under extreme-flip and 18–20 under subtle-flip; Haiku stays at 20/20
(tool-deferent).

---

## E3 — Tier 2 LLM-as-formalizer (§4.3, Table 5)

n=120 claims × 2 models; formalizer proposes the goal type (no gold
bypass).

```
python3 scripts/tier2/run_tier2_pilot.py --limit 120 --model claude-sonnet-4-6 --out results/tier2_sonnet.jsonl
python3 scripts/tier2/run_tier2_pilot.py --limit 120 --model claude-opus-4-7  --out results/tier2_opus.jsonl
```

Paper numbers (semantic formalizer error rate): Sonnet 3.3%, Opus 1.7%.
Six-category decomposition in Table 5; definitions in Appendix H.

---

## E4 — Tier 2 under counterfactual pressure (§4.3 last paragraph)

Tier 2 with LLM formalization, on the extreme-flip v4 panel, restricted
to shape-B + shape-D (n=20). Option-Z formalizer prompt shape
(`src/python/runtime/pipeline.py:590+`, FORMALIZER_SYSTEM) is always-on;
no additional flag.

```
python3 scripts/tier1/test_counterfactual_tier2.py --pilot data/tier1_counterfactual_pilot_v4.json --model claude-sonnet-4-6 --log results/cf_v4_tier2_sonnet.jsonl
```

`--shapes B,D` default restricts to the 20-claim binary-compare +
direction subset. Paper reports 20/20 source-faithful VERIFIED outputs.

---

## Replay-audit capability demo (§5.1, Appendix G — no API key)

Verifies that each of 11 committed generated proofs passes the
whitelist gate. `verify_axioms.py` parses `#print axioms` output and
checks every axiom lands in one of four buckets (kernel prelude, Lean 4
core, runtime-emitted observation leaves, per-source audited lifts).

```
python3 scripts/replay.py
```

Expect all 11 fixtures to pass. This is the same check that runs
post-hoc on every `VERIFIED` output in E1, E3, and E4.

---

## Guarantees shipped

- **No unsupported `VERIFIED` outputs** (Theorem 1). Every `VERIFIED`
  verdict implies every runtime-emitted axiom in the proof's
  dependency closure is backed by a prior tool-call record.
- **No deductive errors** (Theorem 2). The Lean 4 kernel is the sole
  type-checker for every proof step.

Full proofs + emission-site coverage table in Appendix P of the paper.
