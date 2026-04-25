/-
  EG-VAR Heuristic Rules (Γh)

  Three categories of evidence constructors per Round 3 convergence:
  1. mkVerified (in Basic.lean) — sole source of Verified, requires Attested_T
  2. Domain-specific heuristic axioms (here) — derive non-Verified evidence
     from typed premises per EG-VAR.md §3.6
  3. llmAbduction (here) — zero-premise, Speculative only

  Design decision: ADR-002 (domain-specific axioms, not generic mkHeuristic)
-/
import EGVar.Basic
import EGVar.Ontology
set_option autoImplicit false

-- ============================================================
-- § 1. Domain-Independent Downgrade Rules
-- ============================================================

/-- Staleness: Verified → Supported when freshness fails.
    Evidence bound to timestamp via At (audit fix). -/
noncomputable def staleDowngrade {w : WProp} {t : Timestamp} {δ now : Nat}
    (ev : Evidence Grade.Verified (At t w))
    (_stale : ¬ FreshEnough t δ now)
    : Evidence Grade.Supported (At t w) :=
  downcast (by decide) ev

/-- LLM formalization: caps at Supported (not Plausible).
    Round 2 convergence: Supported because the mapping is untrusted
    but downstream evidence may still come from attested sources. -/
noncomputable def llmFormalizationDowngrade {w : WProp} {g : Grade}
    (ev : Evidence g w)
    : Evidence (Grade.meet g Grade.Supported) w :=
  downcast (Grade.meet_le_left g Grade.Supported) ev

/-- Heuristic inference caps grade at Supported (generic). -/
noncomputable def heuristicDowngrade {w : WProp} {g : Grade}
    (ev : Evidence g w)
    : Evidence (Grade.meet g Grade.Supported) w :=
  downcast (Grade.meet_le_left g Grade.Supported) ev

-- ============================================================
-- § 2. Zero-Premise LLM Abduction (Speculative only)
--
-- The LLM proposes a claim from its training data with no tool
-- backing. This is the ONLY ex nihilo evidence constructor.
-- Output is always Speculative — the lowest grade.
-- ============================================================

/-- LLM abduction: zero-premise, Speculative only.
    The LLM believes w based on parametric knowledge.
    No tool evidence required. -/
axiom llmAbduction (w : WProp) : Evidence Grade.Speculative w

-- ============================================================
-- § 3. Source Conflict Downgrade (Weather Domain)
-- ============================================================

/-- Times are close enough that different values indicate conflict. -/
def TimesClose (t1 t2 : Timestamp) (ε : Duration) : Prop :=
  (if t1 ≥ t2 then t1 - t2 else t2 - t1) ≤ ε

instance {t1 t2 ε : Nat} : Decidable (TimesClose t1 t2 ε) := by
  unfold TimesClose; exact inferInstance

/-- Source provenance: identifies which tool+query produced a piece of evidence.
    Both fields are Nat hashes computed by the trusted runtime. -/
structure SourceId where
  toolHash : Nat   -- hash of the ToolId
  queryHash : Nat  -- hash of the query, distinguishes calls to the same tool
  deriving DecidableEq, Repr

/-- Two sources are distinct (different tool OR different query to same tool). -/
def SourcesDistinct (s1 s2 : SourceId) : Prop := s1 ≠ s2
instance {s1 s2 : SourceId} : Decidable (SourcesDistinct s1 s2) :=
  inferInstanceAs (Decidable (s1 ≠ s2))

/-- Temperature conflict witness: two observations of the same location
    from DISTINCT sources at close timestamps disagree beyond threshold. -/
structure TempConflictWitness
    (loc : Location) (t1 t2 : Timestamp) (v1 v2 : Temperature)
    (εTime : Duration) (minGap : Nat) where
  /-- The conflicting observation. -/
  ev_other : Evidence Grade.Verified (At t2 (TempAt loc t2 v2))
  /-- Timestamps are close enough that difference = disagreement, not change. -/
  times_close : TimesClose t1 t2 εTime
  /-- Values disagree beyond threshold. -/
  disagreement : TempDisagreement v1 v2 minGap
  /-- Provenance of the original observation. -/
  source_self : SourceId
  /-- Provenance of the conflicting observation. -/
  source_other : SourceId
  /-- Proof the sources are distinct. -/
  sources_distinct : SourcesDistinct source_self source_other

/-- Temperature conflict downgrade: Verified → Supported when another
    source disagrees on the temperature at the same location. -/
noncomputable def tempConflictDowngrade
    {loc : Location} {t1 t2 : Timestamp} {v1 v2 : Temperature}
    {εTime : Duration} {minGap : Nat}
    (ev : Evidence Grade.Verified (At t1 (TempAt loc t1 v1)))
    (_conflict : TempConflictWitness loc t1 t2 v1 v2 εTime minGap)
    : Evidence Grade.Supported (At t1 (TempAt loc t1 v1)) :=
  downcast (by decide) ev

-- ============================================================
-- § 4. Domain-Specific Heuristic Axioms (Weather / Finance)
--
-- Each axiom:
-- - Takes typed evidence premises (per EG-VAR.md §3.6)
-- - Outputs grade ≤ max(input grades), never Verified
-- - Is individually auditable by inspecting its type signature
-- ============================================================

/-- Cross-rate heuristic (§3.6, specs/query-formalization-rules.md F3):
    Derive B/C exchange rate from A/B and A/C rates.
    Output is Supported because the no-arbitrage assumption is unattested.
    Premises: two Verified FX rates + TimesClose witness. -/
axiom crossRateAxiom
    {base quote1 quote2 : Currency}
    {t1 t2 : Timestamp} {r1 r2 : Ratio}
    {ε : Duration}
    (ev1 : Evidence Grade.Verified
      (At t1 (ExchangeRate base quote1 t1 r1)))
    (ev2 : Evidence Grade.Verified
      (At t2 (ExchangeRate base quote2 t2 r2)))
    (close : TimesClose t1 t2 ε)
    (r1_pos : r1 > 0)  -- guard: denominator must be nonzero
    : Evidence Grade.Supported
        (At t1 (ExchangeRate quote1 quote2 t1 (r2 * 10000 / r1)))

-- ============================================================
-- § 5. No-Upcast Proofs
-- ============================================================

theorem verified_not_le_supported : ¬(Grade.Verified ≤ Grade.Supported) := by decide
theorem verified_not_le_plausible : ¬(Grade.Verified ≤ Grade.Plausible) := by decide
theorem verified_not_le_speculative : ¬(Grade.Verified ≤ Grade.Speculative) := by decide

-- ============================================================
-- § 6. Freshness Guard Confirmation
-- ============================================================

/-- FreshEnough guards against future timestamps. -/
theorem freshEnough_not_future {t δ now : Nat} (h : FreshEnough t δ now) : t ≤ now := h.1

/-- FreshEnough bounds staleness. -/
theorem freshEnough_within_delta {t δ now : Nat} (h : FreshEnough t δ now) : now - t ≤ δ := h.2
