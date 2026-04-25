/-
  EG-VAR World-Level Rules

  Downgrade and heuristic rules for the world-level ontology.
  Generalized from weather/finance-specific rules to work with
  HasProperty/Relation predicates.

  Per ADR-005.
-/
import EGVar.Basic
import EGVar.WorldOntology

set_option autoImplicit false

-- ============================================================
-- § 1. Domain-Independent Downgrade Rules
-- (Imported from Rules.lean — these don't change)
-- ============================================================

/-- Staleness: Verified → Supported when freshness fails. -/
noncomputable def staleDowngrade' {w : WProp} {t : Timestamp} {δ now : Nat}
    (ev : Evidence Grade.Verified (At t w))
    (_stale : ¬ FreshEnough t δ now)
    : Evidence Grade.Supported (At t w) :=
  downcast (by decide) ev

/-- LLM formalization: caps at Supported. -/
noncomputable def llmFormalizationDowngrade' {w : WProp} {g : Grade}
    (ev : Evidence g w)
    : Evidence (Grade.meet g Grade.Supported) w :=
  downcast (Grade.meet_le_left g Grade.Supported) ev

/-- LLM abduction: zero-premise, Speculative only. -/
axiom llmAbduction' (w : WProp) : Evidence Grade.Speculative w

-- ============================================================
-- § 2. World-Level Conflict Detection
-- ============================================================

/-- Two values disagree beyond a threshold. -/
def ValueDisagreement (v1 v2 : Value) (minGap : Nat) : Prop :=
  match v1.toInt, v2.toInt with
  | some x, some y => minGap ≤ Int.natAbs (x - y)
  | _, _ => False

instance {v1 v2 : Value} {minGap : Nat} : Decidable (ValueDisagreement v1 v2 minGap) := by
  unfold ValueDisagreement
  cases v1.toInt <;> cases v2.toInt <;> simp <;> exact inferInstance

/-- Source provenance for conflict detection. -/
structure SourceId' where
  toolHash : Nat
  queryHash : Nat
  deriving DecidableEq, Repr

/-- Conflict witness: two sources report different values for the same
    entity/property at close timestamps. -/
structure ConflictWitness'
    (entity : String) (prop : String) (v1 v2 : Value)
    (t1 t2 : Timestamp) (εTime : Duration) (minGap : Nat) where
  ev_other : Evidence Grade.Verified (At t2 (HasProperty entity prop v2))
  times_close : t1 ≤ t2 ∧ t2 - t1 ≤ εTime ∨ t2 ≤ t1 ∧ t1 - t2 ≤ εTime
  disagreement : ValueDisagreement v1 v2 minGap
  source_self : SourceId'
  source_other : SourceId'
  sources_distinct : source_self ≠ source_other

/-- Conflict downgrade: Verified → Supported when sources disagree. -/
noncomputable def conflictDowngrade'
    {entity : String} {prop : String} {v1 v2 : Value}
    {t1 t2 : Timestamp} {εTime : Duration} {minGap : Nat}
    (ev : Evidence Grade.Verified (At t1 (HasProperty entity prop v1)))
    (_conflict : ConflictWitness' entity prop v1 v2 t1 t2 εTime minGap)
    : Evidence Grade.Supported (At t1 (HasProperty entity prop v1)) :=
  downcast (by decide) ev

-- ============================================================
-- § 3. No-Upcast Proofs
-- ============================================================

theorem verified_not_le_supported' : ¬(Grade.Verified ≤ Grade.Supported) := by decide
theorem verified_not_le_plausible' : ¬(Grade.Verified ≤ Grade.Plausible) := by decide
