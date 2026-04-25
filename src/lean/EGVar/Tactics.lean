/-
  EG-VAR Proof Tactics Library

  Pre-proven lemmas that the solver can reference instead of writing
  from scratch. Each pattern corresponds to a common claim type.

  Usage: the solver writes `exact gte_pair obs0_att obs0_tag obs1_att obs1_tag (by native_decide)`
  instead of manually constructing 8 lines of PProd nesting.
-/
import EGVar.Basic
import EGVar.WorldOntology

set_option autoImplicit false

-- ============================================================
-- § 1. Comparison (A > B, A < B, A = B)
-- ============================================================

/-- Prove that entity1.property > entity2.property.
    Given two attested values and a decidable comparison. -/
noncomputable def gte_pair
    {T : ToolId} {q0 q1 : Query_T T} {p0 : Payload_T T q0} {p1 : Payload_T T q1}
    {e1 e2 : String} {prop : String} {v1 v2 : Value}
    (att0 : Attested_T T q0 p0) (tag0 : ClaimTag_T T q0 p0)
    (att1 : Attested_T T q1 p1) (tag1 : ClaimTag_T T q1 p1)
    (shape0 : claimWProp tag0 = HasProperty e1 prop v1)
    (shape1 : claimWProp tag1 = HasProperty e2 prop v2)
    (h : ValueGte v1 v2) :
    PProd (Evidence Grade.Verified (HasProperty e1 prop v1))
    (PProd (Evidence Grade.Verified (HasProperty e2 prop v2))
    (PLift (ValueGte v1 v2))) :=
  PProd.mk (shape0 ▸ mkVerified att0 (claimOfTag tag0))
    (PProd.mk (shape1 ▸ mkVerified att1 (claimOfTag tag1))
    (PLift.up h))

/-- Prove that entity1.property > entity2.property (strict). -/
noncomputable def gt_pair
    {T : ToolId} {q0 q1 : Query_T T} {p0 : Payload_T T q0} {p1 : Payload_T T q1}
    {e1 e2 : String} {prop : String} {v1 v2 : Value}
    (att0 : Attested_T T q0 p0) (tag0 : ClaimTag_T T q0 p0)
    (att1 : Attested_T T q1 p1) (tag1 : ClaimTag_T T q1 p1)
    (shape0 : claimWProp tag0 = HasProperty e1 prop v1)
    (shape1 : claimWProp tag1 = HasProperty e2 prop v2)
    (h : ValueGt v1 v2) :
    PProd (Evidence Grade.Verified (HasProperty e1 prop v1))
    (PProd (Evidence Grade.Verified (HasProperty e2 prop v2))
    (PLift (ValueGt v1 v2))) :=
  PProd.mk (shape0 ▸ mkVerified att0 (claimOfTag tag0))
    (PProd.mk (shape1 ▸ mkVerified att1 (claimOfTag tag1))
    (PLift.up h))

/-- Prove equality of two attested values. -/
noncomputable def eq_pair
    {T : ToolId} {q0 q1 : Query_T T} {p0 : Payload_T T q0} {p1 : Payload_T T q1}
    {e1 e2 : String} {prop : String} {v1 v2 : Value}
    (att0 : Attested_T T q0 p0) (tag0 : ClaimTag_T T q0 p0)
    (att1 : Attested_T T q1 p1) (tag1 : ClaimTag_T T q1 p1)
    (shape0 : claimWProp tag0 = HasProperty e1 prop v1)
    (shape1 : claimWProp tag1 = HasProperty e2 prop v2)
    (h : ValueEq v1 v2) :
    PProd (Evidence Grade.Verified (HasProperty e1 prop v1))
    (PProd (Evidence Grade.Verified (HasProperty e2 prop v2))
    (PLift (ValueEq v1 v2))) :=
  PProd.mk (shape0 ▸ mkVerified att0 (claimOfTag tag0))
    (PProd.mk (shape1 ▸ mkVerified att1 (claimOfTag tag1))
    (PLift.up h))

-- ============================================================
-- § 2. Single attestation + property check
-- ============================================================

/-- Attest a single value and prove a property about it. -/
noncomputable def attest_and_check
    {T : ToolId} {q : Query_T T} {p : Payload_T T q}
    {e : String} {prop : String} {v : Value} {P : Prop}
    (att : Attested_T T q p) (tag : ClaimTag_T T q p)
    (shape : claimWProp tag = HasProperty e prop v)
    (h : P) :
    PProd (Evidence Grade.Verified (HasProperty e prop v)) (PLift P) :=
  PProd.mk (shape ▸ mkVerified att (claimOfTag tag)) (PLift.up h)

-- ============================================================
-- § 3. Refutation (prove ¬ claim)
-- ============================================================

/-- Refute a comparison: prove ¬ (v1 > v2). -/
noncomputable def refute_gt
    {T : ToolId} {q0 q1 : Query_T T} {p0 : Payload_T T q0} {p1 : Payload_T T q1}
    {e1 e2 : String} {prop : String} {v1 v2 : Value}
    (att0 : Attested_T T q0 p0) (tag0 : ClaimTag_T T q0 p0)
    (att1 : Attested_T T q1 p1) (tag1 : ClaimTag_T T q1 p1)
    (shape0 : claimWProp tag0 = HasProperty e1 prop v1)
    (shape1 : claimWProp tag1 = HasProperty e2 prop v2)
    (h : ¬ ValueGt v1 v2) :
    PProd (Evidence Grade.Verified (HasProperty e1 prop v1))
    (PProd (Evidence Grade.Verified (HasProperty e2 prop v2))
    (PLift (¬ ValueGt v1 v2))) :=
  PProd.mk (shape0 ▸ mkVerified att0 (claimOfTag tag0))
    (PProd.mk (shape1 ▸ mkVerified att1 (claimOfTag tag1))
    (PLift.up h))

/-- Refute equality: prove v1 ≠ v2. -/
noncomputable def refute_eq
    {T : ToolId} {q0 q1 : Query_T T} {p0 : Payload_T T q0} {p1 : Payload_T T q1}
    {e1 e2 : String} {prop : String} {v1 v2 : Value}
    (att0 : Attested_T T q0 p0) (tag0 : ClaimTag_T T q0 p0)
    (att1 : Attested_T T q1 p1) (tag1 : ClaimTag_T T q1 p1)
    (shape0 : claimWProp tag0 = HasProperty e1 prop v1)
    (shape1 : claimWProp tag1 = HasProperty e2 prop v2)
    (h : ¬ ValueEq v1 v2) :
    PProd (Evidence Grade.Verified (HasProperty e1 prop v1))
    (PProd (Evidence Grade.Verified (HasProperty e2 prop v2))
    (PLift (¬ ValueEq v1 v2))) :=
  PProd.mk (shape0 ▸ mkVerified att0 (claimOfTag tag0))
    (PProd.mk (shape1 ▸ mkVerified att1 (claimOfTag tag1))
    (PLift.up h))

-- ============================================================
-- § 4. Arithmetic Helpers
--
-- The Lean kernel can compute +, -, *, / on Int via `decide`
-- and on Rat via `native_decide`.
-- These helpers combine attestation with arithmetic verification.
-- Int arithmetic: Value.int (a + b), Value.int (a * b) — use `decide`
-- Rat arithmetic: mkRat a b, comparisons — use `native_decide`
-- ============================================================

/-- Attest a value and prove it equals an arithmetic result.
    Example: prove stats_sum = expected total. -/
noncomputable def attest_arith
    {T : ToolId} {q : Query_T T} {p : Payload_T T q}
    {e : String} {prop : String} {v : Value} {result : Value}
    (att : Attested_T T q p) (tag : ClaimTag_T T q p)
    (shape : claimWProp tag = HasProperty e prop v)
    (h : ValueEq v result) :
    PProd (Evidence Grade.Verified (HasProperty e prop v)) (PLift (ValueEq v result)) :=
  PProd.mk (shape ▸ mkVerified att (claimOfTag tag)) (PLift.up h)

/-- Prove arbitrary proposition P about two attested values (possibly different properties).
    Generalizes gt_pair/eq_pair — P can be any decidable arithmetic relation.
    Example: prove obs0 + obs1 = 100, or obs0 * 100 = obs1 * 68 (percentage). -/
noncomputable def arith_check
    {T : ToolId} {q0 q1 : Query_T T} {p0 : Payload_T T q0} {p1 : Payload_T T q1}
    {e1 e2 : String} {prop1 prop2 : String} {v1 v2 : Value} {P : Prop}
    (att0 : Attested_T T q0 p0) (tag0 : ClaimTag_T T q0 p0)
    (att1 : Attested_T T q1 p1) (tag1 : ClaimTag_T T q1 p1)
    (shape0 : claimWProp tag0 = HasProperty e1 prop1 v1)
    (shape1 : claimWProp tag1 = HasProperty e2 prop2 v2)
    (h : P) :
    PProd (Evidence Grade.Verified (HasProperty e1 prop1 v1))
    (PProd (Evidence Grade.Verified (HasProperty e2 prop2 v2))
    (PLift P)) :=
  PProd.mk (shape0 ▸ mkVerified att0 (claimOfTag tag0))
    (PProd.mk (shape1 ▸ mkVerified att1 (claimOfTag tag1))
    (PLift.up h))

-- ============================================================
-- § 5. Aggregate Attestation + Decomposition
--
-- Patterns for working with world-level aggregate predicates
-- (SumWhere, CountWhere, EntitySum, etc.) and decomposing totals.
-- ============================================================

/-- Attest any world fact and prove a property about it.
    Works with SumWhere, CountWhere, MaxWhere, EntitySum, etc. -/
noncomputable def attest_agg
    {T : ToolId} {q : Query_T T} {p : Payload_T T q}
    {w : WProp} {P : Prop}
    (att : Attested_T T q p) (tag : ClaimTag_T T q p)
    (shape : claimWProp tag = w)
    (h : P) :
    PProd (Evidence Grade.Verified w) (PLift P) :=
  PProd.mk (shape ▸ mkVerified att (claimOfTag tag)) (PLift.up h)

/-- Decomposition: given a total and one part, prove an arithmetic relationship.

    Example: EntitySum "brazil" (Value.int 100) and HasProperty "brazil" "medals" (Value.int 10)
    → the remainder is 90 (by 100 - 10 = 90).

    Works for any pair of attested world facts + arithmetic check. -/
noncomputable def sum_decompose
    {T1 : ToolId} {q0 : Query_T T1} {p0 : Payload_T T1 q0}
    {T2 : ToolId} {q1 : Query_T T2} {p1 : Payload_T T2 q1}
    {w_total w_part : WProp} {P : Prop}
    (att0 : Attested_T T1 q0 p0) (tag0 : ClaimTag_T T1 q0 p0)
    (shape0 : claimWProp tag0 = w_total)
    (att1 : Attested_T T2 q1 p1) (tag1 : ClaimTag_T T2 q1 p1)
    (shape1 : claimWProp tag1 = w_part)
    (h : P) :
    PProd (Evidence Grade.Verified w_total)
    (PProd (Evidence Grade.Verified w_part)
    (PLift P)) :=
  PProd.mk (shape0 ▸ mkVerified att0 (claimOfTag tag0))
    (PProd.mk (shape1 ▸ mkVerified att1 (claimOfTag tag1))
    (PLift.up h))

/-- Average verification via cross-multiplication.
    Given SumWhere and CountWhere attestations, prove arithmetic relationship.

    Example: avg = 10.6 → sum = 106, count = 10 →
    prove 106 * 10 = 10 * 106 (cross-multiply to avoid division). -/
noncomputable def avg_cross_multiply
    {T1 : ToolId} {q0 : Query_T T1} {p0 : Payload_T T1 q0}
    {T2 : ToolId} {q1 : Query_T T2} {p1 : Payload_T T2 q1}
    {col : String} {cond : Cond} {v_sum v_count : Value} {P : Prop}
    (att0 : Attested_T T1 q0 p0) (tag0 : ClaimTag_T T1 q0 p0)
    (shape0 : claimWProp tag0 = SumWhere col cond v_sum)
    (att1 : Attested_T T2 q1 p1) (tag1 : ClaimTag_T T2 q1 p1)
    (shape1 : claimWProp tag1 = CountWhere col cond v_count)
    (h : P) :
    PProd (Evidence Grade.Verified (SumWhere col cond v_sum))
    (PProd (Evidence Grade.Verified (CountWhere col cond v_count))
    (PLift P)) :=
  PProd.mk (shape0 ▸ mkVerified att0 (claimOfTag tag0))
    (PProd.mk (shape1 ▸ mkVerified att1 (claimOfTag tag1))
    (PLift.up h))

