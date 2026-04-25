/-
  EG-VAR Core Definitions
  WProp, Grade, Evidence, Attested, mkVerified, mkPlausible, downcast
-/
set_option autoImplicit false

-- ============================================================
-- § 1. World Propositions (Phase Separation)
-- ============================================================

axiom WProp : Type
axiom WAnd : WProp → WProp → WProp
axiom WOr  : WProp → WProp → WProp
axiom WNot : WProp → WProp

-- ============================================================
-- § 2. Evidence Grades
-- ============================================================

/-- Evidence grades. Speculative is first so Inhabited defaults to lowest grade. -/
inductive Grade where
  | Speculative
  | Plausible
  | Supported
  | Verified
  deriving DecidableEq, Repr, Inhabited

namespace Grade

@[reducible] def toNat : Grade → Nat
  | .Speculative => 0
  | .Plausible   => 1
  | .Supported   => 2
  | .Verified    => 3

@[reducible] instance : LE Grade where le a b := a.toNat ≤ b.toNat
@[reducible] instance : LT Grade where lt a b := a.toNat < b.toNat

instance : DecidableRel (α := Grade) (· ≤ ·) := fun a b => Nat.decLe a.toNat b.toNat
instance : DecidableRel (α := Grade) (· < ·) := fun a b => Nat.decLt a.toNat b.toNat

-- Order properties
theorem le_refl (a : Grade) : a ≤ a := Nat.le_refl _
theorem le_trans {a b c : Grade} (h1 : a ≤ b) (h2 : b ≤ c) : a ≤ c := Nat.le_trans h1 h2

theorem le_antisymm {a b : Grade} (h1 : a ≤ b) (h2 : b ≤ a) : a = b := by
  cases a <;> cases b <;>
    first | rfl | exact absurd h1 (by decide) | exact absurd h2 (by decide)

theorem speculative_bot (g : Grade) : .Speculative ≤ g := by cases g <;> decide
theorem verified_top (g : Grade) : g ≤ .Verified := by cases g <;> decide

-- Meet = min, Join = max
@[reducible] def meet (a b : Grade) : Grade := if a ≤ b then a else b
@[reducible] def join (a b : Grade) : Grade := if a ≤ b then b else a

-- Lattice laws (all decidable on 4 elements)
theorem meet_comm (a b : Grade) : meet a b = meet b a := by
  cases a <;> cases b <;> decide
theorem join_comm (a b : Grade) : join a b = join b a := by
  cases a <;> cases b <;> decide
theorem meet_assoc (a b c : Grade) : meet (meet a b) c = meet a (meet b c) := by
  cases a <;> cases b <;> cases c <;> decide
theorem join_assoc (a b c : Grade) : join (join a b) c = join a (join b c) := by
  cases a <;> cases b <;> cases c <;> decide
theorem meet_idem (a : Grade) : meet a a = a := by cases a <;> decide
theorem join_idem (a : Grade) : join a a = a := by cases a <;> decide
theorem meet_join_absorb (a b : Grade) : meet a (join a b) = a := by
  cases a <;> cases b <;> decide
theorem join_meet_absorb (a b : Grade) : join a (meet a b) = a := by
  cases a <;> cases b <;> decide
theorem meet_join_distrib (a b c : Grade) :
    meet a (join b c) = join (meet a b) (meet a c) := by
  cases a <;> cases b <;> cases c <;> decide
theorem meet_le_left (a b : Grade) : meet a b ≤ a := by
  cases a <;> cases b <;> decide
theorem meet_le_right (a b : Grade) : meet a b ≤ b := by
  cases a <;> cases b <;> decide

end Grade

-- ============================================================
-- § 3. Evidence Type Family
-- ============================================================

axiom Evidence : Grade → WProp → Type

/-- Downcast: weaken evidence to a LOWER grade. h : g₂ ≤ g₁ prevents upcast. -/
axiom downcast {w : WProp} {g₁ g₂ : Grade} (h : g₂ ≤ g₁) :
  Evidence g₁ w → Evidence g₂ w

-- ============================================================
-- § 4. Tool Infrastructure (§3.5 of EG-VAR.md)
--
-- Naming convention: Lean names follow EG-VAR.md subscript notation
--   Query_T  → Query_T    (spec: Query_T)
--   Payload_T → Payload_T (spec: Payload_T)
--   Attested_T → Attested_T (spec: Attested_T)
--   WorldRec_Σ → WorldRec  (spec: WorldRec_Σ; Σ-indexing deferred)
--   interp_T → interp_T   (spec: interp_T)
--   claims membership → ClaimOf (spec: w ∈ claims r)
-- ============================================================

/-- Tool identifier (§2.1). -/
axiom ToolId : Type

/-- Query type for tool T (§3.5: Query_T). -/
axiom Query_T : ToolId → Type

/-- Payload type for tool T and query q (§3.5: Payload_T). -/
axiom Payload_T : (T : ToolId) → Query_T T → Type

/-- Attestation proof (§3.5: Attested_T). Runtime-only constructor. -/
axiom Attested_T : (T : ToolId) → (q : Query_T T) → Payload_T T q → Type

/-- World record under ontology Σ (§3.5: WorldRec_Σ).
    Σ-indexing deferred to Phase 2. -/
axiom WorldRec : Type

/-- Interpretation function (§3.5: interp_T). -/
axiom interp_T : (T : ToolId) → (q : Query_T T) → Payload_T T q → WorldRec

/-- Claim membership: w is among the claims of record r (§3.5: w ∈ claims r).
    In Type (not Prop) to preserve phase separation. -/
axiom ClaimOf : WProp → WorldRec → Type

/-- Structured claim selector for a concrete attested payload.
    The runtime emits tags/selectors as data; it does NOT synthesize arbitrary
    Lean proof terms for `ClaimOf`. -/
axiom ClaimTag_T : (T : ToolId) → (q : Query_T T) → Payload_T T q → Type

/-- The world proposition denoted by a payload-specific claim tag. -/
axiom claimWProp
  {T : ToolId} {q : Query_T T} {p : Payload_T T q} :
  ClaimTag_T T q p → WProp

/-- Soundness bridge from a structured claim tag to `ClaimOf`.
    This is the runtime/kernel bridge for payload-to-claims membership. -/
axiom claimOfTag
  {T : ToolId} {q : Query_T T} {p : Payload_T T q}
  (tag : ClaimTag_T T q p) :
  ClaimOf (claimWProp tag) (interp_T T q p)

/-- THE verified minting rule (§3.5).
    Sole source of Evidence Verified. Requires:
    - att : Attested_T (only from runtime)
    - mem : ClaimOf (w is a claim of the interpreted payload) -/
axiom mkVerified
  {T : ToolId} {q : Query_T T} {p : Payload_T T q}
  (att : Attested_T T q p)
  {w : WProp}
  (mem : ClaimOf w (interp_T T q p)) :
  Evidence Grade.Verified w

-- ============================================================
-- § 4b. Common Sense / World Knowledge (lower-graded evidence)
--
-- Same pattern as tool infrastructure but at Plausible grade.
-- The common sense agent is a less-trusted "tool" — an LLM that
-- answers factual questions. Its attestation is minted by the
-- pipeline only when the agent actually responded.
-- ============================================================

/-- Query to the common sense agent (e.g., "Is united states in north america?"). -/
axiom CSQuery : Type

/-- Attestation that the common sense agent answered a query with a proposition.
    Minted by the pipeline, not forgeable by the solver. -/
axiom CSAttestation : CSQuery → WProp → Type

/-- Mint Plausible evidence from a common sense attestation.
    Same pattern as mkVerified but at Plausible grade — no tool trust. -/
axiom mkPlausible
  {q : CSQuery} {w : WProp}
  (att : CSAttestation q w) :
  Evidence Grade.Plausible w

-- ============================================================
-- § 5. Time and Freshness
-- ============================================================

abbrev Timestamp := Nat
abbrev Duration := Nat
axiom At : Timestamp → WProp → WProp

/-- Freshness: t must be in the past AND within δ of now.
    Guards against future-timestamp exploit (Nat subtraction saturates at 0). -/
def FreshEnough (t : Timestamp) (δ : Duration) (now : Timestamp) : Prop :=
  t ≤ now ∧ now - t ≤ δ

instance {t δ now : Nat} : Decidable (FreshEnough t δ now) :=
  inferInstanceAs (Decidable (_ ∧ _))
