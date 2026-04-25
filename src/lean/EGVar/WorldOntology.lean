/-
  EG-VAR World-Level Ontology

  The formal language describes THE WORLD, not data storage.
  Tables, APIs, and databases are tools that provide evidence for world facts.
  The formal language is tool-agnostic.

  Design note (R5/R9/R10 in specs/phase-a-resolutions.md): role slots are
  pure `String`. There is no `Entity` or `Property` type. Categories like
  "entity" and "determinable" are documented in argument names
  (`entity : String`, `property : String`, `measure_col : String`,
  `answer_name : String`) rather than enforced by the type system.

  This is the strict-nominalist position at the kernel boundary: names
  bear identity; semantic roles are descriptive labels, not metaphysical
  kinds. The choice is required for correctness: record tables (rows
  without canonical entity identity) and rank-r data both refuse a
  pre-committed entity-type discipline; nominalism is the only
  architecturally sound position for the actual diversity of tables.
  See R10 for the full architectural argument.

  Per ADR-005, specs/world-level-ontology.md, specs/table-formalization.md,
  and specs/phase-a-resolutions.md (R5/R9/R10).
-/
import EGVar.Basic

set_option autoImplicit false

abbrev Property : Type := String
abbrev Entity : Type := String

-- ============================================================
-- § 1. World-Level Value Type
-- ============================================================

/-- Values use canonical types. Int for exact integers, Rat for exact decimals.
    Rat (rationals) enable exact arithmetic: 463/9 is represented exactly as mkRat 463 9.
    No truncation, no scaling, no precision loss through chained operations.
    Proofs use native_decide (trusts Lean's compiler via ofReduceBool axiom). -/
inductive Value where
  | int (n : Int)
  | rat (r : Rat)
  | str (s : String)
  deriving DecidableEq, Repr

namespace Value

def toInt : Value → Option Int
  | .int n => some n
  | .rat r => if r.den == 1 then some r.num else none
  | .str _ => none

def toRat : Value → Option Rat
  | .int n => some (mkRat n 1)
  | .rat r => some r
  | .str _ => none

def toStr : Value → String
  | .int n => toString n
  | .rat r => s!"{r.num}/{r.den}"
  | .str s => s

/-- Total projection to Rat for numeric values; str values default to 0.
    Enables clean native-arithmetic statements in goal types without
    Option unwrapping, e.g. `ans.asRat = a.asRat - b.asRat`. -/
def asRat : Value → Rat
  | .int n => mkRat n 1
  | .rat r => r
  | .str _ => 0

end Value

/-- A relation name (in_group, born_in, located_in, etc.).
    Out of Tier 1 scope; kept as an axiomatic Type for `Relation`'s middle
    slot. May be migrated to String in a future tier. -/
axiom RelName : Type

-- ============================================================
-- § 2. World-Level Predicates (WProp)
--
-- Per R5/R9/R10: role slots are pure `String`. The argument names carry
-- the semantic role (entity, property, measure_col, answer_name, etc.).
-- ============================================================

/-- Core world fact: an entity has a property with a value.
    E.g., HasProperty "brazil" "goals" (Value.int 15) means "Brazil has 15 goals." -/
axiom HasProperty (entity : String) (property : String) (value : Value) : WProp

/-- Named relation between two entities.
    E.g., Relation "brazil" rel_in_group "group_a" means "Brazil is in Group A." -/
axiom Relation (subject : String) (rel : RelName) (object : String) : WProp

-- ============================================================
-- § 3. Kernel-Derived Predicates (Prop)
--
-- These are proved by the kernel from attested HasProperty evidence.
-- They are in Prop (kernel-checked), not WProp (tool-attested).
-- ============================================================

/-- Value comparison. Operates on numeric values (Int or Rat). -/
def ValueGt (a b : Value) : Prop :=
  match a.toRat, b.toRat with
  | some x, some y => x > y
  | _, _ => False

def ValueLt (a b : Value) : Prop :=
  match a.toRat, b.toRat with
  | some x, some y => x < y
  | _, _ => False

def ValueEq (a b : Value) : Prop := a = b

def ValueGte (a b : Value) : Prop :=
  match a.toRat, b.toRat with
  | some x, some y => x ≥ y
  | _, _ => False

def ValueLte (a b : Value) : Prop :=
  match a.toRat, b.toRat with
  | some x, some y => x ≤ y
  | _, _ => False

instance {a b : Value} : Decidable (ValueGt a b) := by
  unfold ValueGt; cases a.toRat <;> cases b.toRat <;> simp <;> exact inferInstance

instance {a b : Value} : Decidable (ValueLt a b) := by
  unfold ValueLt; cases a.toRat <;> cases b.toRat <;> simp <;> exact inferInstance

instance {a b : Value} : Decidable (ValueEq a b) := inferInstanceAs (Decidable (a = b))

instance {a b : Value} : Decidable (ValueGte a b) := by
  unfold ValueGte; cases a.toRat <;> cases b.toRat <;> simp <;> exact inferInstance

instance {a b : Value} : Decidable (ValueLte a b) := by
  unfold ValueLte; cases a.toRat <;> cases b.toRat <;> simp <;> exact inferInstance

-- ============================================================
-- § 4. Evidence Minting for World Facts
--
-- The tool adapter returns attested world facts via mkVerified.
-- The Lean file emits: Evidence Verified (HasProperty entity property value)
-- ============================================================

-- mkVerified (from Basic.lean) works directly:
-- mkVerified att mem : Evidence Grade.Verified (HasProperty entity property value)
-- where att : Attested_T and mem : ClaimOf (HasProperty ...) (interp_T ...)

-- ============================================================
-- § 5. World-Level Aggregate Predicates (WProp)
--
-- Express aggregate world facts without referencing table structure.
-- The tool computes these from whatever data source it has;
-- the formal language only records the world-level result.
--
-- Attested by the trusted stats tool via mkVerified.
-- Same ToolId/Query_T/Attested_T infrastructure as the table tool.
-- ============================================================

/-- Condition for filtering entities in aggregation.
    A composable predicate that the tool can evaluate. The `column` slot
    names the column being constrained (e.g. "rank", "score"). -/
inductive Cond where
  | propEq       (column : String) (value : Value) : Cond  -- column = value
  | propGt       (column : String) (value : Value) : Cond  -- column > value
  | propLt       (column : String) (value : Value) : Cond  -- column < value
  | propGte      (column : String) (value : Value) : Cond  -- column ≥ value
  | propLte      (column : String) (value : Value) : Cond  -- column ≤ value
  | propContains (column : String) (value : Value) : Cond  -- column contains substring
  | and          : Cond → Cond → Cond                      -- conjunction
  | or           : Cond → Cond → Cond                      -- disjunction
  | not          : Cond → Cond                             -- negation
  | all          : Cond                                     -- no filter

/-- Sum of `measure_col` across entities/rows satisfying `cond`.
    E.g., SumWhere "population" Cond.all (Value.int 325000000)
    means "the total population is 325 million." -/
axiom SumWhere (measure_col : String) (cond : Cond) (value : Value) : WProp

/-- Count of entities/rows satisfying `cond` that have `measure_col`. -/
axiom CountWhere (measure_col : String) (cond : Cond) (count : Value) : WProp

/-- Count of distinct values of `measure_col` among entities satisfying `cond`. -/
axiom CountDistinctWhere (measure_col : String) (cond : Cond) (count : Value) : WProp

/-- Mean of `measure_col` across entities/rows satisfying `cond`.
    E.g., MeanWhere "age" Cond.all (Value.rat (mkRat 463 9))
    means "the average age is 463/9 (exact rational)." -/
axiom MeanWhere (measure_col : String) (cond : Cond) (value : Value) : WProp

/-- Maximum value of `measure_col` among entities/rows satisfying `cond`. -/
axiom MaxWhere (measure_col : String) (cond : Cond) (value : Value) : WProp

/-- Minimum value of `measure_col` among entities/rows satisfying `cond`. -/
axiom MinWhere (measure_col : String) (cond : Cond) (value : Value) : WProp

/-- The entity (named by `answer_name`) with the maximum value of `measure_col`
    among those satisfying `cond`.
    E.g., ArgmaxWhere "population" Cond.all "tokyo" (Value.int 37000000)
    means "tokyo has the largest population (37M)."

    Per R10.6, this is a level-3 (entity-anchored) predicate; using it commits
    to the per-table interpretation that rows are entities and the answer_name
    is the canonical name read from the identity column. Per-table lifts that
    produce ArgmaxWhere encode that commitment in their premise structure. -/
axiom ArgmaxWhere
  (measure_col : String) (cond : Cond)
  (answer_name : String) (value : Value) : WProp

/-- The entity (named by `answer_name`) with the minimum value of `measure_col`
    among those satisfying `cond`. -/
axiom ArgminWhere
  (measure_col : String) (cond : Cond)
  (answer_name : String) (value : Value) : WProp

/-- Sum of all properties for a single entity.
    E.g., EntitySum "brazil" (Value.int 100)
    means "the sum of all (numeric) property values for brazil is 100." -/
axiom EntitySum (entity : String) (value : Value) : WProp

-- ============================================================
-- § 5b. Top-K selection operators
--
-- Rank entities by a column (descending) and aggregate over the top k.
-- Enables questions like:
--   "average height of the 5 tallest buildings"     → TopKMeanWhere "height" 5 Cond.all ans
--   "sum pop of top 5 countries by GDP"              → TopKSumByWhere "pop" "gdp" 5 Cond.all ans
--                                                       (target=pop, rank_by=gdp; argument order matches the axiom signature below)
-- Bottom variants select the smallest k values instead.
-- ============================================================

/-- Sum of the K largest values of `measure_col` (after `cond` filter). -/
axiom TopKSumWhere
  (measure_col : String) (k : Nat) (cond : Cond) (value : Value) : WProp

/-- Mean of the K largest values of `measure_col`. -/
axiom TopKMeanWhere
  (measure_col : String) (k : Nat) (cond : Cond) (value : Value) : WProp

/-- Sum of `target_col` over the K entities with largest `rank_by_col`. -/
axiom TopKSumByWhere
  (target_col : String) (rank_by_col : String)
  (k : Nat) (cond : Cond) (value : Value) : WProp

/-- Mean of `target_col` over the K entities with largest `rank_by_col`. -/
axiom TopKMeanByWhere
  (target_col : String) (rank_by_col : String)
  (k : Nat) (cond : Cond) (value : Value) : WProp

/-- Sum of the K smallest values of `measure_col`. -/
axiom BottomKSumWhere
  (measure_col : String) (k : Nat) (cond : Cond) (value : Value) : WProp

/-- Mean of the K smallest values of `measure_col`. -/
axiom BottomKMeanWhere
  (measure_col : String) (k : Nat) (cond : Cond) (value : Value) : WProp

-- ============================================================
-- § 6. Aggregation Helpers (kernel-computed from evidence)
-- ============================================================

/-- Sum of a list of integer values. Kernel-computable. -/
def sumValues (vs : List Value) : Int :=
  vs.foldl (fun acc v => match v.toInt with | some n => acc + n | none => acc) 0

/-- Count of values satisfying a predicate. Kernel-computable. -/
def countWhere (vs : List Value) (pred : Value → Bool) : Nat :=
  vs.filter pred |>.length

/-- Maximum of a list of integer values. Kernel-computable. -/
def maxValue (vs : List Value) : Option Int :=
  vs.foldl (fun acc v => match v.toInt, acc with
    | some n, some m => some (max n m)
    | some n, none => some n
    | none, acc => acc) none

/-- Minimum of a list of integer values. Kernel-computable. -/
def minValue (vs : List Value) : Option Int :=
  vs.foldl (fun acc v => match v.toInt, acc with
    | some n, some m => some (min n m)
    | some n, none => some n
    | none, acc => acc) none
