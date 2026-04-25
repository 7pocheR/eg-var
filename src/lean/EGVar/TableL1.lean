/-
  EG-VAR L1 Primitives — Table Storage Representation

  This module declares the L1 attestation vocabulary the runtime tool
  adapter uses for 2D flat tables: row handles plus cell-level
  certificates (`CellAt`) and column-level aggregate certificates
  (`MaxCell`, `SumCells`, etc.).

  Per R10 (specs/phase-a-resolutions.md): L1 carries STORAGE
  REPRESENTATION — indices the tool needs to find cells. L2 (in
  WorldOntology.lean) carries WORLD FACTS — predicates with no storage
  artifacts. Per-table lift lemmas (in fixture `table_formalization`
  blobs) bridge L1 and L2 by encoding the curator's interpretive
  commitment for that table.

  This module declares only the L1 vocabulary itself — plain axioms,
  no proofs, no lift logic. All lifts live in per-table formalization
  blobs (per spec §3.1 trust boundary).

  Per R10.4: higher-rank storage shapes (3D cubes, sparse matrices,
  federated datasets) would extend this module with new primitives
  (`Handle3D`, `CellAt3`, `SparseHandle`, etc.). They do not enter L2
  or alter the trust boundary.
-/
import EGVar.Basic
import EGVar.WorldOntology  -- for Cond

set_option autoImplicit false

-- ============================================================
-- § 1. Row identity
-- ============================================================

/-- An opaque row identifier. The runtime tool adapter mints instances
    per row of a specific table; handles are stable across tool calls
    on the same table. The kernel does not enumerate or compare
    RowHandles — it only composes attestations indexed by them. -/
axiom RowHandle : Type

-- ============================================================
-- § 2. Cell-level certificates
-- ============================================================

/-- L1: the cell at row `r`, column `col` of some table has value `v`.
    Attested by the tool's cell-lookup operation. -/
axiom CellAt (r : RowHandle) (col : String) (v : Value) : WProp

-- ============================================================
-- § 3. Column-level aggregate certificates (value-only — universally safe)
--
-- These attest the result of an aggregation operation over a column,
-- filtered by `cond`. They produce *value-only* L1 facts: no entity
-- identification, no row anchor in the conclusion. Safe for any table
-- (R10.6 level 1) — the lifts to value-only L2 predicates (SumWhere,
-- CountWhere, MeanWhere, MaxWhere, MinWhere) are universally
-- applicable.
-- ============================================================

/-- L1: the sum of `col` values across rows satisfying `cond` is `v`. -/
axiom SumCells (col : String) (cond : Cond) (v : Value) : WProp

/-- L1: the count of rows satisfying `cond` (with non-null `col`) is `v`. -/
axiom CountCells (col : String) (cond : Cond) (v : Value) : WProp

/-- L1: the count of distinct values of `col` across rows satisfying
    `cond` is `v`. -/
axiom CountDistinctCells (col : String) (cond : Cond) (v : Value) : WProp

/-- L1: the mean of `col` values across rows satisfying `cond` is `v`
    (exact rational). -/
axiom MeanCells (col : String) (cond : Cond) (v : Value) : WProp

/-- L1: the maximum value of `col` across rows satisfying `cond` is `v`
    (value-only — no row identification). Use `MaxCell` (below) when the
    answer row's identity matters. -/
axiom MaxCellValue (col : String) (cond : Cond) (v : Value) : WProp

/-- L1: the minimum value of `col` across rows satisfying `cond` is `v`
    (value-only). Counterpart to `MinCell` (below). -/
axiom MinCellValue (col : String) (cond : Cond) (v : Value) : WProp

-- ============================================================
-- § 4. Row-identifying aggregate certificates (entity-projection-ready)
--
-- These attest both the aggregate result AND the row achieving it.
-- They are the L1 premises a per-table lift uses to produce
-- entity-anchored L2 predicates (`ArgmaxWhere`, `ArgminWhere`).
-- The `RowHandle` slot exposes which row the tool found; the lift
-- combines this with a `CellAt` at the table's identity column to
-- bind the entity name in the L2 conclusion (R10.6 level 3).
-- ============================================================

/-- L1: under `cond`, row `r` achieves the maximum value `v` for column
    `col`. Does NOT specify the entity name — that comes from a paired
    `CellAt r identity_col (Value.str name)` premise via the per-table
    lift. -/
axiom MaxCell (col : String) (cond : Cond) (r : RowHandle) (v : Value) : WProp

/-- L1: under `cond`, row `r` achieves the minimum value `v` for column
    `col`. -/
axiom MinCell (col : String) (cond : Cond) (r : RowHandle) (v : Value) : WProp

-- ============================================================
-- § 5. Top-K aggregate certificates (value-only)
-- ============================================================

/-- L1: the sum of the K largest values of `col` (after `cond` filter)
    is `v`. -/
axiom TopKSumCells
  (col : String) (k : Nat) (cond : Cond) (v : Value) : WProp

/-- L1: the mean of the K largest values of `col` is `v`. -/
axiom TopKMeanCells
  (col : String) (k : Nat) (cond : Cond) (v : Value) : WProp

/-- L1: the sum of `target_col` over the K rows with largest
    `rank_by_col` is `v`. -/
axiom TopKSumByCells
  (target_col : String) (rank_by_col : String)
  (k : Nat) (cond : Cond) (v : Value) : WProp

/-- L1: the mean of `target_col` over the K rows with largest
    `rank_by_col` is `v`. -/
axiom TopKMeanByCells
  (target_col : String) (rank_by_col : String)
  (k : Nat) (cond : Cond) (v : Value) : WProp

/-- L1: the sum of the K smallest values of `col` is `v`. -/
axiom BottomKSumCells
  (col : String) (k : Nat) (cond : Cond) (v : Value) : WProp

/-- L1: the mean of the K smallest values of `col` is `v`. -/
axiom BottomKMeanCells
  (col : String) (k : Nat) (cond : Cond) (v : Value) : WProp
