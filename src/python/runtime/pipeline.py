"""
EG-VAR pipeline.

The solver is an LLM in a loop that:
  - proposes lookups (tool executes, evidence added)
  - writes Lean proof attempts (kernel checks, error fed back if rejected)
  - retries until kernel accepts or gives up

No hardcoded proof templates. The solver writes the proofs; the
kernel is the sole judge of correctness.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Callable, Any

from table_adapter import (
    TableData, load_tabfact_table, schema, cell_lookup,
    find_rows, column_values, SchemaResult,
    stats_count_distinct, stats_sum, stats_max, stats_min,
    stats_mean, stats_argmax, stats_argmin,
    # Several legacy stats ops (stats_count, stats_top_n, stats_median,
    # stats_computed_argmax/min, stats_lag_diff, stats_pct_change) are
    # unreachable via REMOVED_OPS hard-reject; the adapter functions
    # remain for legacy fixtures but are intentionally not imported
    # here so the live dispatch surface stays honest.
)
from dictionary import (
    Dictionary, DictEntry, Alias, init_core_dictionary, init_table_dictionary,
)
from feedback import RefinementSignal, CoverageReport

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LEAN_DIR = PROJECT_ROOT / "src" / "lean"
GENERATED_DIR = PROJECT_ROOT / "results" / "generated"

MAX_SOLVER_STEPS = 20          # upper bound per claim
MAX_FORMALIZER_RETRIES = 3     # feedback loop retries per formalizer
STAGNATION_THRESHOLD = 6       # give up after N consecutive non-progress steps
MAX_REFINEMENT_RETRIES = 1     # variant F: at most one retry

# ============================================================
# Gold formalization fixture
# ============================================================
#
# Fixture format: JSON list of triplets.
#
#   [
#     {
#       "table_id": "29ba5...",
#       "question": "What is the average number of tropical cyclones per season?",
#       "lean_goal_type": "Σ (ans : Value), Evidence Grade.Verified (MeanWhere «tropical cyclones» Cond.all ans)"
#     },
#     ...
#   ]
#
# Each entry is a (table, question, formalization) triplet. Lookup
# is by exact (table_id, question) match. Strict lookup: if a claim
# has no matching entry, run_pipeline raises. No fallback to LLM
# formalization, either the fixture has the claim or the eval
# errors out.


# ---------------------------------------------------------------------------
# REMOVED_OPS: single source of truth for the documented stats surface.
#
# Both the live solver dispatch (`run_solver_loop`) and the validator
# path consult this table; drift between the two has historically been
# a dominant regression class.
#
# Key = removed op name. Value = (replacement_advice, semantic_caveat).
#   - replacement_advice: short string with the canonical L1+L2-dispatched
#     alternative for the solver to use instead.
#   - semantic_caveat: any *non-equivalence* the solver must understand
#     (e.g., median is NOT mean; topk_* return values, NOT entity lists).
#     Empty string when the replacement is a clean drop-in.
#
# `removed_op_message` is the single rendering function so both live
# dispatch and validator emit identical text by construction.
# ---------------------------------------------------------------------------
REMOVED_OPS: dict[str, tuple[str, str]] = {
    "count":           ("use 'count_where' with {op:'all'}",
                        "(semantically equivalent — bare 'count' was the same as count_where with no filter)"),
    "count_rows":      ("use 'count_where' with {op:'all'}",
                        "(semantically equivalent to bare 'count')"),
    "median":          ("no L1+L2 lift exists; if a claim genuinely needs median, escalate",
                        "(NOT equivalent to mean = sum/count — median is the middle value, not the average)"),
    "top_n":           ("if the goal asks for an aggregated value (sum-of-top-k or mean-of-top-k), use topk_sum/topk_mean/bottomk_sum/bottomk_mean",
                        "(NOT a drop-in — legacy top_n returned an ENTITY LIST; topk_* return aggregated VALUES; only use if the goal asks for the value, not the list)"),
    "computed_argmax": ("reformulate as PProd composite (lookup numerator + lookup denominator + arithmetic)",
                        ""),
    "computed_argmin": ("reformulate as PProd composite (lookup numerator + lookup denominator + arithmetic)",
                        ""),
    "lag_diff":        ("reformulate via two cell_lookups + subtract (PProd composite)",
                        ""),
    "pct_change":      ("reformulate via two cell_lookups + ratio (PProd composite)",
                        ""),
}


def removed_op_message(op: str) -> str:
    """Render the canonical rejection message for a removed op.
    Single source of truth: both live dispatch and validator call
    this so the text is identical by construction.
    """
    advice, caveat = REMOVED_OPS[op]
    msg = f"ERROR: operation '{op}' is not in the documented stats surface. {advice}."
    if caveat:
        msg = f"{msg} {caveat}"
    return msg


# ---------------------------------------------------------------------------
# Trace contract.
#
# Every tool-call step persisted by the solver loop carries:
#   raw_action: the parsed solver JSON action (with full payload,
#               including lookup_batch.lookups; not just a count).
#   outcome:    structured success metadata (not an English feedback
#               string).
#
# Schema by action type:
#   stats        -> {"success": bool, "n_evidence_added": int}
#   lookup       -> {"success": bool, "n_evidence_added": int}
#   lookup_batch -> {"success": bool, "n_succeeded": int,
#                    "n_total": int,
#                    "per_entry_status": list[str]}
#
# Non-tool steps (parse_error, stagnation, give_up, prove,
# ask_common_sense, unknown) omit raw_action and outcome (presence
# test, not None-check) so downstream consumers can distinguish tool
# calls structurally.
#
# per_entry_status is a closed enum; extending it requires touching
# the audit script and the per-claim CSV diff:
PER_ENTRY_STATUS_OK            = "OK"             # entry succeeded, evidence added
PER_ENTRY_STATUS_NOT_FOUND     = "NOT_FOUND"      # entity not found in identity column
PER_ENTRY_STATUS_OFF_MENU      = "OFF_MENU"       # rejected by enforcement (only when enforce_menu=True)
PER_ENTRY_STATUS_MALFORMED     = "MALFORMED"      # entry not a dict / missing required field
PER_ENTRY_STATUS_ADAPTER_ERROR = "ADAPTER_ERROR"  # adapter raised exception
PER_ENTRY_STATUS_VALUES = frozenset({
    PER_ENTRY_STATUS_OK,
    PER_ENTRY_STATUS_NOT_FOUND,
    PER_ENTRY_STATUS_OFF_MENU,
    PER_ENTRY_STATUS_MALFORMED,
    PER_ENTRY_STATUS_ADAPTER_ERROR,
})


def _append_tool_step(steps: list, action_type: str, action: dict,
                      success: bool, *, evidence_added: int = 0,
                      per_entry_status: list[str] | None = None,
                      **extra) -> None:
    """Append a tool-call step to `steps` with the structured trace
    contract (raw_action + outcome). Use this instead of bare
    ``steps.append({...})`` for stats / lookup / lookup_batch so
    downstream consumers can read structured fields rather than
    parsing English feedback.

    `extra` carries action-type-specific fields (operation, column,
    entity, property, feedback, ...) preserved for human inspection.
    The trace contract fields (raw_action, outcome) are
    machine-readable.
    """
    outcome: dict = {"success": success, "n_evidence_added": evidence_added}
    if per_entry_status is not None:
        # Validate vocabulary (defensive — surface drift fast).
        bad = [s for s in per_entry_status if s not in PER_ENTRY_STATUS_VALUES]
        if bad:
            raise ValueError(f"per_entry_status contains invalid values {bad}; "
                             f"allowed: {sorted(PER_ENTRY_STATUS_VALUES)}")
        outcome["n_total"] = len(per_entry_status)
        outcome["n_succeeded"] = sum(1 for s in per_entry_status if s == PER_ENTRY_STATUS_OK)
        outcome["per_entry_status"] = per_entry_status
    steps.append({"action": action_type, "raw_action": action,
                  "outcome": outcome, **extra})


# ---------------------------------------------------------------------------
# Shared topk rank_by gates.
#
# Same single-source-of-truth pattern as REMOVED_OPS: define the
# constraint constants and rejection-message renderers here, then
# import from the validator.
# ---------------------------------------------------------------------------
TOPK_BY_OPS: tuple[str, ...] = ("topk_sum_by", "topk_mean_by")
TOPK_NON_BY_OPS: tuple[str, ...] = ("topk_sum", "topk_mean",
                                     "bottomk_sum", "bottomk_mean")
TOPK_ALL_OPS: tuple[str, ...] = TOPK_NON_BY_OPS + TOPK_BY_OPS


def rank_by_leak_message(op: str) -> str:
    """Rejection text for non-`_by` topk that supplies rank_by.
    Without this gate the runtime would rank by one column while the
    emitted L2 wprop drops rank_by, producing a false statement.
    """
    return (f"ERROR: operation {op!r} does NOT accept rank_by. "
            f"Use {op}_by (e.g., 'topk_sum_by') if you need to "
            f"rank by a column different from the aggregation column. "
            f"Without _by suffix, ranking and aggregation are the SAME column.")


def missing_rank_by_message(op: str, rank_by: object) -> str:
    """Rejection text for `_by` topk with empty rank_by. Adapter
    would silently fall back to the aggregation column while L1
    emission writes rank="", producing a false storage premise.
    """
    return (f"ERROR: operation {op!r} REQUIRES a non-empty "
            f"`rank_by` (the column to rank by, distinct from the "
            f"target column being aggregated). Got rank_by={rank_by!r}. "
            f"If you want to rank by the same column, use "
            f"{op.replace('_by','')} instead (no _by suffix).")


def check_topk_rank_by(op: str, rank_by: object) -> tuple[str | None, str | None]:
    """Single-source-of-truth gate for topk rank_by constraints.

    Returns (error_message, cleaned_rank_by). On accept:
    (None, cleaned_str | None). On reject: (message, None). Both
    live dispatch and the validator call this so behavior is
    symbol-identical.

    Constraints:
      - non-`_by` topk MUST NOT receive rank_by.
      - `_by` topk MUST receive a non-empty rank_by string.
    """
    if op in TOPK_NON_BY_OPS and rank_by:
        return rank_by_leak_message(op), None
    if op in TOPK_BY_OPS:
        rb_clean = (rank_by or "").strip() if isinstance(rank_by, str) else ""
        if not rb_clean:
            return missing_rank_by_message(op, rank_by), None
        return None, rb_clean
    # Operation isn't in either set, or non-`_by` with no rank_by — no gate.
    return None, (rank_by if isinstance(rank_by, str) else None)


def validate_gold_fixture(fixture: object) -> None:
    """Validate a gold formalization fixture.

    Raises ``ValueError`` on any violation. Must be called at
    fixture-load time, before any LLM call, so malformed fixtures
    fail fast.

    Expected shape: a JSON list of objects, each with required
    string fields ``table_id``, ``question``, ``lean_goal_type``.
    """
    if not isinstance(fixture, list):
        raise ValueError(
            f"fixture must be a JSON list of triplets, "
            f"got {type(fixture).__name__}"
        )
    for i, entry in enumerate(fixture):
        if not isinstance(entry, dict):
            raise ValueError(f"fixture entry {i} must be an object")
        for field in ("table_id", "question", "lean_goal_type"):
            if field not in entry:
                raise ValueError(f"fixture entry {i} missing required field {field!r}")
            if not isinstance(entry[field], str) or not entry[field].strip():
                raise ValueError(
                    f"fixture entry {i} field {field!r} must be a non-empty string"
                )


_PER_TABLE_FORMALIZATIONS: dict | None = None
_TOOL_LIBRARY: dict | None = None


def _load_per_table_formalizations() -> dict:
    """Load and cache the per-table formalizations file. Returns a
    dict keyed by table_id; empty dict if the file is missing.
    """
    global _PER_TABLE_FORMALIZATIONS
    if _PER_TABLE_FORMALIZATIONS is None:
        path = PROJECT_ROOT / "data" / "tier1_table_formalizations.json"
        if path.exists():
            data = json.loads(path.read_text())
            _PER_TABLE_FORMALIZATIONS = {
                k: v for k, v in data.items() if not k.startswith("_")
            }
        else:
            _PER_TABLE_FORMALIZATIONS = {}
    return _PER_TABLE_FORMALIZATIONS


def _extract_tf_hint(table_id: str) -> str:
    """Extract the per-table interpretation string for the
    anomaly-aware formalizer path.

    Returns the interpretation prose (truncated to ~600 chars). The
    same text is also shown to the solver via the
    per-table-formalization Lean comment, so threading it into the
    formalizer prompt under the opt-in `anomaly_aware` flag
    preserves visibility symmetry.

    Note: ``interpretation`` is curator prose and may contain
    claim-shaped hints; prefer ``_format_menu_hint``, which returns
    only the structural sound-operations subset.
    """
    pt = _load_per_table_formalizations()
    entry = pt.get(table_id) if pt else None
    if not entry:
        return ""
    # Accept both dict (structured form) and string (legacy single-blob).
    if isinstance(entry, str):
        return entry[:600]
    if isinstance(entry, dict):
        interp = entry.get("interpretation", "")
        if isinstance(interp, str) and interp:
            return interp[:600]
    return ""


def _format_menu_hint(table_id: str) -> str:
    """Format the per-table sound-operations menu for the
    formalizer prompt.

    Extracts the ``tools`` array (a list of
    {tool_instance_id, library_ref, column}) from the per-table
    formalizations file. Unlike ``_extract_tf_hint``, this returns
    only the structural fact ``which library operations are sound
    on this table's columns'' with no claim-specific prose. The
    menu itself is structurally derivable from the table; the
    curator pre-computed it offline for this eval.

    Output format (feeds into the formalizer's user prompt):
        PER-TABLE MENU (sound operations on this table):
          - stats_mean_strict on column "local magnitude"
          - stats_min_strict on column "local magnitude"
          ...

    Returns "" if no menu is available for this table_id.
    """
    pt = _load_per_table_formalizations()
    entry = pt.get(table_id) if pt else None
    if not entry or not isinstance(entry, dict):
        return ""
    tools = entry.get("tools", [])
    if not tools or not isinstance(tools, list):
        return ""
    lines = [
        "PER-TABLE MENU (sound operations on this table; R10-audited "
        "structural subset — emit predicates only from this menu; compose "
        "from listed primitives if your preferred op is absent):"
    ]
    for t in tools:
        if not isinstance(t, dict):
            continue
        lib = t.get("library_ref", "?")
        col = t.get("column", "")
        tid = t.get("tool_instance_id", "")
        if col:
            lines.append(f"  - {lib} on column «{col}»  [id: {tid}]")
        else:
            lines.append(f"  - {lib}  [id: {tid}]")
    return "\n".join(lines)


def _load_tool_library() -> dict:
    """Load and cache the tool library file. Returns a dict keyed by
    tool_id; empty dict if the file is missing.
    """
    global _TOOL_LIBRARY
    if _TOOL_LIBRARY is None:
        path = PROJECT_ROOT / "data" / "tier1_tool_library.json"
        if path.exists():
            data = json.loads(path.read_text())
            _TOOL_LIBRARY = {
                k: v for k, v in data.items() if not k.startswith("_")
            }
        else:
            _TOOL_LIBRARY = {}
    return _TOOL_LIBRARY


def _resolve_tool_instance(table_id: str, operation: str, column: str | None = None) -> dict | None:
    """Find the per-table tool instance matching the requested
    operation.

    Each table's tool menu lists tool instances
    (tool_instance_id, library_ref, column, optional
    config_override, optional identity_column). The runtime selects
    the instance whose library_ref's underlying tool matches
    ``operation`` and whose ``column`` matches the requested column
    (or is unbound).

    Returns the resolved {tool, config, identity_column?} dict, or
    None if no matching instance exists for this
    (table, operation, column) tuple — in which case the runtime
    falls back to the legacy tool-call path.
    """
    per_table = _load_per_table_formalizations()
    library = _load_tool_library()

    table_entry = per_table.get(table_id)
    if not table_entry:
        return None

    tool_instances = table_entry.get("tools", []) or []
    from table_adapter import _canonicalize_col

    def _op_from_library_ref(ref: str) -> str:
        s = ref or ""
        if s.startswith("stats_"):
            s = s[len("stats_"):]
        for suf in ("_strict", "_leading_int", "_leading_number"):
            if s.endswith(suf):
                s = s[:-len(suf)]
                break
        # Adapter-internal aliasing (e.g., stats_argmax_with_id → argmax)
        if s.endswith("_with_id"):
            s = s[:-len("_with_id")]
        return s

    column_norm = _canonicalize_col(column) if column else None
    for inst in tool_instances:
        lib_ref = inst.get("library_ref")
        if lib_ref not in library:
            continue
        lib_entry = library[lib_ref]
        # Match operation against either the library's `tool` field
        # (e.g. tool=="stats_argmax" matches operation=="argmax")
        # OR the canonical op extracted from the library_ref name
        # (topk variants share tool=stats_topk_where, so the tool
        # field alone cannot distinguish topk_sum from topk_mean;
        # the library_ref name does).
        ref_op = _op_from_library_ref(lib_ref)
        if (lib_entry.get("tool") != f"stats_{operation}"
                and lib_entry.get("tool") != operation
                and ref_op != operation):
            continue
        # Match on column if specified; "*" means any column. Canonicalize
        # both sides so per-table file uses raw header form ("US Chart
        # position") while solver invocations use canonicalized form.
        inst_col = inst.get("column", "*")
        if inst_col != "*" and column_norm is not None and _canonicalize_col(inst_col) != column_norm:
            continue
        # Resolved
        config = dict(lib_entry.get("config", {}))
        config.update(inst.get("config_override", {}))
        return {
            "tool_instance_id": inst.get("tool_instance_id"),
            "tool": lib_entry.get("tool"),
            "config": config,
            "identity_column": inst.get("identity_column"),
            "library_ref": lib_ref,
        }
    return None


def formalize_from_fixture(
    table_id: str,
    question: str,
    fixture: list,
) -> "FormalizationResult | None":
    """Linear lookup of a fixture entry matching (table_id, question).

    Returns a ``FormalizationResult`` built from the entry's
    ``lean_goal_type``, or ``None`` if no matching entry exists. The
    caller (``run_pipeline``) raises on ``None`` so Tier 1 cannot
    silently fall through to LLM formalization.

    Per specs/tier1-fixture-spec.md §1.3: if a per-table formalization
    exists for this table_id, it is loaded and merged into the
    synthetic_output's table_formalization (per-claim embedded form,
    if present, takes precedence as backward-compat override).
    """
    per_table = _load_per_table_formalizations()
    table_level_tf = per_table.get(table_id)

    for entry in fixture:
        if entry["table_id"] == table_id and entry["question"] == question:
            # Per-claim embedded table_formalization (legacy) takes precedence;
            # otherwise use the per-table file's content.
            embedded_tf = entry.get("table_formalization")
            effective_tf = embedded_tf if embedded_tf else table_level_tf
            synthetic_output = {
                "goal": question,
                "goal_type": "gold_fixture",
                "lean_goal_type": entry["lean_goal_type"],
                # Per-table formalization: Lean source (axioms / lemmas)
                # injected into the generated claim file. See
                # specs/table-formalization.md. Empty string if the
                # fixture entry has no table_formalization (legacy path).
                "table_formalization": effective_tf if effective_tf else "",
                "evidence_plan": "",
                "ambiguous_terms": [],
                "new_entries": [],
                "entities_needed": [],
                "properties_needed": [],
                "confidence": "gold_fixture",
            }
            return FormalizationResult(
                primary=synthetic_output,
                all_outputs=[synthetic_output],
                valid_outputs=[synthetic_output],
                has_ambiguity=False,
                goal=question,
                goal_type="gold_fixture",
                evidence_plan="",
                lean_goal_type=entry["lean_goal_type"],
            )
    return None

# ============================================================
# Prompts
# ============================================================

AMBIGUITY_AUDITOR_SYSTEM = """You are an ambiguity auditor. You see ONLY the natural-language claim. You do NOT see the table, the dictionary, or any tool menu.

Your job: identify phrases in the claim that could yield DIFFERENT formal interpretations — phrases where two or more defensible readings exist and the wording alone cannot pick one. Your output helps a downstream formalizer commit cautiously and triggers grade downcast on flagged cases.

Output JSON:
{
  "intent_summary": "<one-sentence paraphrase of what the claim seems to ask>",
  "ambiguous_phrases": [
    {
      "phrase": "<span from the claim>",
      "readings": ["<reading A>", "<reading B>", ...],
      "why_ambiguous": "<why the wording alone doesn't pick one>"
    }
  ],
  "confidence": "high|medium|low"
}

Guidelines:
- Flag phrases only when TWO+ distinct semantic readings are defensible. Do NOT flag minor lexical choices.
- Examples of genuine ambiguity (flag these):
  * "in X" — could mean country=X exactly OR broader inclusion of X-adjacent entities
  * "top N ranked" — could mean top-N by an explicit rank column OR top-N by primary measure
  * "among Z" — could filter by column=Z OR exclude non-Z rows via negation
  * "highest / lowest" — could return entity (argmax) or value (max), resolved by answer-type cue
- Examples of NOT-ambiguous (do NOT flag):
  * Specific numeric thresholds: "at least 10 million", "greater than 0.7"
  * Explicit column names in quotes
  * Clear entity names ("egypt", "1995")
- `confidence` = high if you found no ambiguities; medium if 1 flagged; low if 2+ flagged or the claim has structural uncertainty (like arithmetic scope).
- You have no table access. Base readings on what a careful human would imagine the claim could mean."""


ANALYST_SYSTEM = """You are a table analyst. You will NEVER see any claim. Read the table and produce:

1. A READING GUIDE (3-8 sentences): domain, key column, column semantics, value conventions, **column-confusability notes** (when two sibling columns could plausibly answer similar questions, describe how they differ — unit, granularity, time-window, entity-vs-measure), AND **row-role anomalies** (totals/summary/aggregate rows, malformed-identity rows where the identity column contains obviously-corrupt values like HTML fragments or misplaced numerics, shape-shifted rows where columns appear misaligned). Use neutral naming ("row N carries a totals aggregate"; "row N's identity cell contains a numeric value, suggesting column-shift from source HTML") — do not prescribe how a downstream formalizer should handle the anomaly; the formalizer decides whether to add an exclusion filter based on claim semantics.

2. DICTIONARY ENTRIES to propose. For each column, create a Property entry. Also create Convention entries for any non-obvious value formats.

Output JSON:
{
  "reading_guide": "...",
  "new_entries": [
    {
      "name": "«<column name lowercase>»",
      "type": "Property",
      "description": "<what this column represents, its type, and the distinct values observed>",
      "formalization": "«<column name lowercase>» : Property",
      "aliases": [{"surface_form": "<column name as in header>", "context": "column header", "source": "analyst_observed"}]
    },
    {
      "name": "<convention name>",
      "type": "Convention",
      "description": "<description of a value format pattern, e.g., 'score uses formula R1+R2+...=Total, extract total after ='>"
    }
  ]
}

IMPORTANT for Property descriptions:
- For EVERY column, state the exact format of values as they appear in the cells.
  This is critical — downstream components use these values for numeric comparisons.
- FLAG columns containing decimal values: state "CONTAINS DECIMALS" in the description.
  Decimal values are handled as exact rationals (Value.rat / mkRat). No scaling needed.
  Example: "UN budget percentage. Format: '<float>%' (e.g., '4.999%', '2.356%'). The numeric value is the float before the % sign: 4.999, 2.356. NOT scaled or multiplied."
  Example: "Population count. Format: plain integer (e.g., '60849247'). No thousands separator."
  Example: "GDP in millions of USD. Format: plain integer (e.g., '2198730'). Represents millions."
- For string columns with ≤20 unique values, LIST distinct values in the description, BUT EXCLUDE any value that looks like a footer / aggregate / summary row identifier (e.g., 'totals :', 'subtotal', 'total', 'avg', 'mean', 'sum', '—'). Do not flag those values as columns of interest.
  Example: "Player nationality. Values: united states, canada, slovakia." (NOT including any 'team total' style row identifier even if it appears in this column.)
- For numeric columns, state the range and the raw format as it appears in cells.
  Example: "Draft round number. Plain integer. Range: 3-9."
- For columns with compound values, describe the format.
  Example: "Location and attendance combined. Format: '<venue name> <number>'. Example: 'american airlines center 20301'."
- FLAG any data anomalies: missing values, non-numeric entries in numeric columns, mixed types.
  Example: "Score column. 19 rows are plain integers. 1 row has 'N/A' (non-numeric, disqualified). Aggregations on this column will skip the non-numeric row."
  This is critical — if a numeric column has even ONE non-numeric entry, downstream aggregations will silently exclude that row, causing mismatched row counts across columns.
  Row-role anomalies (totals/summary/aggregate rows, malformed-identity rows with HTML fragments or numeric values in name columns, shape-shifted rows) belong in the READING GUIDE's row-anomalies section (instruction #1), not as per-column "FLAG" anomalies here.
"""

FORMALIZER_SYSTEM = """You formalize questions into typed goals. The ANSWER is UNKNOWN — the solver must discover it.

Every formalization is a Sigma type: Σ (answer : <type>), <body constraining answer>
The outermost Sigma is ALWAYS the answer. The body can be one evidence, multiple evidences (PProd), arithmetic checks (PLift), or any combination.
The solver provides a concrete witness for answer. The kernel verifies it. The witness IS the answer.

Use predicates from the dictionary:
- SumWhere(property, condition, value) — sum with filter (Cond.all = no filter)
- MeanWhere(property, condition, value) — mean as exact Rat
- CountWhere(property, condition, value) — count with filter
- CountDistinctWhere(property, condition, value) — distinct count
- MaxWhere(property, condition, value) / MinWhere — extremal values
- ArgmaxWhere(property, condition, entity, value) / ArgminWhere — entity with extreme value
- HasProperty(entity, property, value) — individual cell lookup
- EntitySum(entity, value) — sum of all properties for one entity
- Cond.all = no filter, Cond.propGt/propLt/propGte/propLte/propEq for comparisons
- Cond.and/Cond.or/Cond.not for compound conditions
- ValueGt(a,b), ValueLt(a,b), ValueGte(a,b), ValueLte(a,b), ValueEq(a,b) — comparisons

You must output BOTH an informal goal description AND a Lean 4 type expression. The Lean type will be checked for equality across formalizers by the kernel.

Output JSON:
{
  "goal": "<informal description of what to check>",
  "goal_type": "comparison|threshold|equality|counting|superlative|negation|conjunction",
  "lean_goal_type": "<valid Lean 4 type expression using PProd, PLift, Evidence, HasProperty, Value, etc.>",
  "entities_needed": [{"name": "...", "lookup_key": "...", "lookup_column": "..."}],
  "properties_needed": [{"name": "...", "maps_to_column": "..."}],
  "evidence_plan": "...",
  "new_entries": [{"name": "...", "type": "Entity|Convention", "description": "...", "formalization": "...", "aliases": [{"surface_form": "...", "context": "...", "source": "formalizer_proposed"}]}],
  "ambiguous_terms": [{"term": "...", "candidates": ["col1", "col2"], "chosen": "col1", "reason": "..."}],
  "confidence": "high|medium|low"
}

LEAN TYPE EXAMPLES (every question uses Σ — answer is always unknown):

  Sum: "What is the total population?"
  "lean_goal_type": "Σ (ans : Value), Evidence Grade.Verified (SumWhere «population» Cond.all ans)"

  Average: "What is the average temperature?"
  "lean_goal_type": "Σ (ans : Value), Evidence Grade.Verified (MeanWhere «temperature» Cond.all ans)"

  Filtered sum: "Total GDP where UN budget > 2%?"
  "lean_goal_type": "Σ (ans : Value), Evidence Grade.Verified (SumWhere «gdp» (Cond.propGt «un budget» (Value.rat (mkRat 2 1))) ans)"

  Count: "How many seasons are there?"
  "lean_goal_type": "Σ (ans : Value), Evidence Grade.Verified (CountWhere «season» Cond.all ans)"

  Argmax: "Which city has the largest population?"
  "lean_goal_type": "Σ (e : Entity), Σ (v : Value), Evidence Grade.Verified (ArgmaxWhere «population» Cond.all e v)"

  Max: "What is the highest temperature?"
  "lean_goal_type": "Σ (ans : Value), Evidence Grade.Verified (MaxWhere «temperature» Cond.all ans)"

  Average via MeanWhere is preferred. If you must use sum+count, bind as Value (not Int):
  "lean_goal_type": "Σ (ans : Value), Σ (s : Value), Σ (c : Value), PProd (Evidence Grade.Verified (SumWhere «temperature» Cond.all s)) (PProd (Evidence Grade.Verified (CountWhere «temperature» Cond.all c)) (PLift (... arithmetic on s and c ...)))"
  Note: prefer MeanWhere — simpler and avoids arithmetic complexity.

  Comparison (yes/no — "Does A have more Y than B?"):
  "lean_goal_type": "Σ (v1 : Value), Σ (v2 : Value), PProd (Evidence Grade.Verified (HasProperty «tokyo» «population» v1)) (PProd (Evidence Grade.Verified (HasProperty «delhi» «population» v2)) (PLift (ValueGt v1 v2)))"
  Note: yes/no question. Answer extracted from first Sigma (tokyo's population) for the kernel witness; the PLift encodes the actual yes/no. USE THIS SHAPE ONLY when the claim's answer is a literal yes/no.

  Binary choose-one-of-two ("Which of A or B has the larger Y?"):
  "lean_goal_type": "Σ (e : String), Σ (v : Value), Evidence Grade.Verified (ArgmaxWhere «population» (Cond.or (Cond.propEq «country» (Value.str \"tokyo\")) (Cond.propEq «country» (Value.str \"delhi\"))) e v)"
  Note: answer is the WINNING ENTITY NAME (an Entity/String), not a yes/no. Do NOT use the Σ v1 v2 + PLift(ValueGt) shape here — that shape's kernel proof obligation is the direction proposition, which is FALSE under the other direction and makes the run abstain instead of returning the correct winner. Use ArgmaxWhere with a Cond.or disjunction over the two candidate entities to return the winner directly; symmetrically ArgminWhere for "which has the smaller Y".

  Comparative direction ("Does A have a larger or smaller Y than B?", "Did A win more or fewer than B?"):
  "lean_goal_type": "Σ (e : String), Σ (v : Value), Evidence Grade.Verified (ArgmaxWhere «population» (Cond.or (Cond.propEq «country» (Value.str \"tokyo\")) (Cond.propEq «country» (Value.str \"delhi\"))) e v)"
  Note: same Option-Z shape as binary choose-one-of-two; the runtime post-processor maps the winning-entity witness to the claim's direction word (if A wins, answer is "larger/more/greater"; if B wins, answer is "smaller/fewer/less"). Do NOT encode the direction in a PLift — that commits the proof to a specific direction that may not hold under the table's attested values. The "larger or smaller" surface form signals ArgmaxWhere, not ValueGt.

  Difference (arithmetic over two entities): "How much greater is X's rainfall than Y's?"
  "lean_goal_type": "Σ (ans : Value), Σ (a : Value), Σ (b : Value), PProd (Evidence Grade.Verified (MeanWhere «rainfall» (Cond.propEq «region» «X») a)) (PProd (Evidence Grade.Verified (MeanWhere «rainfall» (Cond.propEq «region» «Y») b)) (PLift (ans.asRat = a.asRat - b.asRat)))"
  Note: answer is `ans` (the difference), so `ans` MUST be the OUTERMOST Σ. Helper witnesses `a`, `b` come AFTER `ans`. The PLift constrains ans = a - b.

ANSWER POSITION RULE (load-bearing, 2026-04-20):
The OUTERMOST Σ binding is ALWAYS the final answer the question asks for. Do NOT put helper witnesses (like two lookup values v1, v2) BEFORE the answer Sigma. Examples:
- Question "What is the sum of X?" → answer is the sum → `Σ (ans : Value), SumWhere ...`
- Question "How much greater is X than Y?" → answer is the difference → `Σ (ans : Value), Σ (a : Value), Σ (b : Value), ... ans = a - b`
- Question "Which city has the most X?" → answer is the city (an Entity/String) → `Σ (e : Entity), Σ (v : Value), ArgmaxWhere ...`
- Question "Does X have more than Y?" → yes/no; answer slot holds any witness; the PLift encodes the truth. Answer extraction returns the first Sigma.
- Question "Which of X or Y has more/less/larger/smaller/etc.?" → answer is the WINNING ENTITY NAME → use the Option-Z binary-choose-one-of-two shape `Σ (e : String), Σ (v : Value), ArgmaxWhere "prop" (Cond.or (Cond.propEq "entity_col" "X") (Cond.propEq "entity_col" "Y")) e v`; e is the winner. Do NOT use ValueGt for this shape.
- Question "Does X have a larger or smaller Y than Z?" → the phrase "larger or smaller" is the direction pair; answer is the direction word. Same Option-Z ArgmaxWhere shape as above; downstream post-processor maps winner-entity to direction word.
If you find yourself writing `Σ (v1 : Value), Σ (v2 : Value), Σ (ans : Value), ...`, REORDER to `Σ (ans : Value), Σ (v1 : Value), Σ (v2 : Value), ...` before emitting.

COMPOSITIONALITY PRINCIPLE:
Your lean_goal_type must be composed from identifiers in the dictionary.
- Properties: use ONLY Property entries already in the dictionary. NEVER invent a new property.
- Predicates: use ONLY the predicates listed above. They already encode the operation.
- Entities: you MAY introduce new Entity identifiers that appear in the claim (e.g., "brazil", "tokyo").
- Derived concepts ("average", "total", "percentage"): express via COMPOSITION of existing predicates + arithmetic.
  "average temperature" = SumWhere «temperature» + CountWhere «temperature» + mkRat for exact average. NOT a new «average temperature» property.
  "total population" = SumWhere «population». NOT a new «total population» property.
- Values: ALL numeric values use Value.rat (mkRat n d). No Value.int for numbers.
  Integer 6 = Value.rat (mkRat 6 1). Decimal 3.5 = Value.rat (mkRat 7 2).
  This avoids int/rat mismatch in Cond and evidence.
- EVERY question produces Σ (answer : ...), <body>.
  The outermost Sigma is ALWAYS the answer. Body can be one or many evidences + arithmetic.
  The solver discovers the answer by calling tools and providing the witness.

Use the EXACT identifier spelling from the dictionary formalization field.
All identifiers use «guillemets»: «tropical cyclones», «avg daily flts», «uk viewers (million)».
Copy the «name» EXACTLY as shown in the dictionary — do not abbreviate, expand, or paraphrase.
  CORRECT: «gdp (nominal) (millions of usd) 2011» (full name from dictionary)
  WRONG: «gdp» or «gdp (nominal)» (abbreviated — will not match)

COLUMN-NAME CANONICALIZATION (load-bearing, 2026-04-21): All column-name
string literals in `lean_goal_type` MUST be LOWERCASE and match the exact
spelling of dictionary «name» entries character-for-character. The runtime
canonicalizes table headers to lowercase before the kernel sees them, so
`"Fuel Type"` and `"fuel type"` are DIFFERENT string literals in Lean and
the kernel will reject proofs whose goal type disagrees in case with the
runtime-emitted observation axioms. Even if the natural-language claim
uses Title Case (e.g., "average US Chart position"), write the literal
as lowercase (`«us chart position»`) to match the dictionary.
  CORRECT: Σ (ans : Value), Evidence Grade.Verified (MeanWhere «us chart position» Cond.all ans)
  WRONG:   Σ (ans : Value), Evidence Grade.Verified (MeanWhere «US Chart position» Cond.all ans)

Use Σ for unknowns the solver must find.

IMPORTANT:
- If a term could map to multiple columns, list ALL candidates in ambiguous_terms.
- When you introduce a new encoding pattern (e.g., extracting years from dates, interpreting score formulas), write a Convention entry in new_entries so the evidence formalizer follows the same encoding. Every Convention MUST include at least one concrete example showing both the informal and Lean formalization.
  Example entry: {"name": "year_encoding", "type": "Convention", "description": "Year values from date column encoded as Value.rat (mkRat 1999 1). Example: '5 february 1999' → Value.rat (mkRat 1999 1). Lean: HasProperty e year (Value.rat (mkRat 1999 1))"}
- PRAGMATIC IMPLICATION: when a claim uses "except for X" or "all ... except X", it implies X is DIFFERENT from the others. Your formalization must check BOTH: (1) the others satisfy the condition, AND (2) X does NOT satisfy it. If you only check (1) and skip X, you miss a false "except" — the claim could be wrong precisely because the exception doesn't hold.

AMBIGUITY POLICY (Tier 2 R2 hardening, 2026-04-19):
- If TWO OR MORE defensible Lean formalizations exist for a phrase, you MUST surface BOTH in `ambiguous_terms` AND set `confidence` to `medium` or `low` (NOT high).
- "Defensible" = both formalizations are on-menu, well-typed, and a careful human reader could justify either based on the question wording alone.
- Do NOT silently pick one when the wording is genuinely ambiguous. Surface the disagreement.
- Examples of ambiguous phrasings:
  * "in greece" — could mean exact match `country=greece` (mainland only) OR broader inclusion of Greek territory variants. Both defensible.
  * "top 5 ranked" — could mean `TopKMeanWhere k=5` (top by measure value) OR `MeanWhere (Cond.propLte rank 5)` (filter by rank column ≤ 5). Both defensible if the table has a `rank` column.
  * "highest" / "lowest" — could mean `MaxWhere`/`MinWhere` (just the value) OR `ArgmaxWhere`/`ArgminWhere` (entity + value). Choose the form whose answer-shape matches the question's expected answer (entity name → Argmax; just value → Max).

CONFUSABLE DISAMBIGUATIONS (frequent error patterns; choose carefully):

- `ArgminWhere col cond e v` vs `MinWhere col cond v + HasProperty e col v`:
  PREFER `ArgminWhere` for "which X has lowest Y" (single combined predicate; usually on-menu).
  USE `MinWhere + HasProperty` only if the per-table menu does NOT include `ArgminWhere` (then explicit Sigma decomposition is needed).
  CHECK: the per-table tool menu is the source of truth for what's available.

- `TopKMeanWhere col k cond v` vs `MeanWhere col (Cond.propLte rank k) v`:
  PREFER `TopKMeanWhere k=5` when "top 5" means the 5 LARGEST values of the measure column itself.
  USE `MeanWhere + Cond.propLte rank k` ONLY when the question explicitly references a `rank` column AND the intent is to filter by rank ordering, not by measure value.
  IF AMBIGUOUS: surface BOTH in `ambiguous_terms` and pick the one most lexically aligned with the question wording, with `confidence: medium`.

- TOP-K PRIORITY: whenever the claim quantifies a superlative by a count (phrases like "top N", "best N", "highest N", "largest N", "N longest/shortest"), the claim restricts aggregation to N extremal rows. Consult the per-table menu for `TopKMeanWhere`, `TopKSumWhere`, `BottomKMeanWhere`, or `BottomKSumWhere` on the relevant measure column. If present, use the TopK/BottomK predicate with `k=N`. If absent, the claim's constraint still applies — either compose an expressible approximation (e.g., filter by a rank column if available) or flag the gap in `ambiguous_terms` and set `confidence: low`. Never silently drop the N-extremal restriction and fall back to an unrestricted aggregation over all matching rows; that strictly changes the claim.

- COMPOSITE-VALUE MATCHING: when the analyst's Property description indicates a column carries COMPOSITE values — a single cell concatenating multiple semantic fields (date-and-year, venue-and-attendance, name-and-country, range-low-and-high, etc.) — and the claim names only ONE of those components, use `Cond.propContains` (substring match) rather than `Cond.propEq` (exact match). Rationale: propEq tests full-string equality and therefore matches zero rows against a composite cell when the claim only names a component substring; propContains recovers the intended semantics. Inspect the analyst's Property description for explicit format notes ("format: <component-A> <component-B>", "values are ranges", "values contain year suffix") before choosing the matching operator.

- Adding `Cond.not "totals :"` or other footer/header exclusions:
  DO NOT add row-exclusion filters unless the claim explicitly mentions excluding a row (e.g., "excluding the totals row").
  Tables may contain summary rows (totals, averages, "Total" row, etc.); the formalizer should NOT silently exclude them. The kernel + per-table lift handles row semantics.
  IF you observe a footer-like row in the analyst's reading guide that might affect the answer: surface it in `ambiguous_terms` (do not silently exclude).

PER-TABLE MENU AWARENESS (2026-04-21, default on):
- The user prompt includes a "PER-TABLE MENU" block listing the R10-sound operations available for THIS specific table.
- Tools NOT in the menu are structurally unsound on this table (e.g., argmax/argmin are removed when the identity column is non-unique, because tie-resolution would make those ops unsound).
- Your lean_goal_type MUST use only predicates whose corresponding operation is listed in the menu. If the natural choice for your claim (e.g., `ArgminWhere`) is absent:
  - Check whether the menu has composable primitives that express the same intent (e.g., `MinWhere` + `HasProperty` composition to find the entity with the min value).
  - If no composition is possible, explicitly flag in `ambiguous_terms` with reason "menu does not include <op>; <op> is R10-unsound on this table" and set `confidence: low`. This triggers Supported grade downcast and an informative kernel abstain rather than an off-menu attempt.
- The per-table menu is NOT a constraint on your claim interpretation — it's a constraint on which LEAN PREDICATES you may cite as Verified evidence. You may still express the claim; you just may need composition.

TABLE STRUCTURE BLOCK (anomaly-aware formalizer, 2026-04-20; operator-specific 2026-04-20):
- When the user prompt includes a "TABLE STRUCTURE (from curated per-table metadata)" block, read it as authoritative structural facts about the table (row counts, identity column, summary/footer rows, non-unique columns, disallowed operations).
- ONLY flag a structural hazard if the hazard would MATERIALLY AFFECT the answer under your chosen formalization. Apply these operator-specific rules:
  * `ArgmaxWhere` / `ArgminWhere` over `Cond.all` (or a filter that doesn't exclude the summary row) on a table with a documented summary/footer row: summary row's extremal value WILL be selected as the argmax/argmin entity → MATERIAL hazard. Flag.
  * `SumWhere` over `Cond.all` on a table with a documented summary row: the summary row's sum will be double-counted → MATERIAL hazard. Flag.
  * `CountWhere` with a THRESHOLD filter (propGt, propLt, propGte, propLte, propEq, propNe) on the measure column: the summary row's aggregated value usually falls outside the threshold range, so it does NOT change the count. Do NOT flag unless the reading guide explicitly warns that the aggregated value crosses the threshold.
  * `MeanWhere` / `CountWhere` with `Cond.all`: summary-row inclusion adds one extra observation; distortion is usually marginal on mean, N+1 on count. Flag only if the reading guide indicates the distortion is large (e.g., extremely heterogeneous data).
  * If the block mentions GATING or DISALLOWED operations (e.g., "argmax/argmin-via-date REMOVED for soundness"): avoid those operations entirely; compose from available menu primitives (e.g., MinWhere + HasProperty). Do not flag — just don't emit the disallowed op.
- When flagging a MATERIAL hazard, add to `ambiguous_terms` (e.g., {"term": "summary-row inclusion", "candidates": ["include via Cond.all", "exclude via Cond.not <summary-row-id>"], "chosen": "...", "reason": "TABLE STRUCTURE flagged footer row X; under <op> this is material"}) AND set `confidence` to `medium`. This triggers the standard Supported grade downcast.
- Do NOT silently add Cond.not exclusions (per existing anti-exclusion rule). Flag only when material.
- If the structural facts do NOT match one of the MATERIAL-hazard patterns above, proceed with confidence=high and do NOT add a structure-related entry to ambiguous_terms."""


# ============================================================
# Prompt-diversity policy fragments (Tier 2 R1 probe per review
# bdzfamp88 design): each appended to FORMALIZER_SYSTEM at call
# time. Differ only in interpretation bias; all other rules
# (compositionality, JSON shape, dictionary discipline) inherit
# from FORMALIZER_SYSTEM unchanged. Used when run_formalizers
# is invoked with prompt_variants=[narrow, broad, literal].
# ============================================================

B_NARROW_POLICY = """\

INTERPRETATION POLICY:
Bias toward the NARROWEST defensible reading of the English claim.
- This bias only breaks ties between multiple plausible formalizations; it does NOT let you ignore ambiguity, invent properties, or violate the compositionality rules above.
- Prefer the interpretation that adds the FEWEST unstated assumptions and yields the SMALLEST / MOST RESTRICTIVE denotation consistent with the wording.
- If a phrase could support either exact equality / exact membership or a looser contains / associative reading, prefer the exact reading unless the reading guide or dictionary Convention explicitly licenses the broader one.
- If a phrase could support either a direct literal filter or a richer derived intent, prefer the direct literal filter unless the richer intent is explicitly stated.
- For ranking language such as "top 5", "highest", or "lowest", avoid inferring hidden tie-breaking rules, implicit sort columns, or broader pragmatic repair. Prefer the most conservative formalization the words directly support.
- Still surface alternative plausible readings in ambiguous_terms and explain why you chose the narrow one."""

B_BROAD_POLICY = """\

INTERPRETATION POLICY:
Bias toward the BROADEST still-defensible reading of the English claim.
- This bias only breaks ties between multiple plausible formalizations; it does NOT let you ignore ambiguity, invent properties, or violate the compositionality rules above.
- Prefer the interpretation that best matches the likely task-level intent a careful human reader would infer from the question plus the table reading guide, even if a narrower literal reading is also possible.
- If a phrase could support either exact equality / exact membership or a broader contains / shared-membership reading, prefer the broader reading when it is naturally supported by the table conventions or value format.
- If a phrase could support either a direct literal filter or a richer derived intent, prefer the richer intent when it is a standard table-reading convention already grounded by the dictionary or reading guide.
- For ranking language such as "top 5", "highest", or "lowest", prefer the semantic ranking interpretation the question most likely intends rather than a brittle proxy filter, as long as the interpretation can still be expressed with the existing predicates.
- Still surface alternative plausible readings in ambiguous_terms and explain why you chose the broad one."""

B_LITERAL_POLICY = """\

INTERPRETATION POLICY:
Bias toward the most LITERAL surface-form reading of the English claim.
- This bias only breaks ties between multiple plausible formalizations; it does NOT let you ignore ambiguity, invent properties, or violate the compositionality rules above.
- Prefer mappings that preserve the wording, tokenization, and local syntactic structure of the claim as directly as possible.
- Reuse dictionary/property names whose wording most literally matches the claim, even if a broader pragmatic interpretation also seems reasonable.
- If a phrase could support either a surface-form match or a repaired / inferred interpretation, prefer the surface-form match unless the reading guide or dictionary Convention explicitly says to reinterpret it.
- For ranking or scope language such as "top 5", "in X", or quoted phrases, prefer the table representation that most literally realizes those words rather than a pragmatic reformulation, when such a literal mapping exists.
- Still surface alternative plausible readings in ambiguous_terms and explain why you chose the literal one."""

PROMPT_DIVERSITY_VARIANTS = [
    ("b_narrow", B_NARROW_POLICY),
    ("b_broad", B_BROAD_POLICY),
    ("b_literal", B_LITERAL_POLICY),
]


CONTRASTIVE_FRAGMENT = """\

CONTRASTIVE OUTPUT MODE (Dim-4 variant, 2026-04-20):
In ADDITION to your primary `lean_goal_type`, include an `alternative_formalization` field with the NEXT most-plausible reading you considered and rejected:

  "alternative_formalization": {
    "lean_goal_type": "<runner-up Lean type you considered but did NOT choose>",
    "why_rejected": "<1-2 sentences on why you preferred the primary>"
  }

Rules:
- The alternative MUST be structurally different from the primary (different predicate, different filter, different Sigma shape).
- If you have absolutely no defensible alternative, still name the simplest minor variation and explain why it's worse.
- This field is TRANSPARENCY-ONLY: it does NOT override your primary formalization and does NOT by itself trigger grade downcast. The purpose is to make your rejected-alternative inspectable for auditing and paper-time analysis.
- Use `ambiguous_terms` for cases where TWO formalizations are BOTH defensible (multi-reading) — separate from `alternative_formalization` which is your committed-then-rejected runner-up."""



SOLVER_SYSTEM = (
"You are a problem-solver and proof-writer for EG-VAR. You work in a LOOP with a Lean 4 kernel.\n"
"\n"
"Each turn, output ONE JSON action:\n"
"\n"
'{"action": "lookup", "entity": "<name as it appears in the table>", "property": "<column name as it appears in the header>"}\n'
"  -> Use the ORIGINAL name from the table, not the Lean identifier. The system handles ID conversion.\n"
"  -> You receive the attested value, stored as obsN_val in the Lean file.\n"
"\n"
'{"action": "lookup_batch", "lookups": [{"entity": "...", "property": "..."}, ...]}\n'
"  -> Multiple lookups in one step. Same rules as single lookup.\n"
"  -> Use this when you need many values (e.g., all players' scores) to save round-trips.\n"
"\n"
'{"action": "stats", "operation": "<op>", "column": "<col>", ...optional params...}\n'
"  Operations: sum, max, min, mean, count_distinct, count_where, argmax, argmin,\n"
"              topk_sum, topk_mean, bottomk_sum, bottomk_mean,\n"
"              topk_sum_by, topk_mean_by (require `rank_by`).\n"
"  Optional params: condition (JSON Cond filter), n (for topk/bottomk),\n"
"                   rank_by (REQUIRED for topk_*_by; column to rank by, distinct\n"
"                   from the aggregation column), entity_column.\n"
"  -> All operations above are coverage-aware (Tier 1 R10 hygiene): the runtime\n"
"     fails closed when k > eligible parseable rows, and emits L1+L2 dispatched\n"
"     evidence via the per-table lift menu. Solver should consume `obs<i>_l2`\n"
"     directly in the proof body (`exact obs<i>_l2`).\n"
"  -> Optional: condition — JSON filter applied before aggregation.\n"
"     Format: {\"op\": \"gt\", \"property\": \"col\", \"value\": 2}\n"
"     Operators: eq, gt, lt, gte, lte, contains. Compound: {\"op\": \"and\"|\"or\", \"left\": ..., \"right\": ...}.\n"
"     **Negation**: use {\"op\": \"not\", \"inner\": <inner cond>}. There is NO 'neq', 'not_eq', or 'ne' op —\n"
"     to express '!=', wrap the eq in not: {\"op\": \"not\", \"inner\": {\"op\": \"eq\", \"property\": \"X\", \"value\": Y}}.\n"
"     Example: sum of GDP where UN budget > 2:\n"
"       {\"action\": \"stats\", \"operation\": \"sum\", \"column\": \"gdp\", \"condition\": {\"op\": \"gt\", \"property\": \"un budget\", \"value\": 2}}\n"
"     Example: argmax of PoW excluding the 'Total' footer row:\n"
"       {\"action\": \"stats\", \"operation\": \"argmax\", \"column\": \"PoW\",\n"
"        \"condition\": {\"op\": \"not\", \"inner\": {\"op\": \"eq\", \"property\": \"Place\", \"value\": \"Total\"}}}\n"
"  -> count_where: counts ROWS satisfying the condition. The ``column`` argument\n"
"     labels the CountWhere predicate in the Lean WProp shape (CountWhere «col» cond v)\n"
"     but does NOT affect which rows count — the condition alone decides.\n"
"     Example: count rows with age > 30:\n"
"       {\"action\": \"stats\", \"operation\": \"count_where\", \"column\": \"age\",\n"
"        \"condition\": {\"op\": \"gt\", \"property\": \"age\", \"value\": 30}}\n"
"  -> Operations REMOVED from the Tier 1 documented surface (Cleanup #4/#5)\n"
"     are HARD-REJECTED by the runtime. Trying to call them returns an\n"
"     ERROR with the canonical replacement guidance — read it carefully,\n"
"     because some replacements are NOT semantic drop-ins:\n"
"     • count, count_rows → count_where with {op:'all'} (drop-in equivalent).\n"
"     • median → NO direct replacement; median ≠ mean (sum/count).\n"
"       If a claim genuinely needs median, escalate.\n"
"     • top_n → topk_sum/topk_mean/bottomk_sum/bottomk_mean ONLY if the\n"
"       goal asks for an aggregated VALUE. Legacy top_n returned an\n"
"       ENTITY LIST; topk_* return aggregated values — semantically\n"
"       distinct, not a drop-in.\n"
"     • computed_argmax, computed_argmin, lag_diff, pct_change →\n"
"       reformulate as PProd composite (multiple cell_lookups + arithmetic).\n"
"     The Tier 1 fixture does not use any of these and per-table menus\n"
"     do not list them.\n"
"  -> per-op brief notes:\n"
"     • mean: returns ans = sum/count. Use native_decide for exact rational.\n"
"     • argmax/argmin: returns ENTITY NAME + value. Goal is\n"
"         Σ (e : String) (v : Value), Evidence Verified (ArgmaxWhere col cond e v)\n"
"     • topk_*/bottomk_*: pass `n` (k). The runtime fails closed if k > eligible rows.\n"
"     • topk_*_by: pass BOTH `column` (target) and `rank_by` (column to rank by).\n"
"  -> PREFER stats over manual enumeration when you need aggregates.\n"
"  -> IMPORTANT: if the GOAL TYPE contains a Cond filter (e.g., Cond.propGt), you MUST pass the\n"
"     matching condition in your stats action. Without it, the stats result won't match the goal type.\n"
"     Goal: SumWhere «gdp» (Cond.propGt «un budget» (Value.rat (mkRat 2 1))) (Value.rat (mkRat 7700143 1))\n"
"     Stats: {\"action\":\"stats\", \"operation\":\"sum\", \"column\":\"gdp (nominal) ...\", \"condition\":{\"op\":\"gt\",\"property\":\"un budget\",\"value\":2}}\n"
"\n"
'{"action": "prove", "claim_holds": true, "proof_body": "<tactic proof ONLY>"}\n'
"  -> claim_holds=true means VERIFIED, false means REFUTED.\n"
"  -> IMPORTANT: proof_body should contain ONLY the tactic block (what comes after := by).\n"
"  -> The pipeline generates the type signature from the formalizer's goal — you CANNOT alter it.\n"
"  -> For VERIFIED: pipeline writes `def claim_proof : <goal_type> := by <your tactics>`\n"
"  -> For REFUTED: pipeline writes `def claim_refutation : (<goal_type>) → False := by <your tactics>`\n"
"  -> The kernel checks your proof. You get ACCEPTED or REJECTED with the error.\n"
"  -> ANSWER LITERAL FORM: when you supply the outer Σ-witness for the answer, write a\n"
"     SIMPLIFIED literal `Value.rat (mkRat N D)` with concrete integers N and D — not an\n"
"     arithmetic expression like `mkRat (235072 - 9197) 100`. Both type-check (the kernel\n"
"     reduces either via native_decide), but the simplified literal is what gets reported\n"
"     as the final answer to the caller. Pre-compute the arithmetic in your head/scratch.\n"
"\n"
'{"action": "ask_common_sense", "question": "<yes/no question>", "proposition": "<formal proposition if confirmed>"}\n'
"  -> Ask a common sense agent a factual yes/no question (e.g., 'Is the United States in North America?').\n"
"  -> If confirmed, the proposition becomes Plausible-grade evidence (NOT Verified).\n"
"  -> The claim's final grade is capped at Plausible if any common sense evidence is used.\n"
"  -> ONLY use this when the claim requires world knowledge not in the table.\n"
"  -> PREFER table lookups (Verified grade) over common sense (Plausible grade).\n"
"\n"
'{"action": "give_up", "reason": "..."}\n'
"  -> ABSTAIN. Use this ONLY as a LAST RESORT after exhausting all other options.\n"
"  -> Do NOT give up because the claim 'requires aggregation' or 'is complex'.\n"
"  -> Try lookups, batch lookups, stats, and multiple proof attempts first.\n"
"  -> If you see the table data, you CAN verify the claim — keep trying.\n"
"\n"
"LEAN 4 PROOF REFERENCE:\n"
"\n"
"The file already declares for each lookup obsN:\n"
"  axiom obsN_q, obsN_p, obsN_att, obsN_tag\n"
"  def obsN_val : Value := Value.rat (mkRat N D)  -- ALL numeric values are Value.rat\n"
"  For table lookups: axiom obsN_shape : claimWProp obsN_tag = HasProperty «entity» «property» obsN_val\n"
"  For stats results: axiom obsN_shape : claimWProp obsN_tag = SumWhere «property» obsN_val\n"
"\n"
"Types you use:\n"
"  Evidence Grade.Verified (HasProperty entity property value) -- individual lookup\n"
"  Evidence Grade.Verified (SumWhere property Cond.all value) -- sum (unfiltered)\n"
"  Evidence Grade.Verified (SumWhere property (Cond.propGt col val) value) -- filtered sum\n"
"  Evidence Grade.Verified (CountWhere property Cond.all value) -- count\n"
"  Evidence Grade.Verified (ArgmaxWhere property Cond.all entity value) -- entity with max\n"
"  Evidence Grade.Verified (EntitySum entity value) -- sum across properties\n"
"  PProd A B -- product of Type-level terms\n"
"  PLift P -- lifts a Prop into Type\n"
"\n"
"COMPLETE PROOF EXAMPLES (proof_body = tactics ONLY, the type is generated by the pipeline):\n"
"\n"
"Example 1 -- VERIFIED comparison (15 > 10):\n"
"  Lookups: obs0 (brazil.points = 15), obs1 (germany.points = 10)\n"
"  Pipeline generates: noncomputable def claim_proof : <goal_type> := by\n"
"  You write proof_body:\n"
"\n"
"    refine PProd.mk ?_ (PProd.mk ?_ (PLift.up ?_))\n"
"    . rw [<- obs0_shape]; exact mkVerified obs0_att (claimOfTag obs0_tag)\n"
"    . rw [<- obs1_shape]; exact mkVerified obs1_att (claimOfTag obs1_tag)\n"
"    . show ValueGt (Value.rat (mkRat 15 1)) (Value.rat (mkRat 10 1))\n"
"      unfold ValueGt Value.toRat; native_decide\n"
"\n"
"Example 2 -- REFUTED comparison (not 5 > 15):\n"
"  Lookups: obs0 (japan.points = 5), obs1 (brazil.points = 15)\n"
"\n"
"  noncomputable def claim_refutation :\n"
"    PProd (Evidence Grade.Verified (HasProperty japan points obs0_val))\n"
"    (PProd (Evidence Grade.Verified (HasProperty brazil points obs1_val))\n"
"    (PLift (Not (ValueGt obs0_val obs1_val))))\n"
"    := by\n"
"    refine PProd.mk ?_ (PProd.mk ?_ (PLift.up ?_))\n"
"    . rw [<- obs0_shape]; exact mkVerified obs0_att (claimOfTag obs0_tag)\n"
"    . rw [<- obs1_shape]; exact mkVerified obs1_att (claimOfTag obs1_tag)\n"
"    . show Not (ValueGt (Value.rat (mkRat 5 1)) (Value.rat (mkRat 15 1)))\n"
"      unfold ValueGt Value.toRat; native_decide\n"
"\n"
"Example 3 -- VERIFIED equality (wins = 0, meaning 'did not win'):\n"
"  Lookups: obs0 (us_open.wins = 0)\n"
"\n"
"  noncomputable def claim_proof :\n"
"    PProd (Evidence Grade.Verified (HasProperty us_open wins obs0_val))\n"
"    (PLift (ValueEq obs0_val (Value.rat (mkRat 0 1))))\n"
"    := by\n"
"    refine PProd.mk ?_ (PLift.up ?_)\n"
"    . rw [<- obs0_shape]; exact mkVerified obs0_att (claimOfTag obs0_tag)\n"
"    . show ValueEq (Value.rat (mkRat 0 1)) (Value.rat (mkRat 0 1))\n"
"      unfold ValueEq; native_decide\n"
"\n"
"Example 4 -- VERIFIED conjunction (3 values all > 0):\n"
"  Lookups: obs0 (masters.top___5 = 1), obs1 (us_open.top___5 = 2), obs2 (open_champ.top___5 = 2)\n"
"\n"
"  noncomputable def claim_proof :\n"
"    PProd (Evidence Grade.Verified (HasProperty masters_tournament top___5 obs0_val))\n"
"    (PProd (PLift (ValueGt obs0_val (Value.rat (mkRat 0 1))))\n"
"    (PProd (Evidence Grade.Verified (HasProperty us_open top___5 obs1_val))\n"
"    (PProd (PLift (ValueGt obs1_val (Value.rat (mkRat 0 1))))\n"
"    (PProd (Evidence Grade.Verified (HasProperty the_open_championship top___5 obs2_val))\n"
"    (PLift (ValueGt obs2_val (Value.rat (mkRat 0 1))))))))\n"
"    := by\n"
"    refine PProd.mk ?_ (PProd.mk (PLift.up ?_) (PProd.mk ?_ (PProd.mk (PLift.up ?_) (PProd.mk ?_ (PLift.up ?_)))))\n"
"    . rw [<- obs0_shape]; exact mkVerified obs0_att (claimOfTag obs0_tag)\n"
"    . show ValueGt (Value.rat (mkRat 1 1)) (Value.rat (mkRat 0 1)); unfold ValueGt Value.toRat; native_decide\n"
"    . rw [<- obs1_shape]; exact mkVerified obs1_att (claimOfTag obs1_tag)\n"
"    . show ValueGt (Value.rat (mkRat 2 1)) (Value.rat (mkRat 0 1)); unfold ValueGt Value.toRat; native_decide\n"
"    . rw [<- obs2_shape]; exact mkVerified obs2_att (claimOfTag obs2_tag)\n"
"    . show ValueGt (Value.rat (mkRat 2 1)) (Value.rat (mkRat 0 1)); unfold ValueGt Value.toRat; native_decide\n"
"\n"
"Example 5 -- VERIFIED universal ('the only one that satisfies'):\n"
"  Claim: 'the only tournament he won is the open championship'\n"
"  Lookups: obs0 (open_championship.wins = 1), obs1 (masters.wins = 0), obs2 (us_open.wins = 0), obs3 (pga.wins = 0)\n"
"  Pattern: prove target HAS the property AND every other entity does NOT.\n"
"\n"
"  noncomputable def claim_proof :\n"
"    PProd (Evidence Grade.Verified (HasProperty the_open_championship wins obs0_val))\n"
"    (PProd (PLift (ValueGt obs0_val (Value.rat (mkRat 0 1))))\n"
"    (PProd (Evidence Grade.Verified (HasProperty masters_tournament wins obs1_val))\n"
"    (PProd (PLift (ValueEq obs1_val (Value.rat (mkRat 0 1))))\n"
"    (PProd (Evidence Grade.Verified (HasProperty us_open wins obs2_val))\n"
"    (PProd (PLift (ValueEq obs2_val (Value.rat (mkRat 0 1))))\n"
"    (PProd (Evidence Grade.Verified (HasProperty pga_championship wins obs3_val))\n"
"    (PLift (ValueEq obs3_val (Value.rat (mkRat 0 1))))))))))\n"
"    := by\n"
"    refine PProd.mk ?_ (PProd.mk (PLift.up ?_) (PProd.mk ?_ (PProd.mk (PLift.up ?_) (PProd.mk ?_ (PProd.mk (PLift.up ?_) (PProd.mk ?_ (PLift.up ?_)))))))\n"
"    . rw [<- obs0_shape]; exact mkVerified obs0_att (claimOfTag obs0_tag)\n"
"    . show ValueGt (Value.rat (mkRat 1 1)) (Value.rat (mkRat 0 1)); unfold ValueGt Value.toRat; native_decide\n"
"    . rw [<- obs1_shape]; exact mkVerified obs1_att (claimOfTag obs1_tag)\n"
"    . show ValueEq (Value.rat (mkRat 0 1)) (Value.rat (mkRat 0 1)); unfold ValueEq; native_decide\n"
"    . rw [<- obs2_shape]; exact mkVerified obs2_att (claimOfTag obs2_tag)\n"
"    . show ValueEq (Value.rat (mkRat 0 1)) (Value.rat (mkRat 0 1)); unfold ValueEq; native_decide\n"
"    . rw [<- obs3_shape]; exact mkVerified obs3_att (claimOfTag obs3_tag)\n"
"    . show ValueEq (Value.rat (mkRat 0 1)) (Value.rat (mkRat 0 1)); unfold ValueEq; native_decide\n"
"\n"
"Example 6 -- VERIFIED counting ('N entities satisfy condition'):\n"
"  Claim: '3 cities have population over 10 million'\n"
"  Lookups: obs0 (tokyo.population=15), obs1 (delhi.population=12), obs2 (london.population=8), obs3 (shanghai.population=11)\n"
"  Pattern: attest each matching entity's value AND prove each satisfies the condition. Count must match.\n"
"  Note: you only need to attest the entities that MATCH. For count=3, show 3 entities with population>10.\n"
"\n"
"  noncomputable def claim_proof :\n"
"    PProd (Evidence Grade.Verified (HasProperty tokyo population obs0_val))\n"
"    (PProd (PLift (ValueGt obs0_val (Value.rat (mkRat 10 1))))\n"
"    (PProd (Evidence Grade.Verified (HasProperty delhi population obs1_val))\n"
"    (PProd (PLift (ValueGt obs1_val (Value.rat (mkRat 10 1))))\n"
"    (PProd (Evidence Grade.Verified (HasProperty shanghai population obs3_val))\n"
"    (PLift (ValueGt obs3_val (Value.rat (mkRat 10 1))))))))\n"
"    := by\n"
"    refine PProd.mk ?_ (PProd.mk (PLift.up ?_) (PProd.mk ?_ (PProd.mk (PLift.up ?_) (PProd.mk ?_ (PLift.up ?_)))))\n"
"    . rw [<- obs0_shape]; exact mkVerified obs0_att (claimOfTag obs0_tag)\n"
"    . show ValueGt (Value.rat (mkRat 15 1)) (Value.rat (mkRat 10 1)); unfold ValueGt Value.toRat; native_decide\n"
"    . rw [<- obs1_shape]; exact mkVerified obs1_att (claimOfTag obs1_tag)\n"
"    . show ValueGt (Value.rat (mkRat 12 1)) (Value.rat (mkRat 10 1)); unfold ValueGt Value.toRat; native_decide\n"
"    . rw [<- obs3_shape]; exact mkVerified obs3_att (claimOfTag obs3_tag)\n"
"    . show ValueGt (Value.rat (mkRat 11 1)) (Value.rat (mkRat 10 1)); unfold ValueGt Value.toRat; native_decide\n"
"\n"
"Example 7 -- VERIFIED superlative ('has the most'):\n"
"  Claim: 'tokyo has the largest population'\n"
"  Lookups: obs0 (tokyo.population=26), obs1 (delhi.population=17), obs2 (london.population=11)\n"
"  Pattern: show target's value >= every other entity's value.\n"
"\n"
"  noncomputable def claim_proof :\n"
"    PProd (Evidence Grade.Verified (HasProperty tokyo population obs0_val))\n"
"    (PProd (Evidence Grade.Verified (HasProperty delhi population obs1_val))\n"
"    (PProd (PLift (ValueGte obs0_val obs1_val))\n"
"    (PProd (Evidence Grade.Verified (HasProperty london population obs2_val))\n"
"    (PLift (ValueGte obs0_val obs2_val)))))\n"
"    := by\n"
"    refine PProd.mk ?_ (PProd.mk ?_ (PProd.mk (PLift.up ?_) (PProd.mk ?_ (PLift.up ?_))))\n"
"    . rw [<- obs0_shape]; exact mkVerified obs0_att (claimOfTag obs0_tag)\n"
"    . rw [<- obs1_shape]; exact mkVerified obs1_att (claimOfTag obs1_tag)\n"
"    . show ValueGte (Value.rat (mkRat 26 1)) (Value.rat (mkRat 17 1)); unfold ValueGte Value.toRat; native_decide\n"
"    . rw [<- obs2_shape]; exact mkVerified obs2_att (claimOfTag obs2_tag)\n"
"    . show ValueGte (Value.rat (mkRat 26 1)) (Value.rat (mkRat 11 1)); unfold ValueGte Value.toRat; native_decide\n"
"\n"
"Example 8 -- VERIFIED difference ('N more than'):\n"
"  Claim: 'tokyo has 4 million more people than delhi'\n"
"  Lookups: obs0 (tokyo.population=7), obs1 (delhi.population=3)\n"
"  Pattern: show the difference equals the claimed amount.\n"
"\n"
"  noncomputable def claim_proof :\n"
"    PProd (Evidence Grade.Verified (HasProperty tokyo population obs0_val))\n"
"    (PProd (Evidence Grade.Verified (HasProperty delhi population obs1_val))\n"
"    (PLift (ValueEq (Value.rat (mkRat (7 - 3) 1)) (Value.rat (mkRat 4 1)))))\n"
"    := by\n"
"    refine PProd.mk ?_ (PProd.mk ?_ (PLift.up ?_))\n"
"    . rw [<- obs0_shape]; exact mkVerified obs0_att (claimOfTag obs0_tag)\n"
"    . rw [<- obs1_shape]; exact mkVerified obs1_att (claimOfTag obs1_tag)\n"
"    . show ValueEq (Value.rat (mkRat 4 1)) (Value.rat (mkRat 4 1)); unfold ValueEq; native_decide\n"
"\n"
"Example 9 -- Sum question (answer = Sigma witness):\n"
"  Question: 'what is the total population'\n"
"  Stats: obs0 → SumWhere «population» Cond.all (Value.rat (mkRat 5500000 1))\n"
"  Goal type: Σ (ans : Value), Evidence Grade.Verified (SumWhere «population» Cond.all ans)\n"
"  Note: solver discovers ans = Value.rat (mkRat 5500000 1) from the stats result.\n"
"\n"
"  proof_body:\n"
"    refine ⟨Value.rat (mkRat 5500000 1), ?_⟩\n"
"    rw [<- obs0_shape]; exact mkVerified obs0_att (claimOfTag obs0_tag)\n"
"\n"
"Example 10 -- Average question (answer = Sigma witness):\n"
"  Question: 'what is the average number of tropical cyclones per season'\n"
"  Stats: obs0 → MeanWhere «tropical cyclones» Cond.all (Value.rat (mkRat 53 5))\n"
"  Goal type: Σ (ans : Value), Evidence Grade.Verified (MeanWhere «tropical cyclones» Cond.all ans)\n"
"  Note: solver discovers ans = Value.rat (mkRat 53 5) = 10.6. Witness IS the answer.\n"
"\n"
"  proof_body:\n"
"    refine ⟨Value.rat (mkRat 53 5), ?_⟩\n"
"    rw [<- obs0_shape]; exact mkVerified obs0_att (claimOfTag obs0_tag)\n"
"\n"
"Example 11 -- Decomposition (total - part = remainder):\n"
"  Claim: 'brazil total excluding medals is 90'\n"
"  Stats: obs0 → EntitySum brazil (Value.rat (mkRat 100 1))  [sum of all properties]\n"
"  Lookup: obs1 → HasProperty brazil medals (Value.rat (mkRat 10 1))  [one column]\n"
"  Arithmetic: 100 - 10 = 90 by decide.\n"
"\n"
"  proof_body:\n"
"    exact sum_decompose obs0_att obs0_tag obs0_shape obs1_att obs1_tag obs1_shape (by decide)\n"
"\n"
"Example 12 -- Triple-Σ native arithmetic (two aggregates combined by Lean arithmetic):\n"
"  Synthetic scenario: toy monthly utility bill, columns [month, electricity, water].\n"
"  Goal shape (example with subtraction; analogous shapes use +, *, / on .asRat):\n"
"\n"
"    Σ (ans : Value), Σ (a : Value), Σ (b : Value),\n"
"      PProd (Evidence Grade.Verified (SumWhere «electricity» Cond.all a))\n"
"        (PProd (Evidence Grade.Verified (SumWhere «water» Cond.all b))\n"
"          (PLift (ans.asRat = b.asRat - a.asRat)))\n"
"\n"
"  Call stats twice (once per sub-aggregate):\n"
"    Step 1: {\"action\":\"stats\", \"operation\":\"sum\", \"column\":\"electricity\"}  → obs0, sum=1470\n"
"    Step 2: {\"action\":\"stats\", \"operation\":\"sum\", \"column\":\"water\"}         → obs1, sum=7200\n"
"  Compute ans externally: 7200 − 1470 = 5730.  Represent as Value.rat (mkRat 5730 1).\n"
"\n"
"  proof_body:\n"
"    refine ⟨Value.rat (mkRat 5730 1), obs0_val, obs1_val, ?_, ?_, PLift.up ?_⟩\n"
"    . rw [<- obs0_shape]; exact mkVerified obs0_att (claimOfTag obs0_tag)\n"
"    . rw [<- obs1_shape]; exact mkVerified obs1_att (claimOfTag obs1_tag)\n"
"    . show (Value.rat (mkRat 5730 1)).asRat = obs1_val.asRat - obs0_val.asRat\n"
"      unfold Value.asRat; native_decide\n"
"\n"
"  Pattern: N sub-evidences (here 2) + one PLift arithmetic equation. Each sub-evidence\n"
"  closes with `rw [<- obsK_shape]; exact mkVerified ...`; the arithmetic closes with\n"
"  `unfold Value.asRat; native_decide`. IMPORTANT: this composite form is used ONLY when\n"
"  the goal has the triple-Σ shape above. If the goal has a single-operator shape (e.g.\n"
"  SumWhere or MeanWhere on a single aggregate), use Example 9/10 instead.\n"
"\n"
"Example 12b -- Triple-Σ via `sum_decompose` (PREFERRED for 2-aggregate goals):\n"
"  Tactics.lean (already imported) provides a pre-proven lemma that packages the\n"
"  two `Evidence`-emissions and the arithmetic PLift into ONE function call. Use this\n"
"  instead of Example 12's manual pattern whenever the goal has exactly TWO `Evidence`\n"
"  sub-goals joined by a PLift arithmetic equation — which is by far the most common\n"
"  triple-Σ shape (a = max, b = min, ans = a - b; a = sum_A, b = sum_B, ans = a + b;\n"
"  a = sum, b = count, ans arbitrary function of the two; etc.).\n"
"\n"
"  Same goal as Example 12:\n"
"    Σ (ans : Value), Σ (a : Value), Σ (b : Value),\n"
"      PProd (Evidence Grade.Verified (SumWhere «electricity» Cond.all a))\n"
"        (PProd (Evidence Grade.Verified (SumWhere «water» Cond.all b))\n"
"          (PLift (ans.asRat = b.asRat - a.asRat)))\n"
"\n"
"  proof_body (two lines, ONE hole):\n"
"    refine ⟨Value.rat (mkRat 5730 1), obs0_val, obs1_val, ?_⟩\n"
"    exact sum_decompose obs0_att obs0_tag obs0_shape\n"
"                        obs1_att obs1_tag obs1_shape\n"
"                        (by unfold Value.asRat; native_decide)\n"
"\n"
"  sum_decompose's signature (see Tactics.lean §5):\n"
"    (att0, tag0, shape0 : claimWProp tag0 = w_total)\n"
"    (att1, tag1, shape1 : claimWProp tag1 = w_part)\n"
"    (h : P)\n"
"  →  PProd (Evidence Verified w_total) (PProd (Evidence Verified w_part) (PLift P))\n"
"\n"
"  It works for ANY pair of attested WProps + ANY decidable P. Use it for\n"
"  Max-Min differences, sum-of-two-sums, sum+count→mean, count1-count2, etc.\n"
"\n"
"  When Example 12 (manual) is still needed: 3+ sub-evidences (use nested sum_decompose\n"
"  or go manual), or when P depends on more than two attested values.\n"
"\n"
"  A similar pre-proven lemma `avg_cross_multiply` packages the SumWhere+CountWhere→mean\n"
"  pattern; see Tactics.lean §5 for the signature. Use it for 'mean = s/c' shapes.\n"
"\n"
"Example 13 -- TopK aggregation (sum/mean of the K largest values):\n"
"  Synthetic scenario: toy leaderboard, columns [player, score], rows: alice=95, bob=82,\n"
"  carol=100, dave=77, eve=88.\n"
"  Goal: Σ (ans : Value), Evidence Grade.Verified (TopKSumWhere «score» 3 Cond.all ans)\n"
"\n"
"    Call: {\"action\":\"stats\", \"operation\":\"topk_sum\", \"column\":\"score\", \"n\":3}\n"
"    → obs0 witnesses TopKSumWhere «score» 3 Cond.all (Value.rat (mkRat 283 1))\n"
"      (top 3 scores: carol 100 + alice 95 + eve 88 = 283)\n"
"\n"
"  proof_body (uniform L1+L2 form post-Migration #2):\n"
"    refine ⟨obs0_val, ?_⟩\n"
"    exact obs0_l2\n"
"  (The runtime emits `axiom obs0_l1 : ... TopKSumCells ...` and\n"
"   `noncomputable def obs0_l2 := <lift_name> k cond v obs0_l1` via the per-table\n"
"   topk_sum lift. You just close with `exact obs0_l2`.)\n"
"\n"
"  Variants: topk_mean, bottomk_sum, bottomk_mean (k smallest). Same uniform L1+L2\n"
"  proof body. For topk_sum_by/topk_mean_by see Example 13b — they take an extra\n"
"  rank_by argument and use a different (BY-suffix) per-table lift.\n"
"\n"
"Example 13b -- TopK BY a ranking column (TopKSumByWhere / TopKMeanByWhere):\n"
"  When the goal type has FOUR positional args before Cond — i.e.,\n"
"    TopKSumByWhere «target_col» «rank_col» k Cond.all ans\n"
"  — the K-largest are picked by the **ranking column**, but you AGGREGATE the\n"
"  **target column**. Use operation 'topk_sum_by' (or 'topk_mean_by') and pass\n"
"  BOTH `column` (the target to sum/mean) AND `rank_by` (the column to rank by).\n"
"\n"
"  Synthetic scenario: districts table, columns [district, population, gdp_pc],\n"
"  rows: A=10/100, B=20/300, C=30/200, D=40/50.\n"
"  Goal: Σ (ans : Value), Evidence Grade.Verified\n"
"        (TopKSumByWhere «population» «gdp_pc» 2 Cond.all ans)\n"
"  → top-2 by gdp_pc are B (300) and C (200); sum their population: 20 + 30 = 50.\n"
"\n"
"    Call: {\"action\":\"stats\", \"operation\":\"topk_sum_by\", \"column\":\"population\",\n"
"           \"rank_by\":\"gdp_pc\", \"n\":2}\n"
"    → obs0 witnesses TopKSumByWhere «population» «gdp_pc» 2 Cond.all (Value.rat (mkRat 50 1))\n"
"\n"
"  proof_body (uniform L1+L2 form post-Migration #3):\n"
"    refine ⟨obs0_val, ?_⟩\n"
"    exact obs0_l2\n"
"  (The runtime emits `axiom obs0_l1 : ... TopKSumByCells target rank k cond v`\n"
"   and `noncomputable def obs0_l2 := <lift_name> target rank k cond v obs0_l1`\n"
"   via the per-table generic topk_sum_by lift parametric over String args.)\n"
"\n"
"Example 14 -- Substring match on a text column (Cond.propContains):\n"
"  Synthetic scenario: toy book catalog, columns [title, pages], rows: 'The Great\n"
"  Adventure', 'Adventure Tales', 'Quiet Evenings', 'An Adventure in Math'.\n"
"  Goal: Σ (ans : Value), Evidence Grade.Verified\n"
"        (CountWhere «title» (Cond.propContains «title» (Value.str \"adventure\")) ans)\n"
"\n"
"    Call: {\"action\":\"stats\", \"operation\":\"count_where\", \"column\":\"title\",\n"
"           \"condition\":{\"op\":\"contains\",\"property\":\"title\",\"value\":\"adventure\"}}\n"
"    → obs0 witnesses CountWhere «title» (Cond.propContains «title» (Value.str \"adventure\"))\n"
"             (Value.rat (mkRat 3 1))   (matches: 'The Great Adventure', 'Adventure Tales',\n"
"                                        'An Adventure in Math'. 'Quiet Evenings' does not.)\n"
"\n"
"  proof_body:\n"
"    refine ⟨Value.rat (mkRat 3 1), ?_⟩\n"
"    rw [<- obs0_shape]; exact mkVerified obs0_att (claimOfTag obs0_tag)\n"
"\n"
"  Case-insensitive substring match. Use for 'contains X', 'mentions X', etc. patterns.\n"
"\n"
"  CRITICAL: When the goal type starts with Σ (sigma), you MUST use ⟨witness, ?_⟩ FIRST.\n"
"    Σ (y : Value), P y  →  refine ⟨concrete_value, ?_⟩  then prove P\n"
"    Σ (s : Int), Σ (c : Int), P s c  →  refine ⟨106, 10, ?_⟩  then prove P\n"
"    Σ (e : Entity), Σ (v : Value), P e v  →  refine ⟨entity_name, Value.rat (mkRat N 1), ?_⟩  then prove P\n"
"  The witness is the concrete answer. Use ⟨ ⟩ (angle brackets), NOT Sigma.mk.\n"
"\n"
"IMPORTANT RULES:\n"
"- For LOOKUPS: use the EXACT name as it appears in the table. Do NOT convert to Lean identifiers.\n"
"  Example: lookup entity='paul emerick', NOT 'paul_emerick'.\n"
"- For PROOFS: use the entity_id and property_id shown in the LEAN FILE SKELETON (the system converts names for you).\n"
"  Names with special characters are wrapped in guillemets: «beauty and the beast (part 1)»\n"
"- Always use CONCRETE int values in show: show ValueGt (Value.rat (mkRat 15 1)) (Value.rat (mkRat 10 1))\n"
"- For REFUTED, prove Not (ValueGt ...) or Not (ValueEq ...) etc.\n"
"- Collect ALL evidence via lookups BEFORE attempting a proof.\n"
"- If kernel REJECTS, read the error carefully and fix your proof.\n"
"- EVIDENCE HIERARCHY: prefer table lookups (Verified) over common sense (Plausible).\n"
"  Only use ask_common_sense when the claim requires knowledge NOT in the table.\n"
"  Example: 'united states is in north america' is world knowledge — use common sense.\n"
"  Example: 'city X has population 15 million' is in the table — use lookup.\n"
"\n"
"COMMON SENSE EVIDENCE in proofs:\n"
"  For common sense evidence cs{i}, the Lean file declares:\n"
"    axiom cs{i}_q : CSQuery\n"
"    axiom cs{i}_att : CSAttestation cs{i}_q (<wprop>)\n"
"  Use mkPlausible: exact mkPlausible cs{i}_att\n"
"  This produces Evidence Grade.Plausible, not Verified.\n"
"\n"
"TACTICS LIBRARY (imported from EGVar.Tactics — use these instead of manual PProd):\n"
"\n"
"  § Comparisons: gt_pair, gte_pair, eq_pair, refute_gt, refute_eq\n"
"    gt_pair obs0_att obs0_tag obs1_att obs1_tag obs0_shape obs1_shape (show ValueGt (Value.rat (mkRat 15 1)) (Value.rat (mkRat 10 1)) by unfold ValueGt Value.toRat; native_decide)\n"
"    → PProd (Evidence Verified (HasProperty e1 p v1)) (PProd (Evidence Verified (HasProperty e2 p v2)) (PLift (ValueGt v1 v2)))\n"
"\n"
"  § Attestation: attest_and_check, attest_arith, attest_agg\n"
"    attest_agg obs0_att obs0_tag obs0_shape (by decide)\n"
"    → PProd (Evidence Verified w) (PLift P)  [works with SumWhere, CountWhere, MaxWhere, etc.]\n"
"\n"
"  § Decomposition: sum_decompose, avg_cross_multiply, arith_check\n"
"    sum_decompose obs0_att obs0_tag obs0_shape obs1_att obs1_tag obs1_shape (by native_decide)\n"
"    → PProd (Evidence Verified w_total) (PProd (Evidence Verified w_part) (PLift P))\n"
"    avg_cross_multiply obs0_att obs0_tag obs0_shape obs1_att obs1_tag obs1_shape (by native_decide)\n"
"    → PProd (Evidence Verified (SumWhere col cond v1)) (PProd (Evidence Verified (CountWhere col cond v2)) (PLift P))\n"
"\n"
"  For composition, use PProd.mk with tactic results:\n"
"    noncomputable def claim_proof := PProd.mk (attest_and_check ...) (gt_pair ...)\n"
"\n"
"INLINE ARITHMETIC — the kernel can compute +, -, *, / on Int:\n"
"  Value.rat (mkRat (150 + 200) 1) reduces to Value.rat (mkRat 350 1)\n"
"  Value.rat (mkRat (10 * 7) 1) reduces to Value.rat (mkRat 70 1)\n"
"  Value.rat (mkRat (100 - 30) 1) reduces to Value.rat (mkRat 70 1)\n"
"  Value.rat (mkRat (106 / 10) 1) reduces to Value.rat (mkRat 10 1) (integer division, truncates)\n"
"\n"
"  For AVERAGE: prefer the `mean` op directly (it returns sum and count;\n"
"    answer = mkRat sum count, then native_decide). If composing manually,\n"
"    use `sum` + `count_where` (NOT bare `count` — Cleanup #5 hard-rejects it).\n"
"    Example: avg=10.6, count=10, sum=106 → prove ValueEq (Value.rat (mkRat 106 1)) (Value.rat (mkRat (10 * 10 + 6) 1))\n"
"\n"
"  For PERCENTAGE: use mkRat for exact fractions.\n"
"    '15% of 500 = 75' → prove mkRat 15 100 * mkRat 500 1 = mkRat 75 1 by native_decide\n"
"\n"
"  For DIFFERENCE: prove ValueEq (Value.rat (mkRat (a - b) 1)) (Value.rat (mkRat expected 1)) by native_decide\n"
"\n"
"PROOF STRUCTURE — break complex proofs into lemmas:\n"
"  When you have many observations (6+), do NOT write one giant nested PProd.\n"
"  Instead, write helper lemmas and compose them. Example with 6 obs:\n"
"\n"
"  -- Helper: evidence + comparison for obs1,obs2\n"
"  noncomputable def batch1 :\n"
"    PProd (Evidence Grade.Verified (HasProperty e1 p obs1_val))\n"
"    (PProd (PLift (ValueGte obs0_val obs1_val))\n"
"    (PProd (Evidence Grade.Verified (HasProperty e2 p obs2_val))\n"
"    (PLift (ValueGte obs0_val obs2_val))))\n"
"    := by\n"
"    refine PProd.mk ?_ (PProd.mk (PLift.up ?_) (PProd.mk ?_ (PLift.up ?_)))\n"
"    . rw [<- obs1_shape]; exact mkVerified obs1_att (claimOfTag obs1_tag)\n"
"    . show ValueGte (Value.rat (mkRat 6 1)) (Value.rat (mkRat 5 1)); unfold ValueGte Value.toRat; native_decide\n"
"    . rw [<- obs2_shape]; exact mkVerified obs2_att (claimOfTag obs2_tag)\n"
"    . show ValueGte (Value.rat (mkRat 6 1)) (Value.rat (mkRat 5 1)); unfold ValueGte Value.toRat; native_decide\n"
"\n"
"  -- Similar for batch2...\n"
"  noncomputable def batch2 :\n"
"    PProd (Evidence Grade.Verified (HasProperty e3 p obs3_val))\n"
"    (PLift (ValueGte obs0_val obs3_val))\n"
"    := by\n"
"    refine PProd.mk ?_ (PLift.up ?_)\n"
"    . rw [<- obs3_shape]; exact mkVerified obs3_att (claimOfTag obs3_tag)\n"
"    . show ValueGte (Value.rat (mkRat 6 1)) (Value.rat (mkRat 4 1)); unfold ValueGte Value.toRat; native_decide\n"
"\n"
"  -- COMPOSING LEMMAS into the final proof:\n"
"  -- IMPORTANT: the final type repeats the batch TYPES, not the batch names.\n"
"  -- The term uses PProd.mk with the batch VALUES (terms).\n"
"\n"
"  -- WRONG: PProd batch1 batch2  (batch1 is a term, not a type!)\n"
"  -- RIGHT:\n"
"  noncomputable def claim_proof :\n"
"    PProd (Evidence Grade.Verified (HasProperty target p obs0_val))\n"
"    (PProd\n"
"      (PProd (Evidence Grade.Verified (HasProperty e1 p obs1_val))\n"
"       (PProd (PLift (ValueGte obs0_val obs1_val))\n"
"       (PProd (Evidence Grade.Verified (HasProperty e2 p obs2_val))\n"
"       (PLift (ValueGte obs0_val obs2_val)))))\n"
"      (PProd (Evidence Grade.Verified (HasProperty e3 p obs3_val))\n"
"       (PLift (ValueGte obs0_val obs3_val))))\n"
"    := by\n"
"    refine PProd.mk ?_ (PProd.mk batch1 batch2)\n"
"    . rw [<- obs0_shape]; exact mkVerified obs0_att (claimOfTag obs0_tag)\n"
"\n"
"  -- Key: the TYPE spells out all Evidence/PLift. The TERM uses batch1, batch2 directly.\n"
"  -- PProd.mk batch1 batch2 works because batch1/batch2 are terms of the right types.\n"
)

# ============================================================
# Pipeline Result
# ============================================================

@dataclass
class PipelineResult:
    claim: str
    table_id: str
    status: str
    grade: str
    formal_goal: str
    goal_type: str
    evidence: list[dict]
    lean_file: str | None
    typecheck_passed: bool
    nl_answer: str
    answer: str = ""  # The Sigma witness — the kernel-verified answer
    steps: list[dict] = field(default_factory=list)
    details: dict = field(default_factory=dict)


# ============================================================
# Pipeline phase types
# ============================================================

@dataclass
class PipelineContext:
    """Shared state across all phases of one run_pipeline call.

    The debug-wrapped llm_call lives here so phases share the same
    wrapper (single debug_tag per call, sequential counter); schema
    is precomputed once and reused.
    """
    claim: str
    table: TableData
    schema_result: SchemaResult
    llm_call: Callable
    common_sense: bool
    batch_lookups: bool
    debug_tag: str
    # Per-table menu enforcement mode. When True, run_solver_loop
    # calls tool_dispatch.is_action_in_per_table_menu before
    # dispatching stats / lookup / lookup_batch actions; off-menu
    # actions are rejected with off_menu_message (no adapter call,
    # no evidence). Default False preserves legacy behavior.
    enforce_menu: bool = False
    # Anomaly-aware formalizer. When True, _run_analyst extracts
    # per-table structural metadata (summary rows, identity column,
    # non-unique columns) from the curated table_formalization and
    # injects it into the formalizer user prompt; the formalizer is
    # expected to flag unresolved structural hazards via the
    # standard ambiguous_terms -> grade downcast path.
    anomaly_aware: bool = False
    # Ambiguity auditor. When True, _run_analyst calls an
    # ambiguity-auditor role BEFORE the formalizer. The auditor sees
    # the natural-language claim only (blind to table / menu), flags
    # multi-reading phrases; output threads into the formalizer
    # prompt and forces has_ambiguity at FormalizationResult level
    # when any phrase was flagged.
    ambiguity_auditor: bool = False
    # Per-table menu injection. When True, _run_analyst injects a
    # PER-TABLE MENU block (the sound-operations subset of the
    # per-table tools list) into the formalizer's user prompt.
    menu_aware: bool = False
    # Contrastive formalizer output. When True, the formalizer is
    # asked to emit an additional `alternative_formalization` field
    # with the runner-up reading it rejected. Transparency-only;
    # does not affect grade.
    contrastive: bool = False


@dataclass
class FormalizationResult:
    """Output of ``run_formalizers``. Always populated; failure is
    indicated by ``is_valid == False`` (i.e. ``valid_outputs`` is
    empty), not by returning ``None``. Callers inspect ``is_valid``
    before consuming ``primary``.

    ``all_outputs`` is every formalizer call's output, preserved on
    both success and failure paths via ``details``.
    ``valid_outputs`` is the filtered subset with a non-empty
    ``goal`` and is what the solver prompt and ambiguity check see.
    """
    primary: dict                    # empty dict when ``is_valid`` is False
    all_outputs: list[dict]
    valid_outputs: list[dict]
    has_ambiguity: bool
    goal: str
    goal_type: str
    evidence_plan: str
    lean_goal_type: str

    @property
    def is_valid(self) -> bool:
        """True iff at least one formalizer call produced a valid goal."""
        return bool(self.valid_outputs)


@dataclass
class SolverLoopResult:
    """Output of ``run_solver_loop``. Carries the full solver
    trace (``evidence``, ``steps``) and the final verdict.

    For variant A, ``refinement_signal`` stays ``None`` and
    ``coverage_observations`` stays empty; the variant-F refinement
    retry path and the coverage-aware grade cap populate them.
    """
    verdict: str            # "VERIFIED" | "REFUTED" | "ABSTAIN"
    evidence: list[dict]
    steps: list[dict]
    lean_file: str | None
    typecheck_passed: bool
    answer: str
    refinement_signal: "RefinementSignal | None" = None
    coverage_observations: tuple["CoverageReport", ...] = ()


def _make_context(
    claim: str,
    table: TableData,
    llm_call: Callable | None,
    backend: Any | None,
    common_sense: bool,
    batch_lookups: bool,
    enforce_menu: bool = False,
    anomaly_aware: bool = False,
    ambiguity_auditor: bool = False,
    contrastive: bool = False,
    menu_aware: bool = False,
) -> PipelineContext:
    """Resolve ``llm_call`` from ``llm_call`` arg or ``backend`` (or
    fall back to ``MockBackend``), wrap with the debug-prompt logger,
    and return a ``PipelineContext``.

    The debug wrapper writes every role's prompt + response to
    ``results/generated/debug_prompts/<tag>_<n>_<role>.txt`` with a
    monotonic counter per context instance. Every phase sees the
    same wrapped ``ctx.llm_call``; no phase should ever touch
    ``backend`` directly.
    """
    import time as _time

    if llm_call is None and backend is not None:
        llm_call = backend.call
    if llm_call is None:
        from llm_backend import MockBackend
        llm_call = MockBackend().call

    debug_dir = GENERATED_DIR / "debug_prompts"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_tag = str(int(_time.time()))[-6:]
    counter = [0]
    raw_llm_call = llm_call

    def wrapped_llm_call(role: str, system_prompt: str, user_prompt: str) -> str:
        counter[0] += 1
        n = counter[0]
        debug_path = debug_dir / f"{debug_tag}_{n:03d}_{role}.txt"
        debug_path.write_text(
            f"=== ROLE: {role} ===\n\n"
            f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n"
            f"=== USER PROMPT ===\n{user_prompt}\n"
        )
        resp = raw_llm_call(role, system_prompt, user_prompt)
        debug_path.write_text(
            debug_path.read_text() + f"\n=== RESPONSE ===\n{resp}\n"
        )
        return resp

    return PipelineContext(
        claim=claim,
        table=table,
        schema_result=schema(table),
        llm_call=wrapped_llm_call,
        common_sense=common_sense,
        batch_lookups=batch_lookups,
        debug_tag=debug_tag,
        enforce_menu=enforce_menu,
        anomaly_aware=anomaly_aware,
        ambiguity_auditor=ambiguity_auditor,
        contrastive=contrastive,
        menu_aware=menu_aware,
    )


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline(
    claim: str,
    table: TableData,
    llm_call=None,
    reading_guide: str | None = None,
    dictionary: Dictionary | None = None,
    backend=None,
    n_formalizers: int = 1,
    batch_lookups: bool = True,
    common_sense: bool = False,
    variant: str = "A",
    question: str | None = None,
    gold_formalization: list | None = None,
    enforce_menu: bool = False,
    prompt_diversity: bool = False,
    anomaly_aware: bool = False,
    ambiguity_auditor: bool = False,
    contrastive: bool = False,
    menu_aware: bool = False,
) -> PipelineResult:
    """Dispatch to a variant implementation.

    variant="A" (default): analyst-driven pipeline.
    variant="B": masked-table pipeline.
    variant="F": masked-table + formalizer refinement retry.

    anomaly_aware: when True, variant A's formalizer receives
    curated per-table structural metadata (summary rows, identity
    column, non-unique columns) to trigger an ambiguous_terms flag
    on structural hazards.

    When ``gold_formalization`` is a validated fixture list, look
    up the (table_id, question) triplet and bypass the LLM
    formalizer. Strict: raises ``ValueError`` if no matching
    entry; the solver loop still runs normally.

    ``question`` is the raw question text used for fixture lookup;
    when ``None``, falls back to ``claim``.
    """
    if variant not in ("A", "B", "F"):
        raise ValueError(f"Unknown variant: {variant!r}. Expected one of 'A', 'B', 'F'.")
    if variant in ("B", "F") and reading_guide is not None:
        raise ValueError(
            f"variant={variant!r} does not accept a reading_guide argument; "
            "the masked-table render is the formalizer's reading guide."
        )

    ctx = _make_context(claim, table, llm_call, backend, common_sense, batch_lookups,
                         enforce_menu=enforce_menu, anomaly_aware=anomaly_aware,
                         ambiguity_auditor=ambiguity_auditor, contrastive=contrastive,
                         menu_aware=menu_aware)

    effective_question = question if question is not None else claim
    gold_form: "FormalizationResult | None" = None
    if gold_formalization is not None:
        gold_form = formalize_from_fixture(
            table.table_id, effective_question, gold_formalization,
        )
        if gold_form is None:
            raise ValueError(
                f"Gold fixture has no entry for table_id={table.table_id!r}, "
                f"question={effective_question[:80]!r}. "
                "Either add the claim to the fixture or drop --gold-formalization."
            )

    if variant == "A":
        return _run_analyst(ctx, reading_guide, dictionary, n_formalizers, gold_form=gold_form,
                            prompt_diversity=prompt_diversity)
    if variant == "B":
        return _run_masked(ctx, dictionary, n_formalizers, gold_form=gold_form,
                           prompt_diversity=prompt_diversity)
    # variant == "F"
    return _run_refinement(ctx, dictionary, n_formalizers, gold_form=gold_form,
                           prompt_diversity=prompt_diversity)


def _run_analyst(
    ctx: PipelineContext,
    reading_guide: str | None,
    dictionary: Dictionary | None,
    n_formalizers: int,
    gold_form: "FormalizationResult | None" = None,
    prompt_diversity: bool = False,
) -> PipelineResult:
    """Variant A: optional analyst call, then formalizer -> solver
    -> render.

    When ``gold_form`` is provided, the LLM formalizer is bypassed.
    The analyst may still be called to produce a reading guide
    used by the solver loop.
    """
    provided_dictionary = dictionary is not None
    d = dictionary if dictionary is not None else init_core_dictionary()
    local_guide = reading_guide

    if local_guide is None:
        print("    [analyst] calling...", end="", flush=True, file=sys.stderr)
        local_guide, analyst_entries = _call_analyst(ctx.table, ctx.schema_result, ctx.llm_call)
        print(f" done ({len(analyst_entries)} entries)", file=sys.stderr)
        if not provided_dictionary:
            for entry_data in analyst_entries:
                try:
                    aliases = []
                    for a in entry_data.get("aliases", []):
                        aliases.append(Alias(
                            surface_form=a.get("surface_form", ""),
                            context=a.get("context", ""),
                            source=a.get("source", "analyst_proposed"),
                            grade_impact=a.get("grade_impact", "no_cap"),
                        ))
                    d.add_entry(DictEntry(
                        name=entry_data.get("name", ""),
                        type=entry_data.get("type", "Property"),
                        description=entry_data.get("description", ""),
                        formalization=entry_data.get("formalization", ""),
                        aliases=aliases,
                    ))
                except Exception as e:
                    print(f"    [analyst entry FAILED] {entry_data.get('name','?')}: {e}", file=sys.stderr)

    variant_meta = {"gold_formalization": True} if gold_form is not None else {}

    # Ambiguity auditor: runs before the formalizer, sees the
    # natural-language claim only and flags multi-reading phrases.
    # Output is appended to the reading guide so the formalizer sees
    # it; if the auditor found any ambiguity, has_ambiguity is forced
    # True regardless of the formalizer's self-flag.
    auditor_output = None
    auditor_flagged_phrases = []
    if getattr(ctx, "ambiguity_auditor", False):
        print("    [ambiguity_auditor] calling...", end="", flush=True, file=sys.stderr)
        auditor_output = _call_ambiguity_auditor(ctx.claim, ctx.llm_call)
        auditor_flagged_phrases = auditor_output.get("ambiguous_phrases", [])
        print(f" done ({len(auditor_flagged_phrases)} flagged, conf={auditor_output.get('confidence','?')})",
              file=sys.stderr)
        if auditor_flagged_phrases:
            import json as _json
            audit_block = (
                "\n\nAMBIGUITY AUDITOR FLAGGED PHRASES (from claim-only audit, "
                "blind to table). Your formalization must honor these:\n"
                + _json.dumps(auditor_flagged_phrases, indent=2)
            )
            local_guide = (local_guide or "") + audit_block

    if gold_form is not None:
        print("    [formalizer bypassed: gold fixture]", file=sys.stderr)
        form = gold_form
    else:
        tf_hint = _extract_tf_hint(ctx.table.table_id) if getattr(ctx, "anomaly_aware", False) else ""
        # Per-table sound-operations menu (structural safety
        # channel; opt-in via --menu flag).
        menu_hint = _format_menu_hint(ctx.table.table_id) if getattr(ctx, "menu_aware", False) else ""
        contrastive_frag = CONTRASTIVE_FRAGMENT if getattr(ctx, "contrastive", False) else ""
        form = run_formalizers(ctx, d, local_guide or "", n_formalizers,
                                prompt_diversity=prompt_diversity,
                                tf_hint=tf_hint,
                                extra_system_fragment=contrastive_frag,
                                menu_hint=menu_hint)
    if not form.is_valid:
        # Even on invalid formalization, propagate auditor info
        if auditor_output is not None:
            variant_meta["ambiguity_auditor"] = auditor_output
        return render_pipeline_result(ctx, form, None, d, "A", variant_meta)

    # If the auditor flagged any ambiguity, force has_ambiguity=True
    # so the grade downcast fires regardless of the formalizer's
    # self-flag. This closes the case where the formalizer misses a
    # reading ambiguity that the auditor caught.
    if auditor_flagged_phrases:
        form.has_ambiguity = True
    if auditor_output is not None:
        variant_meta["ambiguity_auditor"] = auditor_output

    solver = run_solver_loop(ctx, form, d, local_guide or "")
    return render_pipeline_result(ctx, form, solver, d, "A", variant_meta)


def _run_masked(
    ctx: PipelineContext,
    dictionary: Dictionary | None,
    n_formalizers: int,
    gold_form: "FormalizationResult | None" = None,
    prompt_diversity: bool = False,
) -> PipelineResult:
    """Variant B: masked table -> formalizer -> solver -> render.

    When ``gold_form`` is provided, the LLM formalizer is bypassed
    but the masked table and header-derived dictionary are still
    built — the solver loop uses them for evidence formalization
    and prompt context.
    """
    from masked_table import (
        build_masked_table,
        render_masked_for_prompt,
        build_dictionary_from_headers,
        merge_user_dictionary,
    )

    masked = build_masked_table(ctx.table)
    base_dict = build_dictionary_from_headers(masked, init_core_dictionary())
    d = merge_user_dictionary(base_dict, dictionary) if dictionary is not None else base_dict

    masked_prompt = render_masked_for_prompt(masked)

    variant_meta = {
        "masked_anomalies": masked.anomalies,
        "coverage_aware_grading": True,
    }
    if gold_form is not None:
        variant_meta["gold_formalization"] = True

    if gold_form is not None:
        print("    [formalizer bypassed: gold fixture]", file=sys.stderr)
        form = gold_form
    else:
        form = run_formalizers(ctx, d, masked_prompt, n_formalizers,
                                prompt_diversity=prompt_diversity)
    if not form.is_valid:
        return render_pipeline_result(ctx, form, None, d, "B", variant_meta)

    solver = run_solver_loop(ctx, form, d, masked_prompt)
    return render_pipeline_result(ctx, form, solver, d, "B", variant_meta)


def _run_refinement(
    ctx: PipelineContext,
    dictionary: Dictionary | None,
    n_formalizers: int,
    gold_form: "FormalizationResult | None" = None,
    prompt_diversity: bool = False,
) -> PipelineResult:
    """Variant F: variant B + formalizer refinement retry.

    When ``gold_form`` is provided, the LLM formalizer is bypassed
    AND the refinement retry loop is skipped (retrying with the
    same fixture goal cannot refine anything).
    """
    from masked_table import (
        build_masked_table,
        render_masked_for_prompt,
        build_dictionary_from_headers,
        merge_user_dictionary,
    )

    masked = build_masked_table(ctx.table)
    base_dict = build_dictionary_from_headers(masked, init_core_dictionary())
    snapshot_base = deepcopy(base_dict)
    d = merge_user_dictionary(base_dict, dictionary) if dictionary is not None else base_dict
    masked_prompt = render_masked_for_prompt(masked)

    variant_meta = {
        "masked_anomalies": masked.anomalies,
        "coverage_aware_grading": True,
        "retries": 0,
        "last_refinement_kind": None,
    }
    if gold_form is not None:
        variant_meta["gold_formalization"] = True

    # Attempt 1 — fixture-bypass or LLM formalization
    if gold_form is not None:
        print("    [formalizer bypassed: gold fixture]", file=sys.stderr)
        form = gold_form
    else:
        form = run_formalizers(ctx, d, masked_prompt, n_formalizers,
                                prompt_diversity=prompt_diversity)
    if not form.is_valid:
        return render_pipeline_result(ctx, form, None, d, "F", variant_meta)

    solver = run_solver_loop(ctx, form, d, masked_prompt)

    # When formalization came from fixture, retrying with the same
    # gold goal cannot refine it — skip the retry loop entirely.
    if gold_form is not None:
        return render_pipeline_result(ctx, form, solver, d, "F", variant_meta)

    retries = 0
    while (
        solver.verdict == "ABSTAIN"
        and solver.refinement_signal is not None
        and retries < MAX_REFINEMENT_RETRIES
    ):
        retries += 1
        retry_fragment = solver.refinement_signal.render()
        variant_meta["retries"] = retries
        variant_meta["last_refinement_kind"] = solver.refinement_signal.kind

        d = (
            merge_user_dictionary(deepcopy(snapshot_base), dictionary)
            if dictionary is not None
            else deepcopy(snapshot_base)
        )

        form = run_formalizers(
            ctx, d, masked_prompt, n_formalizers,
            extra_prompt_fragment=retry_fragment,
        )
        if not form.is_valid:
            return render_pipeline_result(ctx, form, None, d, "F", variant_meta)

        solver = run_solver_loop(ctx, form, d, masked_prompt)

    return render_pipeline_result(ctx, form, solver, d, "F", variant_meta)


def run_formalizers(
    ctx: PipelineContext,
    dictionary: Dictionary,
    reading_guide: str,
    n_formalizers: int,
    extra_prompt_fragment: str = "",
    prompt_diversity: bool = False,
    tf_hint: str = "",
    extra_system_fragment: str = "",
    menu_hint: str = "",
) -> FormalizationResult:
    """Call ``_call_formalizer`` ``n_formalizers`` times, validating
    each output. Mutates ``dictionary`` via ``_merge_entries_to_dict``.

    Always returns a ``FormalizationResult`` with ``all_outputs``
    populated; the attempted formalizer outputs are preserved on
    both success and failure paths. Callers check
    ``result.is_valid`` to decide whether formalization succeeded.

    ``extra_prompt_fragment`` is always ``""`` for variant A and
    preserves byte-identical formalizer prompts. Variant F populates
    it from a rendered ``RefinementSignal``.
    """
    # Prompt-diversity probe: require n_formalizers to match the
    # variant count and snapshot the dictionary so each call sees the
    # same input (independence requirement).
    diversity_variants: list[tuple[str, str]] = []
    snapshot = None
    if prompt_diversity:
        if n_formalizers != len(PROMPT_DIVERSITY_VARIANTS):
            raise ValueError(
                f"prompt_diversity requires n_formalizers="
                f"{len(PROMPT_DIVERSITY_VARIANTS)}; got {n_formalizers}"
            )
        diversity_variants = list(PROMPT_DIVERSITY_VARIANTS)
        # Deep-copy snapshot so per-call mutations don't leak across
        # the trio (independence requirement for true consensus probe).
        from copy import deepcopy
        snapshot = deepcopy(dictionary)

    formalizations: list[dict] = []
    for i in range(n_formalizers):
        if prompt_diversity:
            label, fragment = diversity_variants[i]
            # Restore the pre-diversity dictionary snapshot before each
            # call so all 3 see the same input. The full mutation set
            # (entries from all 3) is reapplied after the loop, below.
            from copy import deepcopy
            dictionary.entries = deepcopy(snapshot.entries) if snapshot else dictionary.entries
        else:
            # Use the caller-provided extra_system_fragment (e.g., contrastive
            # mode). This never overlaps with prompt_diversity variants because
            # prompt_diversity requires n_formalizers=3 and uses its own
            # fragments; caller should pass "" when prompt_diversity is on.
            label, fragment = ("standard", extra_system_fragment)
        print(f"    [formalizer {i+1}/{n_formalizers} variant={label}] calling...", end="", flush=True, file=sys.stderr)
        f = _call_formalizer(ctx.claim, dictionary, reading_guide, ctx.schema_result, ctx.llm_call,
                             extra_prompt_fragment=extra_prompt_fragment,
                             extra_system_fragment=fragment,
                             tf_hint=tf_hint,
                             menu_hint=menu_hint,
                             table_columns=list(ctx.table.columns))
        # Tag each formalization with its prompt variant for later
        # post-hoc analysis (b-score, calibration, audit).
        f["prompt_variant"] = label
        print(f" done (type={f.get('goal_type', '?')})\n      lean_goal_type: {f.get('lean_goal_type', 'NONE')}", file=sys.stderr)
        formalizations.append(f)
        if not prompt_diversity:
            _merge_entries_to_dict(dictionary, f)

    # In diversity mode: re-apply mutations from ALL calls after the
    # trio completes, so downstream stages (solver) see the union of
    # entries proposed by the diverse formalizers.
    if prompt_diversity and snapshot is not None:
        from copy import deepcopy
        dictionary.entries = deepcopy(snapshot.entries)
        for f in formalizations:
            _merge_entries_to_dict(dictionary, f)

    valid_formalizations = [f for f in formalizations if f.get("goal")]
    if not valid_formalizations:
        return FormalizationResult(
            primary={},
            all_outputs=formalizations,
            valid_outputs=[],
            has_ambiguity=False,
            goal="",
            goal_type="",
            evidence_plan="",
            lean_goal_type="",
        )

    primary = valid_formalizations[0]
    return FormalizationResult(
        primary=primary,
        all_outputs=formalizations,
        valid_outputs=valid_formalizations,
        has_ambiguity=any(f.get("ambiguous_terms") for f in valid_formalizations),
        goal=primary["goal"],
        goal_type=primary.get("goal_type", "other"),
        evidence_plan=primary.get("evidence_plan", ""),
        lean_goal_type=primary.get("lean_goal_type", ""),
    )


def _classify_stats_error(error: str) -> str | None:
    """Map a stats-adapter error string to a closed
    ``ToolFailureReason`` enum value, or ``None`` if the error does
    not match any known actionable pattern.

    Two actionable reasons are recognized:
    - ``column_not_found``: "Column 'X' not found" or
      "Entity column 'X' not found" (both raised by table_adapter.py).
    - ``no_numeric_values``: "No numeric values in" or
      "No matching numeric values for" (both from the adapter).

    All other error shapes (invalid_cond, ratio/lag/pct failures,
    unknown operation, python exceptions) return ``None`` — no
    refinement signal fires and the pipeline abstains without retry.

    String matching is fragile but acceptable here because the
    patterns are pinned by tests against the actual adapter strings
    and unknown patterns degrade to None (no retry rather than a
    wrong retry). A future follow-up should introduce a typed
    failure code on
    ``StatsResult`` so this heuristic can be retired.
    """
    if not isinstance(error, str) or not error:
        return None
    e = error.lower()
    # "Entity column 'X' not found" must NOT be classified as
    # entity_not_found — it's still a column lookup failure, just on
    # the entity-identifier column. We call it column_not_found.
    if "entity column '" in e and "not found" in e:
        return "column_not_found"
    if "column '" in e and "not found" in e:
        return "column_not_found"
    # CondUnknownColumn from eval_cond surfaces as "Column not found
    # in condition: property 'X' not in columns ..." — treat it the
    # same as any other column_not_found so variant-F retries and the
    # solver gets a refinement signal instead of abstaining silently.
    if "column not found in condition" in e:
        return "column_not_found"
    if "no numeric values" in e:
        return "no_numeric_values"
    if "no matching numeric values" in e:
        return "no_numeric_values"
    return None


def _build_refinement_signal(
    verdict: str,
    last_tool_failure: "ToolFailure | None",
    coverage_observations: tuple[CoverageReport, ...],
) -> "RefinementSignal | None":
    """Pure helper that constructs a ``RefinementSignal`` from the
    trusted outcome of a solver loop run.

    Priority: TOOL_FAILURE if ``last_tool_failure`` is non-None,
    else COVERAGE_GAP if any coverage observation is partial, else
    None. Only fires on ``verdict == "ABSTAIN"``.

    The function is deliberately pure: it takes only trusted-tool
    metadata and never inspects solver JSON, kernel error text, or
    the solver's step trace.
    """
    from feedback import RefinementSignal
    if verdict != "ABSTAIN":
        return None
    if last_tool_failure is not None:
        return RefinementSignal(
            kind="TOOL_FAILURE", tool_failure=last_tool_failure,
        )
    for obs in reversed(coverage_observations):
        if obs.is_partial:
            return RefinementSignal(
                kind="COVERAGE_GAP", coverage=obs,
            )
    return None


def run_solver_loop(
    ctx: PipelineContext,
    formalization: FormalizationResult,
    dictionary: Dictionary,
    reading_guide: str,
) -> SolverLoopResult:
    """Run the bounded solver <-> kernel loop.

    Mutates ``dictionary`` via ``_merge_entries_to_dict`` as new
    entities and enrichments arrive from evidence formalization.

    On abstention, builds a ``RefinementSignal`` from trusted-tool
    outcomes; proof events never source a signal.
    """
    # Alias for brevity in the moved loop body
    claim = ctx.claim
    table = ctx.table
    llm_call = ctx.llm_call
    batch_lookups = ctx.batch_lookups
    common_sense = ctx.common_sense

    # Unpack formalization fields the loop references
    formal_goal = formalization.goal
    goal_type = formalization.goal_type
    evidence_plan = formalization.evidence_plan
    primary = formalization.primary
    valid_formalizations = formalization.valid_outputs
    n_formalizers = len(valid_formalizations)

    from feedback import ToolFailure

    evidence: list[dict] = []
    steps: list[dict] = []
    verdict = "ABSTAIN"
    lean_file: str | None = None
    typecheck_passed = False
    coverage_observations: list[CoverageReport] = []
    # Track the most recent actionable stats failure from trusted
    # adapter metadata only (not from solver JSON or kernel text).
    # Reset on a successful stats call so refinement retries fire
    # on the most recent failure the solver encountered.
    last_tool_failure: ToolFailure | None = None

    table_text = _render_table(table)
    consecutive_failures = 0

    for step_num in range(MAX_SOLVER_STEPS):
        # Stagnation check: if last N steps were all non-progress, give up
        if consecutive_failures >= STAGNATION_THRESHOLD:
            print(f"    [stagnation] {consecutive_failures} consecutive failures, giving up", file=sys.stderr)
            steps.append({"action": "stagnation", "feedback": f"ABSTAIN after {consecutive_failures} consecutive failures"})
            break

        # Build context for solver
        user_prompt = _build_solver_prompt(
            formal_goal, goal_type, evidence_plan, table_text, evidence, steps,
            formalizer_output=primary,
            all_formalizations=valid_formalizations if n_formalizers > 1 else None,
        )

        print(f"    [solver step {step_num+1}] calling...", end="", flush=True, file=sys.stderr)
        raw = llm_call("solver", SOLVER_SYSTEM, user_prompt)
        action = _parse_json(raw)
        if action is None:
            print(f" parse error", file=sys.stderr)
            steps.append({"action": "parse_error", "feedback": "Could not parse JSON. Output valid JSON."})
            continue

        action_type = action.get("action", "")

        # Per-table menu enforcement. When ctx.enforce_menu is
        # True, gate stats / lookup / lookup_batch through
        # tool_dispatch.is_action_in_per_table_menu before any
        # adapter call. Off-menu actions reject with the canonical
        # off_menu_message (shared with baseline_agent). Other
        # action shapes (give_up, ask_common_sense, eval, ...) are
        # out of scope and pass through.
        if ctx.enforce_menu and action_type in ("stats", "lookup", "lookup_batch"):
            from tool_dispatch import is_action_in_per_table_menu, off_menu_message
            on_menu, _reason = is_action_in_per_table_menu(table.table_id, action)
            if not on_menu:
                feedback = off_menu_message(table.table_id, action)
                print(f" {action_type} rejected: off-menu", file=sys.stderr)
                if action_type == "lookup_batch":
                    n_entries = len(action.get("lookups", []) or [])
                    _append_tool_step(steps, action_type, action, success=False,
                                      per_entry_status=[PER_ENTRY_STATUS_OFF_MENU] * max(n_entries, 1),
                                      feedback=feedback, off_menu=True)
                else:
                    _append_tool_step(steps, action_type, action, success=False,
                                      feedback=feedback, off_menu=True)
                consecutive_failures += 1
                continue

        # --- LOOKUP ---
        if action_type == "lookup":
            entity = action.get("entity", "")
            prop = action.get("property", "")
            # Soundness: restrict entity search to the per-table
            # cell_lookup lift's identity column when present.
            _id_col = _resolve_lookup_identity(table.table_id)
            result = _execute_lookup(table, entity, prop, identity_col=_id_col)

            if result is not None:
                # Thread the canonical property name returned by
                # strict lookup into evidence formalization.
                # Otherwise the L1 emission would write the
                # solver's typographic variant into a CellAt
                # premise the actual table does not have.
                _canonical_prop = result.get("property", prop)
                ev = _formalize_evidence(entity, _canonical_prop, result,
                                         len(evidence),
                                         llm_call=llm_call, dictionary=dictionary,
                                         reading_guide=reading_guide)
                evidence.append(ev)
                # Dictionary grows after every evidence formalization
                if ev.get("_formalizer_output"):
                    _merge_entries_to_dict(dictionary, ev.pop("_formalizer_output"))
                feedback = f"OK: {entity}.{prop} = {result['raw_value']}. Observation name: obs{len(evidence)-1}"
            else:
                feedback = f"NOT FOUND: no exact unique match for '{entity}' in column '{prop}'. Available columns: {table.columns}. Available values in first column ({table.columns[0]}): {[r[0] for r in table.rows[:5]]}"

            print(f" lookup {entity}.{prop} → {feedback[:40]}", file=sys.stderr)
            _append_tool_step(steps, "lookup", action,
                              success=(result is not None),
                              evidence_added=(1 if result is not None else 0),
                              entity=entity, property=prop, feedback=feedback)
            if result is not None:
                consecutive_failures = 0  # progress
            else:
                consecutive_failures += 1

        # --- BATCH LOOKUP ---
        elif action_type == "lookup_batch" and batch_lookups:
            lookups = action.get("lookups", [])
            if not lookups:
                # Trace contract: empty batch is MALFORMED at the batch level.
                _append_tool_step(steps, "lookup_batch", action, success=False,
                                  per_entry_status=[PER_ENTRY_STATUS_MALFORMED],
                                  feedback="ERROR: 'lookups' list is empty.")
                continue

            batch_feedback = []
            any_success = False
            per_entry: list[str] = []
            n_evidence_added = 0
            _id_col_batch = _resolve_lookup_identity(table.table_id)
            for lk in lookups:
                if not isinstance(lk, dict):
                    per_entry.append(PER_ENTRY_STATUS_MALFORMED)
                    batch_feedback.append(f"MALFORMED: entry not a dict")
                    continue
                entity = lk.get("entity", "")
                prop = lk.get("property", "")
                result = _execute_lookup(table, entity, prop, identity_col=_id_col_batch)

                if result is not None:
                    # Use the canonical property name from the
                    # strict lookup (avoids leaking the solver's
                    # typographic variant into the L1 emission).
                    _canonical_prop = result.get("property", prop)
                    ev = _formalize_evidence(entity, _canonical_prop, result,
                                             len(evidence),
                                             llm_call=llm_call, dictionary=dictionary,
                                             reading_guide=reading_guide)
                    evidence.append(ev)
                    if ev.get("_formalizer_output"):
                        _merge_entries_to_dict(dictionary, ev.pop("_formalizer_output"))
                    batch_feedback.append(f"OK: {entity}.{prop} = {result['raw_value']}. obs{len(evidence)-1}")
                    any_success = True
                    per_entry.append(PER_ENTRY_STATUS_OK)
                    n_evidence_added += 1
                else:
                    batch_feedback.append(f"NOT FOUND: {entity}.{prop}. Available columns: {table.columns}")
                    per_entry.append(PER_ENTRY_STATUS_NOT_FOUND)

            feedback = f"BATCH ({len(lookups)} lookups, {sum(1 for f in batch_feedback if f.startswith('OK'))} found):\n" + "\n".join(f"  {f}" for f in batch_feedback)
            print(f" batch {len(lookups)} lookups → {sum(1 for f in batch_feedback if f.startswith('OK'))} found", file=sys.stderr)
            _append_tool_step(steps, "lookup_batch", action,
                              success=any_success,
                              evidence_added=n_evidence_added,
                              per_entry_status=per_entry,
                              feedback=feedback)
            consecutive_failures = 0 if any_success else consecutive_failures + 1

        # --- STATS ---
        elif action_type == "stats":
            operation = action.get("operation", "")
            column = action.get("column", "")
            match_value = action.get("match_value")
            condition = action.get("condition")  # Cond JSON for filtered aggregation

            entity_col = action.get("entity_column")

            # Resolve the per-table tool instance for
            # (table, operation, column). When found, the curator's
            # tool config (parse_mode etc.) and identity_column are
            # authoritative; they override the solver's hints. The
            # tool instance encodes how to read this table's
            # semantics so the solver does not have to infer it
            # from the column header alone.
            _resolved_tool = _resolve_tool_instance(table.table_id, operation, column)
            _tool_parse_mode = "strict"
            if _resolved_tool:
                _tool_parse_mode = _resolved_tool["config"].get("parse_mode", "strict")
                if _resolved_tool.get("identity_column"):
                    entity_col = _resolved_tool["identity_column"]
            elif operation in ("argmax", "argmin") and primary:
                # Fallback: if there is no per-table tool instance
                # but the per-claim table_formalization has a lift
                # with identity_column, use that.
                _tf_raw = primary.get("table_formalization", "")
                _, _tf_lifts = _normalize_table_formalization(_tf_raw)
                _matched_lift = _find_matching_lift(_tf_lifts, operation)
                if _matched_lift and _matched_lift.get("identity_column"):
                    entity_col = _matched_lift["identity_column"]

            n_val = action.get("n", 5)
            scale = action.get("scale", 100)
            bottom = action.get("bottom", False)

            num_col = action.get("numerator_column", "")
            den_col = action.get("denominator_column", "")
            sort_col = action.get("sort_column", "")

            # REMOVED_OPS lives at module level so the live dispatch
            # and the validator share one source of truth. Gate fires
            # before any cond-unsupported check so removed-op-with-
            # condition rejects with the correct REMOVED_OP reason.
            if operation in REMOVED_OPS:
                feedback = removed_op_message(operation)
                print(f" stats {operation} rejected: REMOVED_OP", file=sys.stderr)
                _append_tool_step(steps, "stats", action, success=False,
                                  operation=operation, column=column,
                                  feedback=feedback)
                consecutive_failures += 1
                continue
            # Reject Cond-bearing actions on ops whose adapter
            # ignores condition. With current REMOVED_OPS hoisting
            # this set is empty; the check is a safety net for any
            # future cond-blind op that is not in REMOVED_OPS.
            _cond_unsupported_ops: tuple[str, ...] = ()
            if condition and operation in _cond_unsupported_ops:
                feedback = (
                    f"ERROR: operation '{operation}' does not support the "
                    f"'condition' argument. Use a coverage-aware "
                    f"alternative (sum/max/min/mean with condition)."
                )
                print(f" stats {operation} rejected: condition not supported", file=sys.stderr)
                _append_tool_step(steps, "stats", action, success=False,
                                  operation=operation, column=column,
                                  feedback=feedback)
                consecutive_failures += 1
                continue
            # count_where counts rows satisfying cond, independent
            # of the aggregation column (which is retained for the
            # CountWhere WProp shape).
            if operation == "count_where":
                from table_adapter import stats_count_where, cond_to_lean
                sr = stats_count_where(table, column, condition or {"op": "all"})
            elif operation in TOPK_ALL_OPS:
                from table_adapter import stats_topk_where, cond_to_lean
                # Shared rank_by gate so live dispatch and validator
                # produce byte-identical rejection text.
                err, rank_by = check_topk_rank_by(operation, action.get("rank_by"))
                if err is not None:
                    print(f" stats {operation} rejected: {err.split(chr(10))[0][:80]}", file=sys.stderr)
                    _append_tool_step(steps, "stats", action, success=False,
                                      operation=operation, column=column,
                                      feedback=err)
                    consecutive_failures += 1
                    continue
                sr = stats_topk_where(table, operation, column, n_val,
                                      condition, rank_by, parse_mode=_tool_parse_mode)
            # If a condition is provided or mean is requested, use
            # the general stats_agg_where. Bare 'count' is rejected
            # upstream; solvers must call count_where explicitly.
            elif condition and operation in ("sum", "max", "min", "mean"):
                from table_adapter import stats_agg_where, cond_to_lean
                sr = stats_agg_where(table, operation, column, condition,
                                     parse_mode=_tool_parse_mode)
            elif operation == "mean" and not condition:
                from table_adapter import stats_agg_where, cond_to_lean
                sr = stats_agg_where(table, "mean", column, {"op": "all"},
                                     parse_mode=_tool_parse_mode)
            else:
                # Bare-op fallthrough. The REMOVED_OPS gate at the
                # top of the stats dispatch already rejected
                # unsupported ops, so the table here only enumerates
                # currently supported operations.
                stats_ops = {
                    "count_distinct": lambda: stats_count_distinct(table, column),
                    "sum": lambda: stats_sum(table, column, parse_mode=_tool_parse_mode),
                    "max": lambda: stats_max(table, column, parse_mode=_tool_parse_mode),
                    "min": lambda: stats_min(table, column, parse_mode=_tool_parse_mode),
                    "mean": lambda: stats_mean(table, column, scale, parse_mode=_tool_parse_mode),
                    "argmax": lambda: stats_argmax(table, column, entity_col, parse_mode=_tool_parse_mode, condition=condition),
                    "argmin": lambda: stats_argmin(table, column, entity_col, parse_mode=_tool_parse_mode, condition=condition),
                }
                if operation not in stats_ops:
                    feedback = (f"ERROR: unknown operation '{operation}'. "
                                f"Documented surface: count_where, count_distinct, sum, max, min, mean, "
                                f"argmax, argmin, topk_{{sum,mean}}, bottomk_{{sum,mean}}, topk_{{sum,mean}}_by, cell_lookup.")
                    _append_tool_step(steps, "stats", action, success=False,
                                      operation=operation, column=column,
                                      feedback=feedback)
                    continue
                sr = stats_ops[operation]()
            if sr.success:
                # Map operation to world-level predicate. Bare
                # 'count' / 'median' are absent because the
                # REMOVED_OPS gate above rejects them; use
                # count_where with Cond.all instead.
                predicate_map = {
                    "sum": "SumWhere",
                    "count_where": "CountWhere",
                    "count_distinct": "CountDistinctWhere",
                    "max": "MaxWhere", "min": "MinWhere",
                    "mean": "MeanWhere",
                    "topk_sum": "TopKSumWhere",
                    "topk_mean": "TopKMeanWhere",
                    "topk_sum_by": "TopKSumByWhere",
                    "topk_mean_by": "TopKMeanByWhere",
                    "bottomk_sum": "BottomKSumWhere",
                    "bottomk_mean": "BottomKMeanWhere",
                }
                col_id = _to_lean_id(column)
                # All numeric values as Value.rat for consistency
                from fractions import Fraction
                from table_adapter import _fraction_to_lean, _fraction_to_lean_paren
                if isinstance(sr.value, Fraction):
                    f = sr.value
                else:
                    f = Fraction(sr.value)
                lean_val = _fraction_to_lean(f)
                lean_val_paren = _fraction_to_lean_paren(f)
                predicate = predicate_map.get(operation, "SumWhere")

                # Build Lean Cond from the JSON condition (or Cond.all if no filter)
                from table_adapter import cond_to_lean
                lean_cond = cond_to_lean(condition) if condition else "Cond.all"

                # Build WProp
                _topk_ops = ("topk_sum", "topk_mean", "topk_sum_by",
                             "topk_mean_by", "bottomk_sum", "bottomk_mean")
                if operation in ("argmax", "argmin"):
                    ent_id = _to_lean_id(str(sr.entity)) if sr.entity else "unknown"
                    pred = "ArgmaxWhere" if operation == "argmax" else "ArgminWhere"
                    wprop = f"{pred} {col_id} {lean_cond} {ent_id} {lean_val_paren}"
                    # Phase C metadata for L1+L2 emission via per-table lift
                    # (specs/runtime-architecture.md §1.3, §2 step 5):
                    # - operation: matches the lift's `operation` field for dispatch
                    # - row_idx: which row achieved the extreme (for stable RowHandle)
                    # - entity_col: which column the entity name was read from
                    #   (matches the lift's `identity_column` field)
                    ev = {"entity": str(sr.entity), "entity_id": ent_id,
                          "property": column, "property_id": col_id,
                          "raw_value": str(sr.value), "lean_value": lean_val,
                          "wprop": wprop, "source": "stats",
                          "operation": operation,
                          "row_idx": sr.row_idx,
                          "entity_col": sr.entity_col,
                          "lean_cond": lean_cond,
                          "tool_instance_id": (_resolved_tool["tool_instance_id"] if _resolved_tool else None),
                          "agg_entities": [ent_id], "agg_properties": [col_id]}
                elif operation in _topk_ops:
                    k_int = int(n_val)
                    rank_by = action.get("rank_by")
                    if operation in ("topk_sum_by", "topk_mean_by") and rank_by:
                        rank_id = _to_lean_id(rank_by)
                        wprop = (f"{predicate} {col_id} {rank_id} {k_int} "
                                 f"{lean_cond} {lean_val_paren}")
                        base_props = [col_id, rank_id]
                    else:
                        wprop = f"{predicate} {col_id} {k_int} {lean_cond} {lean_val_paren}"
                        base_props = [col_id]
                    cond_props = list(base_props)
                    def _extract_cond_props(c):
                        if not c or not isinstance(c, dict): return
                        if c.get("property"):
                            cond_props.append(_to_lean_id(c["property"]))
                        _extract_cond_props(c.get("left"))
                        _extract_cond_props(c.get("right"))
                        _extract_cond_props(c.get("inner"))
                    if condition:
                        _extract_cond_props(condition)
                    ev = {"entity": "table_stats", "entity_id": "table_stats",
                          "property": f"{operation}_{column}",
                          "property_id": f"{operation}_{col_id}",
                          "column_id": col_id,
                          "raw_value": str(sr.value), "lean_value": lean_val,
                          "wprop": wprop, "source": "stats",
                          "operation": operation,
                          "lean_cond": lean_cond,
                          # k and rank_by are available for the
                          # L1+L2 topk dispatch (rank_by used by
                          # the *_by variants).
                          "n": k_int,
                          "rank_by": rank_by,
                          "tool_instance_id": (_resolved_tool["tool_instance_id"] if _resolved_tool else None),
                          "agg_entities": [], "agg_properties": cond_props}
                else:
                    wprop = f"{predicate} {col_id} {lean_cond} {lean_val_paren}"
                    # Collect all properties referenced in the condition for axiom declarations
                    cond_props = [col_id]
                    def _extract_cond_props(c):
                        if not c or not isinstance(c, dict): return
                        if c.get("property"):
                            cond_props.append(_to_lean_id(c["property"]))
                        _extract_cond_props(c.get("left"))
                        _extract_cond_props(c.get("right"))
                        _extract_cond_props(c.get("inner"))
                    if condition:
                        _extract_cond_props(condition)
                    ev = {"entity": "table_stats", "entity_id": "table_stats",
                          "property": f"{operation}_{column}", "property_id": f"{operation}_{col_id}",
                          "column_id": col_id,
                          "raw_value": str(sr.value), "lean_value": lean_val,
                          "wprop": wprop, "source": "stats",
                          "operation": operation,
                          "lean_cond": lean_cond,
                          "tool_instance_id": (_resolved_tool["tool_instance_id"] if _resolved_tool else None),
                          "agg_entities": [], "agg_properties": cond_props}

                evidence.append(ev)
                # Collect coverage observations for grade capping;
                # only successful stats evidence contributes.
                if sr.coverage is not None:
                    coverage_observations.append(sr.coverage)
                # A successful stats call resets the pending
                # tool-failure signal: refinement only retries on
                # the most recent failure, and a success since the
                # last failure means the solver has moved past it.
                last_tool_failure = None
                feedback = f"OK: {operation}({column}" + (f", '{match_value}'" if match_value else "") + f") = {sr.value}. {sr.details}. obs{len(evidence)-1}. WProp: {wprop}"
            else:
                # Classify the failure into a closed enum reason
                # (or None if unrecognized). On recognized failures
                # construct a ToolFailure from trusted adapter
                # metadata only: never copy column / operation
                # values from the solver's raw action JSON.
                reason = _classify_stats_error(sr.error or "")
                if reason is not None:
                    # BARRIER: the column field can only be populated
                    # from a name that CANONICALLY EXISTS in the
                    # trusted table. Check table.columns directly —
                    # never copy from sr.coverage.column (which was
                    # derived from the solver-provided agg_col in
                    # the adapter call). On column_not_found, by
                    # definition no canonical match exists, so set
                    # column=None.
                    adapter_column: str | None = None
                    if reason != "column_not_found" and sr.coverage is not None:
                        raw_col = sr.coverage.column
                        if raw_col and raw_col not in ("<unknown>", "<row count>"):
                            # Verify the name canonically appears in
                            # the trusted table (case-insensitive).
                            for trusted_col in table.columns:
                                if trusted_col.strip().lower() == raw_col.strip().lower():
                                    adapter_column = trusted_col
                                    break
                    # `operation` was read from the solver JSON
                    # action and is therefore solver-provided. The
                    # barrier is preserved by ToolFailure's
                    # __post_init__, which validates operation
                    # against the closed VALID_OPERATIONS frozenset;
                    # any non-enum value raises ValueError and the
                    # signal falls through to last_tool_failure=None.
                    try:
                        last_tool_failure = ToolFailure(
                            tool="stats",
                            operation=operation,
                            reason=reason,
                            column=adapter_column,
                        )
                    except ValueError:
                        # operation wasn't in VALID_OPERATIONS.
                        # Don't fire a signal for this action.
                        last_tool_failure = None
                feedback = f"ERROR: {sr.error}"

            print(f" stats {operation}({column[:20]}) → {sr.value if sr.success else 'ERR'}", file=sys.stderr)
            _append_tool_step(steps, "stats", action,
                              success=sr.success,
                              evidence_added=(1 if sr.success else 0),
                              operation=operation, column=column,
                              feedback=feedback)
            if sr.success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1

        # --- ASK COMMON SENSE ---
        elif action_type == "ask_common_sense" and common_sense:
            question = action.get("question", "")
            proposition = action.get("proposition", "")

            if not question:
                print(f" empty question", file=sys.stderr)
                steps.append({"action": "ask_common_sense", "feedback": "ERROR: 'question' is required."})
                continue

            # Call common sense agent (same LLM, different role)
            cs_prompt = (
                f"Answer this factual question with a single word: YES or NO.\n\n"
                f"Question: {question}\n\n"
                f"Answer (YES or NO):"
            )
            cs_answer = llm_call("common_sense", "You are a factual knowledge oracle. Answer YES or NO only.", cs_prompt).strip().upper()
            confirmed = cs_answer.startswith("YES")

            if confirmed and proposition:
                # Formalize the common sense fact as Plausible evidence
                cs_ev = _formalize_cs_evidence(proposition, len(evidence), llm_call, dictionary, reading_guide)
                evidence.append(cs_ev)
                if cs_ev.get("_formalizer_output"):
                    _merge_entries_to_dict(dictionary, cs_ev.pop("_formalizer_output"))
                feedback = f"CONFIRMED (Plausible): {question} → YES. Observation: cs{len(evidence)-1}. Grade capped at Plausible."
            elif not confirmed:
                feedback = f"DENIED: {question} → {cs_answer}. The common sense agent says NO."
            else:
                feedback = f"CONFIRMED but no proposition given. Include 'proposition' to formalize."

            print(f" common_sense: {question[:30]} → {cs_answer[:3]}", file=sys.stderr)
            steps.append({"action": "ask_common_sense", "question": question, "feedback": feedback})
            consecutive_failures = 0 if confirmed else consecutive_failures + 1

        # --- PROVE ---
        elif action_type == "prove":
            claim_holds = action.get("claim_holds", True)
            proof_body = action.get("proof_body", "")

            if not proof_body:
                print(f" empty proof", file=sys.stderr)
                steps.append({"action": "prove", "feedback": "REJECTED: proof_body is empty. Write ONLY the tactic proof (after := by)."})
                continue

            # Get the trusted goal type from formalizer
            lean_goal_type = primary.get("lean_goal_type", "")
            # Per-table formalization (new, Phase A): curated Lean
            # axioms for this specific table, injected into the
            # generated file alongside the claim goal. Empty for
            # entries that don't carry one.
            table_formalization = primary.get("table_formalization", "")

            # Build full .lean file — pipeline controls the type, solver provides only tactics
            lean_content = _build_lean_file(
                claim, table.table_id, formal_goal, claim_holds, evidence, proof_body,
                lean_goal_type=lean_goal_type,
                table_formalization=table_formalization,
            )

            GENERATED_DIR.mkdir(parents=True, exist_ok=True)
            lean_path = GENERATED_DIR / f"pipeline_{table.table_id}_{hash(claim) % 10000:04d}.lean"
            lean_path.write_text(lean_content)

            print(f" kernel check...", end="", flush=True, file=sys.stderr)
            success, output = _typecheck(lean_path)

            if success:
                # Check no sorry
                has_sorry = "sorryAx" in output or "sorry" in output.lower().split("depends on axioms:")[1] if "depends on axioms:" in output else False
                if has_sorry:
                    print(f" REJECTED (sorry)", file=sys.stderr)
                    steps.append({"action": "prove", "feedback": "REJECTED: proof contains sorry. Write a complete proof."})
                else:
                    verdict = "VERIFIED" if claim_holds else "REFUTED"
                    lean_file = str(lean_path)
                    typecheck_passed = True
                    # Extract the Sigma witness (answer) from the proof_body
                    sigma_answer = _extract_sigma_witness(proof_body)
                    print(f" ACCEPTED → {verdict} (answer={sigma_answer})", file=sys.stderr)
                    steps.append({"action": "prove", "verdict": verdict, "feedback": "ACCEPTED", "answer": sigma_answer})
                    break
            else:
                error_msg = output.strip()
                # Give the solver the FULL error output so it can diagnose and fix.
                # Lean errors are precise — line numbers, expected tokens, type mismatches.
                # Truncating them is the #1 reason the solver can't self-correct.
                error_summary = error_msg
                print(f" REJECTED", file=sys.stderr)
                steps.append({"action": "prove", "feedback": f"REJECTED by kernel:\n{error_summary}"})

        # --- GIVE UP ---
        elif action_type == "give_up":
            print(f" give up: {action.get('reason', '')[:50]}", file=sys.stderr)
            steps.append({"action": "give_up", "feedback": "ABSTAIN"})
            break

        else:
            print(f" unknown action", file=sys.stderr)
            steps.append({"action": "unknown", "feedback": f"Unknown action '{action_type}'. Use lookup, prove, or give_up."})

    # Extract answer from the accepted proof's Sigma witness, if any
    answer = ""
    for s in steps:
        if s.get("answer"):
            answer = s["answer"]

    # Construct the refinement signal from trusted metadata only.
    # Variant F's retry path reads this field (and nothing else) to
    # decide whether to retry.
    refinement_signal = _build_refinement_signal(
        verdict=verdict,
        last_tool_failure=last_tool_failure,
        coverage_observations=tuple(coverage_observations),
    )

    return SolverLoopResult(
        verdict=verdict,
        evidence=evidence,
        steps=steps,
        lean_file=lean_file,
        typecheck_passed=typecheck_passed,
        answer=answer,
        refinement_signal=refinement_signal,
        coverage_observations=tuple(coverage_observations),
    )


def grade_result(
    formalization: FormalizationResult,
    solver: SolverLoopResult,
) -> str:
    """Compute the final grade for this claim.

    Coverage-aware cap: any stats evidence whose
    ``CoverageReport.is_partial`` is True within its Cond-eligible
    rows drops the grade to Supported. This is independent of the
    existing ambiguity cap; either triggers Supported.

    Gates on ``solver.typecheck_passed``, not ``solver.verdict``;
    REFUTED proofs are kernel-accepted and receive a grade.

    Cap precedence (weakest cap wins):
      - N/A: solver did not typecheck
      - Plausible: common-sense evidence (strictest cap)
      - Supported: ambiguity OR partial coverage
      - Verified: none of the above

    If both common-sense AND partial coverage are present,
    Plausible wins (strictly weaker than Supported).
    """
    if not solver.typecheck_passed:
        return "N/A"
    has_cs = any(ev.get("source") == "common_sense" for ev in solver.evidence)
    if has_cs:
        return "Plausible"
    if formalization.has_ambiguity:
        return "Supported"
    if any(obs is not None and obs.is_partial for obs in solver.coverage_observations):
        return "Supported"
    return "Verified"


def render_pipeline_result(
    ctx: PipelineContext,
    formalization: FormalizationResult | None,
    solver: SolverLoopResult | None,
    dictionary: Dictionary,
    variant: str,
    variant_meta: dict,
) -> PipelineResult:
    """Compose the final PipelineResult.

    Three branches:
    1. Formalization failed (``formalization is None``) → ABSTAIN with
       the exact ``"No valid formalization produced."`` NL answer.
       Does NOT call ``_render_nl_answer``, which would fall through
       to ``"Could not verify this claim."`` instead.
    2. Solver absent (defensive): treat as ABSTAIN with empty state.
    3. Solver present: compute grade, call ``_render_nl_answer`` for
       VERIFIED/REFUTED, return the full result.

    ``variant_meta`` is merged into ``details``. For variant A it's
    typically empty; stages 5 and 7 populate it with masked-table
    anomaly counts and refinement-retry metadata.
    """
    # Case 1: formalization failure (None or not valid)
    if formalization is None or not formalization.is_valid:
        attempted = formalization.all_outputs if formalization is not None else []
        details = {"formalizer_outputs": attempted, "variant": variant}
        details.update(variant_meta)
        return PipelineResult(
            claim=ctx.claim,
            table_id=ctx.table.table_id,
            status="ABSTAIN",
            grade="N/A",
            formal_goal="",
            goal_type="",
            evidence=[],
            lean_file=None,
            typecheck_passed=False,
            nl_answer="No valid formalization produced.",
            details=details,
        )

    # Case 2 (defensive): solver absent
    if solver is None:
        solver = SolverLoopResult(
            verdict="ABSTAIN",
            evidence=[],
            steps=[],
            lean_file=None,
            typecheck_passed=False,
            answer="",
        )

    # Case 3: formalization + solver present
    nl_answer = _render_nl_answer(
        ctx.claim, solver.verdict, formalization.goal,
        solver.evidence, dictionary, ctx.llm_call,
    )
    grade = grade_result(formalization, solver)

    details = {
        "formalizer_outputs": formalization.all_outputs,
        "has_ambiguity": formalization.has_ambiguity,
        "variant": variant,
    }
    details.update(variant_meta)

    return PipelineResult(
        claim=ctx.claim,
        table_id=ctx.table.table_id,
        status=solver.verdict,
        grade=grade,
        formal_goal=formalization.goal,
        goal_type=formalization.goal_type,
        evidence=solver.evidence,
        lean_file=solver.lean_file,
        typecheck_passed=solver.typecheck_passed,
        nl_answer=nl_answer,
        answer=solver.answer,
        steps=solver.steps,
        details=details,
    )


# ============================================================
# Helpers
# ============================================================

def _build_solver_prompt(goal, goal_type, plan, table_text, evidence, steps,
                         formalizer_output=None, all_formalizations=None):
    # Design intent: the solver is blind to the natural-language
    # claim. It receives only the formal goal type, evidence, and
    # table; this prevents back-fill from NL intent that could
    # silently mask formalization errors.
    parts = [
        f"GOAL TYPE: {goal_type}\n",
        f"EVIDENCE PLAN: {plan}\n",
    ]

    # Show ALL formalizations — solver must prove equivalence before using
    if all_formalizations and len(all_formalizations) > 1:
        parts.append("THREE INDEPENDENT FORMALIZATIONS of this claim:")
        parts.append("You MUST first prove these are equivalent (or pick one and prove the others")
        parts.append("follow from it) before proceeding to prove the claim itself.")
        parts.append("Use a 'prove' action with claim_holds=true and a proof body that includes")
        parts.append("equivalence lemmas. If you cannot prove equivalence, give_up.\n")
        for i, f in enumerate(all_formalizations):
            parts.append(f"  Formalization {i+1}:")
            parts.append(f"    Informal: {f.get('goal', '')}")
            parts.append(f"    Lean type: {f.get('lean_goal_type', 'MISSING')}")
            parts.append(f"    Goal type: {f.get('goal_type', '?')}")
            parts.append("")
    else:
        parts.append(f"FORMAL GOAL: {goal}\n")

    # Show the EXACT Lean type the solver must prove
    if formalizer_output and formalizer_output.get("lean_goal_type"):
        parts.append(f"LEAN GOAL TYPE (you must prove THIS exact type — match every predicate and identifier):")
        parts.append(f"  {formalizer_output['lean_goal_type']}\n")

    # Include formalizer's property/entity mappings
    if formalizer_output:
        props = formalizer_output.get("properties_needed", [])
        ents = formalizer_output.get("entities_needed", [])
        if props or ents:
            parts.append("FORMALIZER MAPPINGS (formal name → table column/value):")
            for p in props:
                parts.append(f"  property '{p.get('name','')}' → column '{p.get('maps_to_column','')}'")
            for e in ents:
                parts.append(f"  entity '{e.get('name','')}' → lookup key '{e.get('lookup_key','')}' in column '{e.get('lookup_column','')}'")
            parts.append("")

    parts.append(f"TABLE:\n{table_text}\n")

    # Per-table tool menu summary: show which operations are
    # available on this table, grouped by family. Source of truth
    # is the `tools` list in the per-table formalization;
    # library-internal config (parse_mode, identity_column,
    # library_ref) is not exposed because the runtime applies it
    # silently. Rendered between TABLE and EVIDENCE/LEAN FILE
    # SKELETON so the solver sees the local menu before reading the
    # Lean-side lift axioms.
    tf_for_menu = formalizer_output.get("table_formalization", "") if formalizer_output else ""
    # _normalize_table_formalization may have already converted to
    # a dict via embedded JSON parsing; pass the raw structured form
    # when present.
    if isinstance(tf_for_menu, dict):
        _menu_lines = _summarize_table_tools(tf_for_menu)
    else:
        _menu_lines = []
    if _menu_lines:
        parts.extend(_menu_lines)

    if evidence:
        # Each obs shows its full WProp and whether it matches the goal
        goal_type = formalizer_output.get("lean_goal_type", "") if formalizer_output else ""
        parts.append("EVIDENCE COLLECTED:")
        for i, ev in enumerate(evidence):
            if ev.get("source") == "stats":
                wprop = ev['wprop']
            else:
                wprop = f"HasProperty {ev['entity_id']} {ev['property_id']} ({ev['lean_value']})"
            relevant = "✓ MATCHES GOAL" if wprop in goal_type else ""
            parts.append(f"  obs{i}: {wprop}  {relevant}")
        parts.append("")

        parts.append("LEAN FILE SKELETON (observations already declared):")
        parts.append("```")
        # Show the goal type the solver must prove
        if formalizer_output and formalizer_output.get("lean_goal_type"):
            parts.append(f"-- GOAL TYPE (you must prove this):")
            parts.append(f"-- noncomputable def claim_proof : {formalizer_output['lean_goal_type']} := by")
            parts.append("")
        # Per-table formalization axioms: audited per-table Lean
        # declarations (lift lemmas). Visible to the solver so it
        # can use them via `exact <axiom_name>`. When a per-table
        # lift matches a stats operation, the runtime emits a
        # ready-to-use L2 def `obs{i}_l2` that the solver should
        # consume directly.
        tf_raw = formalizer_output.get("table_formalization", "") if formalizer_output else ""
        tf_text, tf_lifts = _normalize_table_formalization(tf_raw)
        if tf_text:
            parts.append("-- === PER-TABLE FORMALIZATION (audited axioms for THIS table) ===")
            parts.append(tf_text.rstrip())
            if tf_lifts:
                parts.append("-- The runtime auto-applies matching lifts and emits `obs{i}_l2` defs;")
                parts.append("-- prefer `exact obs{i}_l2` over composing lifts manually.")
            else:
                parts.append("-- You may use these directly in your proof via `exact <axiom_name>`.")
            parts.append("")
        for i, ev in enumerate(evidence):
            if ev.get("source") == "stats":
                operation = ev.get("operation", "")
                # Pass tool_instance_id so the solver-facing
                # skeleton picks the same lift _build_lean_file
                # picks. Without this, multi-instance same-op
                # tables could show a different lift name in the
                # prompt than in the generated Lean file.
                lift = _find_matching_lift(tf_lifts, operation, ev.get("tool_instance_id"))
                if (lift and lift.get("identity_column")
                        and operation in ("argmax", "argmin")):
                    # Mirror C.3's L1+L2 emission so the prompt matches
                    # what the actual file build will produce.
                    from table_adapter import _canonicalize_col as _canon_col
                    identity_col = _canon_col(lift["identity_column"])
                    lift_name = lift["lift_name"]
                    col_lit = _to_string_literal(ev["property_id"])
                    identity_col_lit = _to_string_literal(identity_col)
                    name_lit = _to_string_literal(ev["entity"])
                    val_paren = _ensure_paren(ev["lean_value"])
                    l1_pred = "MaxCell" if operation == "argmax" else "MinCell"
                    l2_pred = "ArgmaxWhere" if operation == "argmax" else "ArgminWhere"
                    # Use the actual Cond from the evidence, not a
                    # hardcoded Cond.all; otherwise the solver-facing
                    # skeleton diverges from what _build_lean_file
                    # emits for conditional argmax.
                    cond_term = ev.get("lean_cond", "Cond.all").strip()
                    if " " in cond_term and not (cond_term.startswith("(") and cond_term.endswith(")")):
                        cond_term_paren = f"({cond_term})"
                    else:
                        cond_term_paren = cond_term
                    cond_term_lit = _lean_strip_guillemets(cond_term_paren)
                    parts.append(f"-- obs{i}: STATS {operation} {_safe_comment_text(ev.get('property', '?'))} = {_safe_comment_text(ev['entity'])}={_safe_comment_text(ev['raw_value'])} (row {ev.get('row_idx', '?')})")
                    parts.append(f"-- L1+L2 path via per-table lift `{lift_name}`")
                    parts.append(f"axiom obs{i}_row : RowHandle")
                    parts.append(f"def obs{i}_val : Value := {ev['lean_value']}")
                    parts.append(f"axiom obs{i}_max : Evidence Grade.Verified ({l1_pred} {col_lit} {cond_term_lit} obs{i}_row obs{i}_val)")
                    parts.append(f"axiom obs{i}_cell : Evidence Grade.Verified (CellAt obs{i}_row {identity_col_lit} (Value.str {name_lit}))")
                    parts.append(f"noncomputable def obs{i}_l2 : Evidence Grade.Verified ({l2_pred} {col_lit} {cond_term_lit} {name_lit} obs{i}_val) :=")
                    parts.append(f"  {lift_name} {col_lit} {cond_term_lit} obs{i}_row {name_lit} obs{i}_val obs{i}_max obs{i}_cell")
                    parts.append(f"-- Use: `refine ⟨{name_lit}, obs{i}_val, ?_⟩; exact obs{i}_l2`")
                else:
                    # Check if this stats op has a value-only L1+L2 lift in the
                    # per-table menu — if so, mirror what _build_lean_file emits
                    # (obs_l1 + obs_l2) instead of legacy obs_shape.
                    op = ev.get("operation", "")
                    vo_lift = _find_matching_lift(tf_lifts, op, ev.get("tool_instance_id"))
                    TOPK_OPS = ("topk_sum", "topk_mean", "bottomk_sum", "bottomk_mean")
                    TOPK_BY_OPS = ("topk_sum_by", "topk_mean_by")
                    is_value_only = (vo_lift is not None
                                     and not vo_lift.get("identity_column")
                                     and op in ("mean", "sum", "count", "max", "min",
                                                "count_distinct", "count_where"))
                    is_topk = (vo_lift is not None
                               and not vo_lift.get("identity_column")
                               and op in TOPK_OPS)
                    is_topk_by = (vo_lift is not None
                                  and not vo_lift.get("identity_column")
                                  and op in TOPK_BY_OPS)
                    if is_topk_by:
                        # topk_*_by L1+L2 mirror.
                        col_id = ev.get("column_id", _to_lean_id(ev.get("property", "")))
                        col_lit = _to_string_literal(col_id)
                        rank_by = ev.get("rank_by") or ""
                        rank_lit = _to_string_literal(_to_lean_id(rank_by)) if rank_by else '""'
                        cond_term = ev.get("lean_cond", "Cond.all").strip()
                        if " " in cond_term and not (cond_term.startswith("(") and cond_term.endswith(")")):
                            cond_term_paren = f"({cond_term})"
                        else:
                            cond_term_paren = cond_term
                        cond_term_lit = _lean_strip_guillemets(cond_term_paren)
                        l1_pred = {"topk_sum_by": "TopKSumByCells",
                                   "topk_mean_by": "TopKMeanByCells"}[op]
                        l2_pred = {"topk_sum_by": "TopKSumByWhere",
                                   "topk_mean_by": "TopKMeanByWhere"}[op]
                        lift_name = vo_lift["lift_name"]
                        k_int = int(ev.get("n") or 5)
                        parts.append(f"-- obs{i}: STATS {op} target={_safe_comment_text(col_id)} rank_by={_safe_comment_text(rank_by)} = {_safe_comment_text(ev['raw_value'])} (k={k_int})")
                        parts.append(f"-- L1+L2 path via per-table lift `{lift_name}` (parametric target+rank+k+Cond)")
                        parts.append(f"def obs{i}_val : Value := {ev['lean_value']}")
                        parts.append(f"axiom obs{i}_l1 : Evidence Grade.Verified ({l1_pred} {col_lit} {rank_lit} {k_int} {cond_term_lit} obs{i}_val)")
                        parts.append(f"noncomputable def obs{i}_l2 : Evidence Grade.Verified ({l2_pred} {col_lit} {rank_lit} {k_int} {cond_term_lit} obs{i}_val) :=")
                        parts.append(f"  {lift_name} {col_lit} {rank_lit} {k_int} {cond_term_lit} obs{i}_val obs{i}_l1")
                        parts.append(f"-- Use: `refine ⟨obs{i}_val, ?_⟩; exact obs{i}_l2`")
                    elif is_value_only or is_topk:
                        col_id = ev.get("column_id", _to_lean_id(ev.get("property", "")))
                        col_lit = _to_string_literal(col_id)
                        cond_term = ev.get("lean_cond", "Cond.all").strip()
                        if " " in cond_term and not (cond_term.startswith("(") and cond_term.endswith(")")):
                            cond_term_paren = f"({cond_term})"
                        else:
                            cond_term_paren = cond_term
                        cond_term_lit = _lean_strip_guillemets(cond_term_paren)
                        if is_value_only:
                            l1_pred = {"mean":"MeanCells","sum":"SumCells","count":"CountCells",
                                       "max":"MaxCellValue","min":"MinCellValue",
                                       "count_distinct":"CountDistinctCells",
                                       "count_where":"CountCells"}[op]
                            l2_pred = {"mean":"MeanWhere","sum":"SumWhere","count":"CountWhere",
                                       "max":"MaxWhere","min":"MinWhere",
                                       "count_distinct":"CountDistinctWhere",
                                       "count_where":"CountWhere"}[op]
                            lift_name = vo_lift["lift_name"]
                            parts.append(f"-- obs{i}: STATS {op} {_safe_comment_text(ev.get('property', '?'))} = {_safe_comment_text(ev['raw_value'])}")
                            parts.append(f"-- L1+L2 path via per-table lift `{lift_name}` (parametric Cond)")
                            parts.append(f"def obs{i}_val : Value := {ev['lean_value']}")
                            parts.append(f"axiom obs{i}_l1 : Evidence Grade.Verified ({l1_pred} {col_lit} {cond_term_lit} obs{i}_val)")
                            parts.append(f"noncomputable def obs{i}_l2 : Evidence Grade.Verified ({l2_pred} {col_lit} {cond_term_lit} obs{i}_val) :=")
                            parts.append(f"  {lift_name} {cond_term_lit} obs{i}_val obs{i}_l1")
                            parts.append(f"-- Use: `refine ⟨obs{i}_val, ?_⟩; exact obs{i}_l2`")
                        else:
                            # topk_* / bottomk_* L1+L2 mirror.
                            l1_pred = {"topk_sum":"TopKSumCells","topk_mean":"TopKMeanCells",
                                       "bottomk_sum":"BottomKSumCells",
                                       "bottomk_mean":"BottomKMeanCells"}[op]
                            l2_pred = {"topk_sum":"TopKSumWhere","topk_mean":"TopKMeanWhere",
                                       "bottomk_sum":"BottomKSumWhere",
                                       "bottomk_mean":"BottomKMeanWhere"}[op]
                            lift_name = vo_lift["lift_name"]
                            k_int = int(ev.get("n") or 5)
                            parts.append(f"-- obs{i}: STATS {op} {_safe_comment_text(ev.get('property', '?'))} = {_safe_comment_text(ev['raw_value'])} (k={k_int})")
                            parts.append(f"-- L1+L2 path via per-table lift `{lift_name}` (parametric k+Cond)")
                            parts.append(f"def obs{i}_val : Value := {ev['lean_value']}")
                            parts.append(f"axiom obs{i}_l1 : Evidence Grade.Verified ({l1_pred} {col_lit} {k_int} {cond_term_lit} obs{i}_val)")
                            parts.append(f"noncomputable def obs{i}_l2 : Evidence Grade.Verified ({l2_pred} {col_lit} {k_int} {cond_term_lit} obs{i}_val) :=")
                            parts.append(f"  {lift_name} {k_int} {cond_term_lit} obs{i}_val obs{i}_l1")
                            parts.append(f"-- Use: `refine ⟨obs{i}_val, ?_⟩; exact obs{i}_l2`")
                    else:
                        tool = "statsTool"
                        wprop_lit = _lean_strip_guillemets(ev['wprop'])
                        parts.append(f"-- obs{i}: STATS {_safe_comment_text(ev.get('property', '?'))} = {_safe_comment_text(ev['raw_value'])}")
                        parts.append(f"axiom obs{i}_q : Query_T {tool}")
                        parts.append(f"axiom obs{i}_att : Attested_T {tool} obs{i}_q obs{i}_p")
                        parts.append(f"axiom obs{i}_tag : ClaimTag_T {tool} obs{i}_q obs{i}_p")
                        parts.append(f"def obs{i}_val : Value := {ev['lean_value']}")
                        parts.append(f"axiom obs{i}_shape : claimWProp obs{i}_tag = {wprop_lit}")
            else:
                # cell_lookup -> L1+L2 dispatch via per-table lift.
                # Mirror _build_lean_file's emission so the prompt
                # matches the actual generated file.
                entity_raw = ev['entity']
                property_raw = ev['property']
                entity_lit = _to_string_literal(entity_raw)
                property_lit = _to_string_literal(property_raw)
                cl_lift = next(
                    (l for l in tf_lifts
                     if l.get("operation") == "cell_lookup"
                     and (l.get("applies_to_tool_instance") in ("*", None)
                          or l.get("applies_to_tool_instance") == ev.get("tool_instance_id"))),
                    None,
                )
                if cl_lift and cl_lift.get("identity_column"):
                    from table_adapter import _canonicalize_col as _canon_col2
                    id_col_canon = _canon_col2(cl_lift["identity_column"])
                    id_col_lit = _to_string_literal(id_col_canon)
                    lift_name = cl_lift["lift_name"]
                    parts.append(f"-- obs{i}: \"{_safe_comment_text(entity_raw)}\".\"{_safe_comment_text(property_raw)}\" = \"{_safe_comment_text(ev['raw_value'])}\"")
                    parts.append(f"-- L1+L2 path via per-table cell_lookup lift `{lift_name}`")
                    parts.append(f"axiom obs{i}_row : RowHandle")
                    parts.append(f"def obs{i}_val : Value := {ev['lean_value']}")
                    parts.append(f"axiom obs{i}_cell : Evidence Grade.Verified (CellAt obs{i}_row {property_lit} obs{i}_val)")
                    parts.append(f"axiom obs{i}_id : Evidence Grade.Verified (CellAt obs{i}_row {id_col_lit} (Value.str {entity_lit}))")
                    parts.append(f"noncomputable def obs{i}_l2 : Evidence Grade.Verified (HasProperty {entity_lit} {property_lit} obs{i}_val) :=")
                    parts.append(f"  {lift_name} {property_lit} obs{i}_row {entity_lit} obs{i}_val obs{i}_cell obs{i}_id")
                    parts.append(f"-- Use: `exact obs{i}_l2` to discharge HasProperty {entity_lit} {property_lit} obs{i}_val")
                else:
                    parts.append(f"-- obs{i}: \"{_safe_comment_text(entity_raw)}\".\"{_safe_comment_text(property_raw)}\" = \"{_safe_comment_text(ev['raw_value'])}\"")
                    parts.append(f"axiom obs{i}_q : Query_T tableTool")
                    parts.append(f"axiom obs{i}_att : Attested_T tableTool obs{i}_q obs{i}_p")
                    parts.append(f"axiom obs{i}_tag : ClaimTag_T tableTool obs{i}_q obs{i}_p")
                    parts.append(f"def obs{i}_val : Value := {ev['lean_value']}")
                    parts.append(f"axiom obs{i}_shape : claimWProp obs{i}_tag = HasProperty {entity_lit} {property_lit} obs{i}_val")
            parts.append("")
        parts.append("-- YOUR PROOF GOES HERE (as proof_body in your JSON)")
        parts.append("```")
        parts.append("")

    if steps:
        # No truncation — the solver needs full feedback to self-correct.
        parts.append("HISTORY (all actions):")
        for s in steps:
            fb = s.get('feedback', '')
            parts.append(f"  {s.get('action', '?')}: {fb}")
        parts.append("")

    parts.append("Propose your next action as JSON.")
    return "\n".join(parts)


def _align_goal_type(lean_goal_type: str, evidence: list[dict], formalizer_output: dict) -> str:
    """Align property/entity names in lean_goal_type to match actual evidence names.

    The formalizer invents property names (e.g., total_goals) that may not match
    the auto-generated evidence property names (e.g., sum_goals). This function
    substitutes the formalizer's names with the actual evidence names using the
    formalizer's properties_needed/entities_needed mappings.
    """
    import re

    def norm(s):
        """Normalize for comparison: lowercase, spaces→underscores."""
        return s.lower().strip().replace(" ", "_")

    # Collect all entity_ids and property_ids from evidence
    ev_ent_ids = {ev["entity_id"] for ev in evidence}
    ev_prop_ids = {ev["property_id"] for ev in evidence}

    # Build mapping: formalizer_name → column it maps to
    prop_map = {}
    for p in formalizer_output.get("properties_needed", []):
        name = p.get("name", "")
        col = p.get("maps_to_column", "")
        if name and col:
            prop_map[name] = norm(col)

    ent_map = {}
    for e in formalizer_output.get("entities_needed", []):
        name = e.get("name", "")
        key = e.get("lookup_key", "")
        if name and key:
            ent_map[name] = norm(key)

    result = lean_goal_type

    # Substitute property names
    for formal_name, col_norm in prop_map.items():
        if formal_name in ev_prop_ids:
            continue  # already matches, no substitution needed
        matched_prop_id = None
        for ev in evidence:
            ev_prop_norm = norm(ev.get("property", ""))
            # Match strategies:
            # 1. Exact match after normalization
            # 2. Column name appears in evidence property (stats: "sum_tropical_cyclones" contains "tropical_cyclones")
            # 3. Evidence property ends with column name (stats prefix)
            if ev_prop_norm == col_norm:
                matched_prop_id = ev["property_id"]
                break
            if col_norm in ev_prop_norm or ev_prop_norm.endswith("_" + col_norm):
                matched_prop_id = ev["property_id"]
                break
        if matched_prop_id and matched_prop_id != formal_name:
            result = re.sub(r'\b' + re.escape(formal_name) + r'\b', matched_prop_id, result)

    # Substitute entity names
    for formal_name, key_norm in ent_map.items():
        if formal_name in ev_ent_ids:
            continue
        for ev in evidence:
            ev_ent_norm = norm(ev.get("entity", ""))
            if ev_ent_norm == key_norm:
                if ev["entity_id"] != formal_name:
                    result = re.sub(r'\b' + re.escape(formal_name) + r'\b', ev["entity_id"], result)
                break

    # Special case: formalizer might use "table_summary"/"stats_summary"/etc. for stats entity
    # The stats handler always uses "table_stats"
    if "table_stats" in ev_ent_ids:
        for variant in ["table_summary", "stats_summary", "summary_stats", "aggregate_stats"]:
            if variant in result and variant not in ev_ent_ids:
                result = re.sub(r'\b' + re.escape(variant) + r'\b', "table_stats", result)

    return result


def _build_lean_file(claim, table_id, formal_goal, claim_holds, evidence, proof_body,
                     lean_goal_type: str = "", table_formalization=""):
    """Build the full .lean file. The PIPELINE controls the type signature.
    The solver provides ONLY the tactic proof (after := by).

    Trust model:
    - The type comes from the formalizer's lean_goal_type (validated to compile)
    - The pipeline generates observation axioms (from trusted tool attestations)
    - ``table_formalization`` is the audited per-table formalization. Per
      specs/runtime-architecture.md §5, accepts two forms:
        (a) a bare string — legacy single-blob form; injected verbatim.
        (b) a dict {"interpretation", "interpretation_level", "lifts": [...]}
            — structured form with per-lift metadata. The combined
            lean_source is emitted; the lift metadata is also threaded
            forward (Phase C.3 will use it for runtime L1+L2 emission).
      Trust boundary: per-table curation, per specs/table-formalization.md §5.
    - The solver writes only the tactic block — cannot alter what is being proved
    - If lean_goal_type is empty, falls back to solver-specified type (legacy mode)
    """
    tf_lean_source, tf_lifts = _normalize_table_formalization(table_formalization)
    lines = [
        "import EGVar.Basic",
        "import EGVar.WorldOntology\nimport EGVar.TableL1\nimport EGVar.Tactics",
        "set_option autoImplicit false",
        "",
        f"-- Claim: {claim}",
        f"-- Goal: {formal_goal}",
        f"-- Verdict attempt: {'VERIFIED' if claim_holds else 'REFUTED'}",
        "",
        "axiom tableTool : ToolId",
        "axiom statsTool : ToolId",
        "",
    ]

    # Per R5/R10 (specs/phase-a-resolutions.md): Entity and Property are
    # no longer types in the kernel. Role-slot names are emitted as String
    # literals at the obs_shape / HasProperty / per-table-lift sites. No
    # global Entity/Property axioms are needed.

    # --- PER-TABLE FORMALIZATION (audited axioms/lemmas for this table)
    # Injected verbatim; trust boundary is curator audit. See
    # specs/table-formalization.md and specs/runtime-architecture.md §1.2.
    if tf_lean_source:
        lines.append("-- === PER-TABLE FORMALIZATION (audited per-table axioms) ===")
        lines.append(tf_lean_source.rstrip())
        lines.append("")

    # Emit observation constants (tool-attested, stats, and common-sense)
    for i, ev in enumerate(evidence):
        if ev.get("source") == "common_sense":
            entity_lit = _to_string_literal(ev['entity_id'])
            property_lit = _to_string_literal(ev['property_id'])
            wprop = _lean_strip_guillemets(
                ev.get("wprop",
                       f"HasProperty {entity_lit} {property_lit} {ev['lean_value']}"))
            lines.extend([
                f"-- cs{i}: COMMON SENSE (Plausible): {_safe_comment_text(ev['raw_value'])}",
                f"axiom cs{i}_q : CSQuery",
                f"axiom cs{i}_att : CSAttestation cs{i}_q ({wprop})",
                f"-- Grade: Plausible (not tool-attested)",
                "",
            ])
        elif ev.get("source") == "stats":
            # Phase C/D dispatch (specs/runtime-architecture.md §2 step 5,
            # specs/tier1-fixture-spec.md §2): if this stats evidence's
            # operation matches a per-table lift in tf_lifts, take the
            # L1+L2 path: emit appropriate L1 attestations and an L2 def
            # composing them via the lift. Suppress the legacy obs_shape
            # emission for this entry (R3 demotion).
            #
            # Two flavors:
            #  - argmax/argmin with identity_column (Phase C, entity-anchored):
            #    emit RowHandle + MaxCell/MinCell + CellAt + L2 ArgmaxWhere/ArgminWhere
            #  - mean/sum/count/max/min (Phase D, value-only): emit a single
            #    L1 attestation (MeanCells/SumCells/etc.) + L2 def via lift.
            operation = ev.get("operation", "")
            lift = _find_matching_lift(tf_lifts, operation, ev.get("tool_instance_id"))
            if (lift and lift.get("identity_column")
                    and operation in ("argmax", "argmin")):
                from table_adapter import _canonicalize_col
                identity_col = _canonicalize_col(lift["identity_column"])
                lift_name = lift["lift_name"]
                col_lit = _to_string_literal(ev["property_id"])
                identity_col_lit = _to_string_literal(identity_col)
                name_lit = _to_string_literal(ev["entity"])
                val_paren = _ensure_paren(ev["lean_value"])
                l1_pred = "MaxCell" if operation == "argmax" else "MinCell"
                l2_pred = "ArgmaxWhere" if operation == "argmax" else "ArgminWhere"
                # Phase D entity-anchored dispatch — parametric over Cond.
                cond_term = ev.get("lean_cond", "Cond.all").strip()
                if " " in cond_term and not (cond_term.startswith("(") and cond_term.endswith(")")):
                    cond_term_paren = f"({cond_term})"
                else:
                    cond_term_paren = cond_term
                cond_term_lit = _lean_strip_guillemets(cond_term_paren)
                lines.extend([
                    f"-- obs{i}: STATS {operation} {_safe_comment_text(ev.get('property', '?'))} = {_safe_comment_text(ev['entity'])}={_safe_comment_text(ev['raw_value'])} (row {ev.get('row_idx', '?')})",
                    f"-- L1+L2 path via per-table lift `{lift_name}` (R10/Phase C/D, parametric Cond)",
                    f"axiom obs{i}_row : RowHandle",
                    f"def obs{i}_val : Value := {ev['lean_value']}",
                    f"axiom obs{i}_max : Evidence Grade.Verified ({l1_pred} {col_lit} {cond_term_lit} obs{i}_row obs{i}_val)",
                    f"axiom obs{i}_cell : Evidence Grade.Verified (CellAt obs{i}_row {identity_col_lit} (Value.str {name_lit}))",
                    f"noncomputable def obs{i}_l2 : Evidence Grade.Verified ({l2_pred} {col_lit} {cond_term_lit} {name_lit} obs{i}_val) :=",
                    f"  {lift_name} {col_lit} {cond_term_lit} obs{i}_row {name_lit} obs{i}_val obs{i}_max obs{i}_cell",
                    "",
                ])
            elif lift and (operation in ("mean", "sum", "count", "max", "min", "count_distinct")
                           or operation == "count_where"):
                # Phase D value-only L1+L2 path. Lift signature expected
                # to be PARAMETRIC OVER COND (per the canonical curator
                # template in specs/d-e-curation-prompts.md):
                #   axiom <lift_name> (c : Cond) (v : Value) :
                #     Evidence Grade.Verified (<L1Pred> "<col>" c v) →
                #     Evidence Grade.Verified (<L2Pred> "<col>" c v)
                # The runtime threads the actual Cond from the action
                # through to both the L1 axiom and the L2 def application.
                lift_name = lift["lift_name"]
                # Use ev["column_id"] (raw column id) not ev["property_id"]
                # (which is "mean_<col>" for value-only ops).
                col_lit = _to_string_literal(ev.get("column_id") or ev.get("property_id", ""))
                _l1_map = {"mean": "MeanCells", "sum": "SumCells",
                           "count": "CountCells", "count_where": "CountCells",
                           "max": "MaxCellValue", "min": "MinCellValue",
                           "count_distinct": "CountDistinctCells"}
                _l2_map = {"mean": "MeanWhere", "sum": "SumWhere",
                           "count": "CountWhere", "count_where": "CountWhere",
                           "max": "MaxWhere", "min": "MinWhere",
                           "count_distinct": "CountDistinctWhere"}
                l1_pred = _l1_map[operation]
                l2_pred = _l2_map[operation]
                # Cond from evidence (threaded through from solver action);
                # parenthesize for safe argument position.
                cond_term = ev.get("lean_cond", "Cond.all").strip()
                if " " in cond_term and not (cond_term.startswith("(") and cond_term.endswith(")")):
                    cond_term_paren = f"({cond_term})"
                else:
                    cond_term_paren = cond_term
                # Strip guillemets in cond (column references inside Cond
                # were emitted via _to_lean_id; we need string literals).
                cond_term_lit = _lean_strip_guillemets(cond_term_paren)
                lines.extend([
                    f"-- obs{i}: STATS {operation} {_safe_comment_text(ev.get('property', '?'))} = {_safe_comment_text(ev['raw_value'])}",
                    f"-- L1+L2 path via per-table lift `{lift_name}` (value-only, parametric Cond)",
                    f"def obs{i}_val : Value := {ev['lean_value']}",
                    f"axiom obs{i}_l1 : Evidence Grade.Verified ({l1_pred} {col_lit} {cond_term_lit} obs{i}_val)",
                    f"noncomputable def obs{i}_l2 : Evidence Grade.Verified ({l2_pred} {col_lit} {cond_term_lit} obs{i}_val) :=",
                    f"  {lift_name} {cond_term_lit} obs{i}_val obs{i}_l1",
                    "",
                ])
            elif lift and operation in ("topk_sum_by", "topk_mean_by"):
                # topk_*_by L1+L2 path. Lift signature (parametric
                # over target, rank, k, Cond, value):
                #   axiom <lift_name> (target : String) (rank : String)
                #                      (k : Nat) (c : Cond) (v : Value) :
                #     Evidence Verified (<L1Pred> target rank k c v) ->
                #     Evidence Verified (<L2Pred> target rank k c v)
                # One lift per BY variant per table; applies to any
                # (target, rank).
                lift_name = lift["lift_name"]
                target_lit = _to_string_literal(ev.get("column_id") or ev.get("property_id", ""))
                rank_by = ev.get("rank_by") or ""
                rank_lit = _to_string_literal(_to_lean_id(rank_by)) if rank_by else '""'
                _byby_l1 = {"topk_sum_by": "TopKSumByCells",
                            "topk_mean_by": "TopKMeanByCells"}
                _byby_l2 = {"topk_sum_by": "TopKSumByWhere",
                            "topk_mean_by": "TopKMeanByWhere"}
                l1_pred = _byby_l1[operation]
                l2_pred = _byby_l2[operation]
                k_int = int(ev.get("n") or 5)
                cond_term = ev.get("lean_cond", "Cond.all").strip()
                if " " in cond_term and not (cond_term.startswith("(") and cond_term.endswith(")")):
                    cond_term_paren = f"({cond_term})"
                else:
                    cond_term_paren = cond_term
                cond_term_lit = _lean_strip_guillemets(cond_term_paren)
                lines.extend([
                    f"-- obs{i}: STATS {operation} target={_safe_comment_text(ev.get('property', '?'))} rank_by={_safe_comment_text(rank_by)} = {_safe_comment_text(ev['raw_value'])} (k={k_int})",
                    f"-- L1+L2 path via per-table lift `{lift_name}` (parametric target+rank+k+Cond)",
                    f"def obs{i}_val : Value := {ev['lean_value']}",
                    f"axiom obs{i}_l1 : Evidence Grade.Verified ({l1_pred} {target_lit} {rank_lit} {k_int} {cond_term_lit} obs{i}_val)",
                    f"noncomputable def obs{i}_l2 : Evidence Grade.Verified ({l2_pred} {target_lit} {rank_lit} {k_int} {cond_term_lit} obs{i}_val) :=",
                    f"  {lift_name} {target_lit} {rank_lit} {k_int} {cond_term_lit} obs{i}_val obs{i}_l1",
                    "",
                ])
            elif lift and operation in ("topk_sum", "topk_mean",
                                         "bottomk_sum", "bottomk_mean"):
                # topk_* / bottomk_* L1+L2 path. Lift signature
                # (parametric over k, Cond, v):
                #   axiom <lift_name> (k : Nat) (c : Cond) (v : Value) :
                #     Evidence Grade.Verified (<L1Pred> "<col>" k c v) ->
                #     Evidence Grade.Verified (<L2Pred> "<col>" k c v)
                lift_name = lift["lift_name"]
                col_lit = _to_string_literal(ev.get("column_id") or ev.get("property_id", ""))
                _topk_l1 = {"topk_sum": "TopKSumCells",
                            "topk_mean": "TopKMeanCells",
                            "bottomk_sum": "BottomKSumCells",
                            "bottomk_mean": "BottomKMeanCells"}
                _topk_l2 = {"topk_sum": "TopKSumWhere",
                            "topk_mean": "TopKMeanWhere",
                            "bottomk_sum": "BottomKSumWhere",
                            "bottomk_mean": "BottomKMeanWhere"}
                l1_pred = _topk_l1[operation]
                l2_pred = _topk_l2[operation]
                k_int = int(ev.get("n") or 5)
                cond_term = ev.get("lean_cond", "Cond.all").strip()
                if " " in cond_term and not (cond_term.startswith("(") and cond_term.endswith(")")):
                    cond_term_paren = f"({cond_term})"
                else:
                    cond_term_paren = cond_term
                cond_term_lit = _lean_strip_guillemets(cond_term_paren)
                lines.extend([
                    f"-- obs{i}: STATS {operation} {_safe_comment_text(ev.get('property', '?'))} = {_safe_comment_text(ev['raw_value'])} (k={k_int})",
                    f"-- L1+L2 path via per-table lift `{lift_name}` (parametric k+Cond)",
                    f"def obs{i}_val : Value := {ev['lean_value']}",
                    f"axiom obs{i}_l1 : Evidence Grade.Verified ({l1_pred} {col_lit} {k_int} {cond_term_lit} obs{i}_val)",
                    f"noncomputable def obs{i}_l2 : Evidence Grade.Verified ({l2_pred} {col_lit} {k_int} {cond_term_lit} obs{i}_val) :=",
                    f"  {lift_name} {k_int} {cond_term_lit} obs{i}_val obs{i}_l1",
                    "",
                ])
            else:
                # Legacy L2 obs_shape path (no matching lift).
                # Sweep guillemets to string literals.
                wprop = _lean_strip_guillemets(ev["wprop"])
                lines.extend([
                    f"-- obs{i}: STATS {_safe_comment_text(ev.get('property', '?'))} = {_safe_comment_text(ev['raw_value'])}",
                    f"axiom obs{i}_q : Query_T statsTool",
                    f"axiom obs{i}_p : Payload_T statsTool obs{i}_q",
                    f"axiom obs{i}_att : Attested_T statsTool obs{i}_q obs{i}_p",
                    f"axiom obs{i}_tag : ClaimTag_T statsTool obs{i}_q obs{i}_p",
                    f"def obs{i}_val : Value := {ev['lean_value']}",
                    f"axiom obs{i}_shape : claimWProp obs{i}_tag = {wprop}",
                    "",
                ])
        else:
            # cell_lookup -> L1+L2 dispatch via per-table lift. If
            # the per-table menu has a cell_lookup lift, emit the
            # uniform L1+L2 form (axiom row + axiom cell + axiom
            # id_cell + def l2 := lift app); otherwise fall back to
            # the legacy obs_shape redex.
            entity_raw = ev['entity']
            property_raw = ev['property']
            entity_lit = _to_string_literal(entity_raw)
            property_lit = _to_string_literal(property_raw)
            cl_lift = next(
                (l for l in tf_lifts
                 if l.get("operation") == "cell_lookup"
                 and (l.get("applies_to_tool_instance") in ("*", None)
                      or l.get("applies_to_tool_instance") == ev.get("tool_instance_id"))),
                None,
            )
            if cl_lift and cl_lift.get("identity_column"):
                from table_adapter import _canonicalize_col as _canon_col
                id_col_canon = _canon_col(cl_lift["identity_column"])
                id_col_lit = _to_string_literal(id_col_canon)
                lift_name = cl_lift["lift_name"]
                lines.extend([
                    f"-- obs{i}: {_safe_comment_text(entity_raw)}.{_safe_comment_text(property_raw)} = {_safe_comment_text(ev['raw_value'])}",
                    f"-- L1+L2 path via per-table cell_lookup lift `{lift_name}`",
                    f"axiom obs{i}_row : RowHandle",
                    f"def obs{i}_val : Value := {ev['lean_value']}",
                    f"axiom obs{i}_cell : Evidence Grade.Verified (CellAt obs{i}_row {property_lit} obs{i}_val)",
                    f"axiom obs{i}_id : Evidence Grade.Verified (CellAt obs{i}_row {id_col_lit} (Value.str {entity_lit}))",
                    f"noncomputable def obs{i}_l2 : Evidence Grade.Verified (HasProperty {entity_lit} {property_lit} obs{i}_val) :=",
                    f"  {lift_name} {property_lit} obs{i}_row {entity_lit} obs{i}_val obs{i}_cell obs{i}_id",
                    "",
                ])
            else:
                lines.extend([
                    f"-- obs{i}: {_safe_comment_text(entity_raw)}.{_safe_comment_text(property_raw)} = {_safe_comment_text(ev['raw_value'])}",
                    f"axiom obs{i}_q : Query_T tableTool",
                    f"axiom obs{i}_p : Payload_T tableTool obs{i}_q",
                    f"axiom obs{i}_att : Attested_T tableTool obs{i}_q obs{i}_p",
                    f"axiom obs{i}_tag : ClaimTag_T tableTool obs{i}_q obs{i}_p",
                    f"def obs{i}_val : Value := {ev['lean_value']}",
                    f"axiom obs{i}_shape : claimWProp obs{i}_tag = HasProperty {entity_lit} {property_lit} obs{i}_val",
                    "",
                ])

    # === TRUSTED TYPE ENFORCEMENT ===
    if lean_goal_type:
        # Pipeline controls the type — solver cannot alter what is being proved.
        # The lean_goal_type was validated by the formalizer feedback loop.
        lines.append("-- TRUSTED: type generated by pipeline from formalizer's lean_goal_type")
        lines.append(f"-- Solver cannot alter this type — only provide the tactic proof.")
        if claim_holds:
            lines.append(f"noncomputable def claim_proof : {lean_goal_type} := by")
        else:
            # For REFUTED: negate the goal. The solver proves the negation.
            # We wrap in a function type: goal → False (i.e., the goal is contradictory)
            lines.append(f"noncomputable def claim_refutation : ({lean_goal_type}) → False := by")
        # Solver provides only the tactic block — indent each line
        for tactic_line in proof_body.split("\n"):
            lines.append(f"  {tactic_line}")
        lines.append("")
        lines.append("-- Axiom audit:")
        if claim_holds:
            lines.append("#print axioms claim_proof")
        else:
            lines.append("#print axioms claim_refutation")
    else:
        # Legacy mode: solver specifies full proof (type + term).
        # Less secure — solver can alter what is being proved.
        lines.append("-- LEGACY: solver specifies full proof (type + term)")
        lines.append("-- WARNING: solver controls the type — not enforced by pipeline")
        lines.append("-- Solver's proof:")
        lines.append(proof_body)
        lines.append("")
        lines.append("-- Axiom audit (last def = main proof):")
        last_name = None
        for line in proof_body.split("\n"):
            if line.strip().startswith("noncomputable def ") or line.strip().startswith("def "):
                last_name = line.split("def ")[1].split(":")[0].split(" ")[0].strip()
        if last_name:
            lines.append(f"#print axioms {last_name}")

    return "\n".join(lines) + "\n"


def _call_ambiguity_auditor(claim: str, llm_call):
    """Call the ambiguity auditor role. Sees the natural-language
    claim only; blind to table, menu, and dictionary. Returns a
    dict with `intent_summary`, `ambiguous_phrases` (list), and
    `confidence`.

    Output is threaded into the formalizer's user prompt and also
    used to force `has_ambiguity=True` at the FormalizationResult
    level when the auditor flagged any phrase, so grade downcast
    fires on multi-reading claims even if the formalizer missed
    the flag.

    Empty / failed output returns a neutral shape so downstream
    code can proceed safely.
    """
    user = f"CLAIM TO AUDIT:\n{claim}\n\nOutput JSON."
    raw = llm_call("ambiguity_auditor", AMBIGUITY_AUDITOR_SYSTEM, user)
    parsed = _parse_json(raw)
    if not parsed or not isinstance(parsed, dict):
        return {"intent_summary": "", "ambiguous_phrases": [], "confidence": "high"}
    return {
        "intent_summary": parsed.get("intent_summary", ""),
        "ambiguous_phrases": parsed.get("ambiguous_phrases", []) or [],
        "confidence": parsed.get("confidence", "high"),
    }


def _call_analyst(table, s, llm_call):
    """Call table analyst. Returns (reading_guide, new_entries_list).

    The analyst produces both a reading guide (prose) and dictionary entries
    (structured). Both are LLM-generated, not hardcoded.
    """
    table_text = _render_table(table)
    user = f"Table ID: {table.table_id}\nCaption: {table.caption}\nColumns ({len(s.columns)}): {', '.join(s.columns)}\n\nFull table:\n{table_text}"
    raw = llm_call("analyst", ANALYST_SYSTEM, user)

    parsed = _parse_json(raw)
    if parsed and parsed.get("reading_guide"):
        guide = parsed["reading_guide"]
        entries = parsed.get("new_entries", [])
        return guide, entries

    # Fallback: treat entire response as reading guide, no structured entries
    return raw, []


def _call_formalizer(claim, d, guide, s, llm_call, extra_prompt_fragment: str = "",
                      extra_system_fragment: str = "", tf_hint: str = "",
                      menu_hint: str = "", table_columns: list[str] | None = None):
    """Call the formalizer with a feedback loop; retries if output
    is invalid.

    ``extra_prompt_fragment`` is appended to ``base_user`` before
    each retry attempt. For variant A this is always ``""``;
    variant F populates it from a rendered ``RefinementSignal``.

    ``extra_system_fragment`` is appended to the system prompt;
    used by prompt-diversity mode to inject an interpretation
    policy per call.

    ``tf_hint`` (anomaly-aware formalizer): per-table structural
    metadata (summary rows, identity column, non-unique columns)
    extracted from the per-table formalizations file. Symmetric
    with the solver, which sees the same content via a Lean
    comment.
    """
    dict_text = d.to_prompt() if d else ""
    base_user = (
        f"DICTIONARY (existing formal terms — reuse these):\n{dict_text}\n\n"
        f"TABLE READING GUIDE:\n{guide}\n\n"
    )
    if menu_hint:
        # Per-table sound-operations menu. Structural fact about
        # which tools are sound on this table; curator pre-computed.
        base_user += f"{menu_hint}\n\n"
    if tf_hint:
        base_user += f"TABLE STRUCTURE (from curated per-table metadata):\n{tf_hint}\n\n"
    base_user += f"CLAIM TO FORMALIZE:\n{claim}\n\n"
    if extra_prompt_fragment:
        base_user = base_user + extra_prompt_fragment + "\n\n"

    system_prompt = FORMALIZER_SYSTEM + extra_system_fragment
    feedback = ""
    for attempt in range(MAX_FORMALIZER_RETRIES):
        user = base_user + feedback + "Output JSON."
        raw = llm_call("formalizer", system_prompt, user)
        parsed = _parse_json(raw)

        if not parsed or not parsed.get("goal"):
            feedback = f"\n\nPREVIOUS ATTEMPT FAILED: Could not parse JSON or missing 'goal'. Raw output: {raw}\nPlease fix and try again.\n\n"
            continue

        if not parsed.get("lean_goal_type"):
            feedback = f"\n\nPREVIOUS ATTEMPT FAILED: Missing 'lean_goal_type'. You must output a valid Lean 4 type expression. Your goal was: {parsed.get('goal','')}\nPlease include lean_goal_type and try again.\n\n"
            continue

        # Quick lean_goal_type compile check. Dictionary entries'
        # types ("Property"/"Entity") are documentation only;
        # names appear as String literals inside the goal type
        # so no per-name declarations are needed.
        lean_type = _lean_strip_guillemets(parsed["lean_goal_type"])
        # Persist the canonical (guillemet-stripped) form back so
        # downstream consumers see Strings, not bare identifiers.
        # Without this, LLM-formalized goals using `«col»` would
        # land in `claim_proof` with `set_option autoImplicit
        # false`, and Lean would reject every solver attempt as
        # "Unknown identifier `«col»`".
        parsed["lean_goal_type"] = lean_type

        # Column-literal alignment: rewrite column-name string
        # literals in lean_goal_type to match the formalizer's own
        # `maps_to_column` (which mirrors the source spelling).
        # Prevents kernel rejection when an LLM normalizes spacing
        # ("july 1, 2013" vs "july 1 , 2013"); the goal and axiom
        # would otherwise be different string literals at the Lean
        # type level and HasProperty would fail to unify.
        try:
            for p in parsed.get("properties_needed", []) or []:
                formalizer_name = (p.get("name", "") or "").strip("«»")
                fixture_col = (p.get("maps_to_column", "") or "").strip("«»")
                if not formalizer_name or not fixture_col:
                    continue
                if formalizer_name == fixture_col:
                    continue
                old_lit = f'"{formalizer_name}"'
                new_lit = f'"{fixture_col}"'
                if old_lit in parsed["lean_goal_type"]:
                    parsed["lean_goal_type"] = parsed["lean_goal_type"].replace(old_lit, new_lit)
        except Exception:
            # Column-literal alignment is best-effort; never fail the
            # formalizer call because of it.
            pass

        # Canonical column-literal enforcement: rewrite any
        # column-name string literal in lean_goal_type that
        # case-insensitively matches a canonical column name in
        # ``table_columns`` to the exact canonical (lowercase) form.
        # Fixes capitalization-mismatch abstains where the LLM
        # preserved the claim's Title Case while runtime-emitted
        # axioms use the canonical lowercase form. Only rewrites;
        # never introduces new literals.
        try:
            if table_columns:
                from table_adapter import _canonicalize_col
                canonical_by_lower = {
                    _canonicalize_col(c): _canonicalize_col(c) for c in table_columns
                }
                import re as _re
                def _canonicalize_literal(m):
                    literal = m.group(1)
                    can = _canonicalize_col(literal)
                    if can in canonical_by_lower and literal != can:
                        return f'"{can}"'
                    return m.group(0)
                parsed["lean_goal_type"] = _re.sub(
                    r'"([^"]+)"', _canonicalize_literal, parsed["lean_goal_type"]
                )
        except Exception:
            pass
        # Refresh lean_type local var for the subsequent typecheck.
        lean_type = parsed["lean_goal_type"]

        decls = ["import EGVar.Basic", "import EGVar.WorldOntology",
                 "import EGVar.TableL1", "import EGVar.Tactics",
                 "set_option autoImplicit true",
                 "axiom tableTool : ToolId", "axiom statsTool : ToolId"]
        decls.append(f"noncomputable def _goal_check := {lean_type}")
        lean_src = "\n".join(decls) + "\n"
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False) as f:
                f.write(lean_src)
                f.flush()
                success, output = _typecheck(f.name)
                if not success:
                    feedback = f"\n\nPREVIOUS ATTEMPT FAILED: lean_goal_type does not compile.\nYour type: {lean_type}\nLean error: {output.strip()}\nPlease fix the Lean type expression and try again.\n\n"
                    continue
        except Exception:
            pass

        return parsed

    # All retries exhausted
    return parsed if parsed else {"goal": "", "error": "formalizer_failed"}


def _merge_entries_to_dict(dictionary, formalizer_output):
    """Append new_entries and enrichments from a formalizer output into the dictionary.

    This is the SOLE mechanism for dictionary growth. Analyst, claim formalizer,
    and evidence formalizer all contribute through the same path.
    """
    for entry_data in formalizer_output.get("new_entries", []):
        try:
            raw_name = entry_data.get("name", "")
            entry_type = entry_data.get("type", "Entity")
            # Normalize names to guillemet Lean identifier form for dictionary
            # internal storage (per `_to_lean_id`'s docstring).
            lean_name = _to_lean_id(raw_name)
            # The `formalization` field is documentation only; no
            # `axiom X : Entity/Property` is emitted. `entry_type`
            # ("Entity"/"Property") stays as a semantic tag while
            # `formalization` just notes the role for the LLM.
            if entry_type == "Property":
                formalization = f"{lean_name} : String  -- column header (per R10 nominalist design)"
            elif entry_type == "Entity":
                formalization = f"{lean_name} : String  -- entity name (per R10 nominalist design)"
            else:
                formalization = entry_data.get("formalization", "")
            aliases = [
                Alias(surface_form=a.get("surface_form", ""),
                      context=a.get("context", ""),
                      source=a.get("source", "formalizer_proposed"))
                for a in entry_data.get("aliases", [])
            ]
            dictionary.add_entry(DictEntry(
                name=lean_name,
                type=entry_type,
                description=entry_data.get("description", ""),
                formalization=formalization,
                aliases=aliases,
            ))
        except Exception:
            pass
    for enrichment in formalizer_output.get("enrichments", []):
        try:
            entry_name = enrichment.get("entry_name", "")
            new_alias = enrichment.get("new_alias", {})
            if entry_name and new_alias:
                dictionary.add_alias(entry_name, Alias(
                    surface_form=new_alias.get("surface_form", ""),
                    context=new_alias.get("context", ""),
                    source=new_alias.get("source", "formalizer_enrichment"),
                ))
        except Exception:
            pass


def _check_consensus(formalizations):
    """Check consensus among formalizer outputs using Lean type equality.

    Strategy: for each pair of goals, check if they are definitionally equal
    as Lean types. If ≥2 agree, use the majority. This is the only reliable
    way to check semantic equivalence — string comparison misses alpha-equivalence,
    and goal_type matching is too coarse.
    """
    valid = [f for f in formalizations if f.get("goal")]
    if len(valid) < 2:
        return None

    # Check pairwise equality via Lean kernel
    n = len(valid)
    # Build equivalence classes
    eq_classes = list(range(n))  # union-find

    def find(i):
        while eq_classes[i] != i:
            eq_classes[i] = eq_classes[eq_classes[i]]
            i = eq_classes[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            # Use lean_goal_type for kernel equality; fall back to goal string
            type_a = valid[i].get("lean_goal_type", valid[i].get("goal", ""))
            type_b = valid[j].get("lean_goal_type", valid[j].get("goal", ""))
            if _goals_equal(type_a, type_b):
                eq_classes[find(j)] = find(i)

    # Find largest equivalence class
    from collections import Counter
    roots = [find(i) for i in range(n)]
    class_counts = Counter(roots)
    best_root, best_count = class_counts.most_common(1)[0]

    has_ambiguity = any(f.get("ambiguous_terms") for f in valid)

    if best_count >= 2:
        winner = valid[best_root]
        agreeing = [valid[i] for i in range(n) if find(i) == best_root]
        winner["consensus_details"] = {
            "agreement": f"{best_count}/{n}",
            "method": "lean_equality" if best_count > 1 else "single",
            "has_ambiguity": has_ambiguity,
            "all_goals": [f.get("goal", "") for f in valid],
        }
        return winner

    return None


def _goals_equal(lean_type_a: str, lean_type_b: str) -> bool:
    """Check if two Lean goal types are definitionally equal via the kernel.

    Fast path: normalized string comparison.
    Slow path: write a .lean file with both types and check `rfl`.
    """
    if not lean_type_a or not lean_type_b:
        return False

    # Fast path
    norm_a = " ".join(lean_type_a.split())
    norm_b = " ".join(lean_type_b.split())
    if norm_a == norm_b:
        return True

    # Slow path: Lean kernel definitional equality
    lean_src = (
        "import EGVar.Basic\nimport EGVar.WorldOntology\nimport EGVar.Tactics\n"
        "set_option autoImplicit true\n\n"
        "-- Check if two goal types are definitionally equal\n"
        f"noncomputable def goal_a := {lean_type_a}\n"
        f"noncomputable def goal_b := {lean_type_b}\n"
        "example : goal_a = goal_b := rfl\n"
    )
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False) as f:
            f.write(lean_src)
            f.flush()
            success, _ = _typecheck(f.name)
            return success
    except Exception:
        return False


def _resolve_lookup_identity(table_id: str) -> str | None:
    """Return the identity_column from the per-table cell_lookup
    lift, or None.

    Used by lookup-action handlers (live pipeline + validator) to
    ensure that entity matching happens against the identity column
    declared by the curator's cell_lookup lift. Without this, the
    L1 premise `CellAt r id_col (Value.str entity)` could be false:
    the solver could match the entity in a non-identity column and
    mint a false storage fact.
    """
    per_table = _load_per_table_formalizations()
    tf = per_table.get(table_id)
    if not isinstance(tf, dict):
        return None
    for l in tf.get("lifts", []):
        if l.get("operation") == "cell_lookup" and l.get("identity_column"):
            return l["identity_column"]
    return None


def _execute_lookup(table, entity, prop, identity_col: str | None = None):
    """Look up `(entity, prop)` in `table`.

    If ``identity_col`` is provided, the entity match is restricted
    to that column only; this is required for soundness when the
    caller plans to emit an L1
    `CellAt r identity_col (Value.str entity)` premise. Without
    the restriction, the L1 axiom could be false if the entity
    matched a different column.

    If ``identity_col`` is None, falls back to the legacy "scan any
    column" behavior used only by the legacy obs_shape emission
    path.
    """
    from table_adapter import _canonicalize_col
    # Strict L1+L2 mode: when identity_col is set, the caller plans
    # to emit `CellAt r identity_col (Value.str entity)` as an L1
    # storage premise. To keep that L1 truthful at the trusted
    # runtime boundary, this branch enforces:
    #   - property matches a table column exactly after
    #     column-canonicalization,
    #   - identity_col matches a table column the same way,
    #   - entity is byte-exactly equal to some row[id_col_idx]
    #     cell (no case-folding, no aliasing, no strip),
    #   - the entity uniquely identifies a row (multi-match fails
    #     closed),
    #   - the returned property name is the table's canonical
    #     column form (not the solver's input).
    # Anything else fails closed.
    if identity_col:
        id_canon = _canonicalize_col(identity_col)
        id_col_idx = None
        for i, c in enumerate(table.columns):
            if _canonicalize_col(c) == id_canon:
                id_col_idx = i
                break
        if id_col_idx is None:
            return None
        prop_canon = _canonicalize_col(prop)
        prop_col_idx = None
        prop_canonical_name = None
        for i, c in enumerate(table.columns):
            if _canonicalize_col(c) == prop_canon:
                prop_col_idx = i
                prop_canonical_name = c
                break
        if prop_col_idx is None:
            return None
        # Exact entity match (NO normalization at all)
        match_row = None
        for r_idx, row in enumerate(table.rows):
            if id_col_idx < len(row) and row[id_col_idx] == entity:
                if match_row is not None:
                    return None  # ambiguous — fail closed
                match_row = r_idx
        if match_row is None:
            return None
        if prop_col_idx >= len(table.rows[match_row]):
            return None
        val = table.rows[match_row][prop_col_idx]
        return {"entity": entity, "property": prop_canonical_name,
                "raw_value": val, "row_idx": match_row,
                "match_col": table.columns[id_col_idx]}

    # Legacy obs_shape path (no L1 storage premise to falsify): alias
    # variants OK. Used when no per-table cell_lookup lift exists.
    variants = [entity]
    if "_" in entity:
        variants.append(entity.replace("___", " - ").replace("_", " "))
    if " - " in entity:
        variants.append(entity.replace(" - ", "___").replace(" ", "_"))
    prop_variants = [prop]
    if "_" in prop:
        prop_variants.append(prop.replace("___", " - ").replace("_", " "))
    if " - " in prop:
        prop_variants.append(prop.replace(" - ", "___").replace(" ", "_"))
    for ent in variants:
        for col in table.columns:
            r = find_rows(table, col, ent)
            if r.success and r.all_matches and len(r.all_matches) == 1:
                row_idx = r.all_matches[0]["row_idx"]
                for p in prop_variants:
                    val = cell_lookup(table, row_idx, p)
                    if val.success:
                        return {"entity": entity, "property": prop, "raw_value": val.value,
                                "row_idx": row_idx, "match_col": col}
    return None


def _to_lean_id(s: str) -> str:
    """Convert a name to a Lean 4 identifier using guillemet
    escaping.

    Used for dictionary internal keys (where guillemet identifiers
    are convenient and unambiguous). For Lean source emission of
    role-slot arguments (entity, property, column, ...), use
    ``_to_string_literal`` instead: L2 predicates take String, not
    identifier-typed atoms.

    Normalizes whitespace (newlines, tabs to single space).
    """
    import re
    clean = re.sub(r'\s+', ' ', s.lower().strip())
    return f'«{clean}»'


def _safe_comment_text(s: object) -> str:
    """Sanitize a string for inclusion in a single-line Lean
    `--` comment.

    Solver-supplied strings (e.g. column headers carrying literal
    `\\n` from the source HTML) can contain embedded newlines that
    would split a single-line comment across multiple lines, with
    the second line parsed as Lean code. This helper collapses all
    whitespace runs to a single space and strips.

    Use for any value derived from solver input
    (ev['property'], ev['entity'], ev['raw_value'], ...) before
    interpolating into a Lean comment.

    Note: this is a comment-text sanitizer only. For column
    literals that go into L2 predicate arguments, use
    ``_canonicalize_col`` + ``_to_string_literal``.
    """
    import re
    if not isinstance(s, str):
        s = str(s)
    # Collapse all whitespace (\n, \r, \t, vertical tabs, etc.) to single space.
    return re.sub(r'\s+', ' ', s).strip()


def _to_string_literal(s: str) -> str:
    """Convert a name (possibly guillemet-wrapped) to a Lean 4
    String literal.

    L2 predicates take String slots, so per-table lifts and
    obs_shape axioms need string literals like ``"females (%)"``,
    not identifier-typed atoms like ``«females (%)»``. Handles both
    bare names and ``_to_lean_id``-style guillemet-wrapped inputs;
    already-quoted strings pass through unchanged.
    """
    if s.startswith('«') and s.endswith('»'):
        s = s[1:-1]
    elif s.startswith('"') and s.endswith('"'):
        return s
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _lean_strip_guillemets(src: str) -> str:
    """Sweep a Lean source fragment, converting any `«name»` to `"name"`.

    Used at emission time when a Lean fragment was constructed earlier
    via `_to_lean_id` (so contains guillemet identifiers) but must be
    consumed under R5/R10's String discipline.
    """
    import re
    def _repl(m: 're.Match[str]') -> str:
        inner = m.group(1)
        return '"' + inner.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return re.sub(r'«([^»]*)»', _repl, src)


def _find_matching_lift(tf_lifts, operation, tool_instance_id=None):
    """Find the per-table lift whose `operation` matches.

    The runtime dispatches on the evidence entry's operation to
    pick which per-table lift to apply for L1->L2 composition.

    When `tool_instance_id` is provided, prefer a lift whose
    `applies_to_tool_instance` matches (disambiguates among
    multiple lifts sharing an operation across different columns).
    Falls back to first-by-operation if no instance-matching lift
    exists.
    """
    if not tf_lifts:
        return None
    if tool_instance_id:
        for lift in tf_lifts:
            if (lift.get("operation") == operation
                    and lift.get("applies_to_tool_instance") == tool_instance_id):
                return lift
    for lift in tf_lifts:
        if lift.get("operation") == operation:
            return lift
    return None


def _ensure_paren(s):
    """Wrap a Lean term in parens if it contains a space and isn't
    already parenthesized. Used at L1/L2 emission sites to keep the
    term safe in argument position."""
    s = (s or "").strip()
    if not s:
        return s
    if s.startswith('(') and s.endswith(')'):
        return s
    if ' ' in s:
        return f'({s})'
    return s


def _summarize_table_tools(tf):
    """Build a compressed solver-facing summary of the per-table
    tool menu.

    Returns a list of prompt lines, grouping the per-table ``tools``
    list by operation family and column-type compatibility:
      - aggregations (mean/sum/max/min) on numeric columns
      - count_where / count_distinct on any column
      - entity-anchored (argmax/argmin) on numeric columns via identity
      - top-k (topk_sum/mean, bottomk_sum/mean) on numeric columns
      - top-k-by (topk_sum_by, topk_mean_by), generic
      - cell_lookup on any column via identity

    Hides library_ref / parse_mode / config_override; the solver
    consults the global prompt for call schemas. Returns [] if
    ``tf`` has no tools.
    """
    if not isinstance(tf, dict):
        return []
    tools = tf.get("tools", [])
    if not tools:
        return []

    def _op_from_ref(ref: str) -> str:
        s = ref or ""
        if s.startswith("stats_"):
            s = s[len("stats_"):]
        for suf in ("_strict", "_leading_int", "_leading_number"):
            if s.endswith(suf):
                s = s[:-len(suf)]
                break
        if s == "argmax_with_id":
            return "argmax"
        if s == "argmin_with_id":
            return "argmin"
        return s

    by_op: dict[str, set[str]] = {}
    identity_col = None
    for t in tools:
        op = _op_from_ref(t.get("library_ref", ""))
        col = t.get("column", "")
        by_op.setdefault(op, set()).add(col)
        if t.get("identity_column") and not identity_col:
            # Canonicalize for display (collapse embedded whitespace).
            identity_col = _safe_comment_text(t["identity_column"])

    NUMERIC_AGG = ["mean", "sum", "max", "min"]
    # count_rows is removed; solvers must use count_where with Cond.all.
    COUNT_ANY = ["count_where", "count_distinct"]
    ENTITY = ["argmax", "argmin"]
    TOPK = ["topk_sum", "topk_mean", "bottomk_sum", "bottomk_mean"]
    TOPK_BY = ["topk_sum_by", "topk_mean_by"]

    def _cols_for(ops):
        # Canonicalize the displayed column names so the solver
        # never sees embedded whitespace from raw HTML headers.
        # Tool resolution canonicalizes both sides; doing the same
        # at display time keeps menu and dispatch consistent.
        s = set()
        for op in ops:
            for c in by_op.get(op, ()):
                if c and c != "*":
                    s.add(_safe_comment_text(c))
        return sorted(s)

    lines = ["TOOL MENU FOR THIS TABLE (call schemas in the global prompt; "
             "the runtime applies per-table parse_mode/identity automatically):"]

    agg_ops_present = [o for o in NUMERIC_AGG if o in by_op]
    agg_cols = _cols_for(NUMERIC_AGG)
    if agg_ops_present and agg_cols:
        lines.append(f"  aggregations ({', '.join(agg_ops_present)}) — NUMERIC columns:")
        lines.append(f"    {agg_cols!r}")

    count_ops_present = [o for o in COUNT_ANY if o in by_op]
    if count_ops_present:
        lines.append(f"  {', '.join(count_ops_present)} — ANY column "
                     f"(filter via JSON `condition` for count_where).")

    ent_ops_present = [o for o in ENTITY if o in by_op]
    ent_cols = _cols_for(ENTITY)
    if ent_ops_present and ent_cols:
        id_part = f" via identity={identity_col!r}" if identity_col else ""
        lines.append(f"  entity-anchored ({', '.join(ent_ops_present)}) on NUMERIC columns{id_part}:")
        lines.append(f"    {ent_cols!r}")
        lines.append(f"    (returns the entity name + value; pair with `entity_column` "
                     f"in your call if needed)")

    topk_ops_present = [o for o in TOPK if o in by_op]
    topk_cols = _cols_for(TOPK)
    if topk_ops_present and topk_cols:
        lines.append(f"  top-k ({', '.join(topk_ops_present)}) on NUMERIC columns "
                     f"(pass integer `n`):")
        lines.append(f"    {topk_cols!r}")

    topk_by_present = [o for o in TOPK_BY if o in by_op]
    if topk_by_present:
        lines.append(f"  top-k-by ({', '.join(topk_by_present)}) — GENERIC: "
                     f"pass BOTH `column` (target) and `rank_by` (ranking col) and `n`.")

    if "cell_lookup" in by_op:
        id_part = f" via identity={identity_col!r}" if identity_col else ""
        lines.append(f"  cell_lookup on ANY column{id_part} "
                     f"(pass `entity` and `property`).")

    lines.append("")
    return lines


def _normalize_table_formalization(tf):
    """Normalize a table_formalization fixture field to
    (lean_source, lifts).

    `tf` may be a bare string (legacy single-blob form) or a
    structured object with `interpretation`,
    `interpretation_level`, and a `lifts` array.

    Returns a tuple:
        - lean_source (str): Lean source fragment injected verbatim
          into the generated file's per-table formalization block.
          For the structured form, includes the curator's
          interpretive commitment as a header comment plus each
          lift's lean_source.
        - lifts (list[dict]): lift metadata for runtime selection.
          Empty if `tf` is a bare string. Each entry has:
              {"lift_name": str, "operation": str,
               "identity_column": Optional[str], "lean_source": str}
    """
    if not tf:
        return "", []
    if isinstance(tf, str):
        # Legacy single-blob form: no metadata, no runtime selection.
        return tf, []
    if isinstance(tf, dict):
        lifts = tf.get("lifts", []) or []
        parts = []
        if tf.get("interpretation"):
            parts.append(f"-- Interpretive commitment: {tf['interpretation']}")
        if tf.get("interpretation_level") is not None:
            parts.append(f"-- Interpretation hierarchy level: {tf['interpretation_level']}")
        if parts:
            parts.append("")
        for lift in lifts:
            src = lift.get("lean_source", "").rstrip()
            if src:
                parts.append(src)
                parts.append("")
        return "\n".join(parts).rstrip(), lifts
    return "", []


EVIDENCE_FORMALIZER_SYSTEM = """\
You formalize a raw table lookup result into a Lean 4 Value term.

You receive: the raw cell value, the entity name, the property name, and the dictionary of existing formal terms.

RULES:
- REUSE entity/property names from the dictionary. Do NOT invent new names if an existing entry matches.
- If the raw value is numeric (integer), encode as: Value.rat (mkRat <number> 1)
- If the raw value is a formula like "70 + 71 + 68 + 72 = 281", extract the result: Value.rat (mkRat 281 1)
- If the raw value is a compound like "american airlines center 20301", decide what the claim needs:
  if the property is about attendance/count, extract the number: Value.rat (mkRat 20301 1)
  if the property is about location/name, use the string: Value.str "american airlines center"
- If the raw value is purely non-numeric text, encode as: Value.str "<value>"
- Entity and property IDs must be valid Lean identifiers. Use «guillemets» for names with special characters.

Output JSON:
{
  "entity_id": "...",
  "property_id": "...",
  "lean_value": "Value.rat (mkRat 281 1) or Value.str \"...\"",
  "reasoning": "...",
  "new_entries": [
    {"name": "...", "type": "Entity", "description": "...", "formalization": "...",
     "aliases": [{"surface_form": "...", "context": "from table", "source": "evidence_formalizer"}]}
  ]
}

IMPORTANT: If this entity is NEW (not in the dictionary), add it as a new_entries Entity.
This ensures the next evidence formalization can reuse the same identifier."""


def _formalize_evidence(entity, prop, lookup, idx, llm_call=None, dictionary=None, reading_guide=""):
    """Formalize a raw lookup result into Lean terms.

    The formalizer (LLM, not hardcoded code) decides the encoding.
    Falls back to deterministic encoding if no LLM is available.
    """
    raw = lookup["raw_value"]

    if llm_call is not None:
        dict_text = dictionary.to_prompt() if dictionary else ""
        user = (
            f"RAW LOOKUP RESULT:\n"
            f"  Entity: \"{entity}\"\n"
            f"  Property (column): \"{prop}\"\n"
            f"  Raw cell value: \"{raw}\"\n\n"
            f"DICTIONARY (existing formal terms — reuse these):\n{dict_text}\n\n"
            f"READING GUIDE:\n{reading_guide}\n\n"
            f"Output JSON."
        )
        resp = llm_call("evidence_formalizer", EVIDENCE_FORMALIZER_SYSTEM, user)
        parsed = _parse_json(resp)
        if parsed and parsed.get("lean_value"):
            ent_id = parsed.get("entity_id", _to_lean_id(entity))
            prop_id = parsed.get("property_id", _to_lean_id(prop))
            lean_val = parsed["lean_value"]

            # Evidence-encoding auditor: deterministic
            # Fraction-based validation of the LLM's
            # Value.rat (mkRat N D) against raw_value. Catches
            # decimal-encoding mistakes (e.g., "44.5" rendered as
            # mkRat 45 10 = 4.5, off by 10x). When the LLM's
            # numeric encoding disagrees with the deterministic
            # Fraction parse of the raw value, override with the
            # Fraction encoding (deterministic arithmetic beats
            # LLM self-report).
            try:
                import re as _re
                from fractions import Fraction as _Fraction
                from table_adapter import _parse_exact as _auditor_parse_exact
                from table_adapter import _fraction_to_lean as _auditor_to_lean
                expected_frac = _auditor_parse_exact(raw) if raw else None
                if expected_frac is not None:
                    m = _re.match(
                        r'Value\.rat\s*\(?\s*mkRat\s+(-?\d+)\s+(\d+)\s*\)?',
                        lean_val.strip(),
                    )
                    if m:
                        llm_frac = _Fraction(int(m.group(1)), int(m.group(2)))
                        if llm_frac != expected_frac:
                            print(
                                f"    [evidence-auditor] MISMATCH for {entity}.{prop} = {raw!r}: "
                                f"LLM encoded {lean_val} (= {float(llm_frac):.4g}) "
                                f"but Fraction parse gives {expected_frac} (= {float(expected_frac):.4g}). "
                                f"Overriding with deterministic encoding.",
                                file=sys.stderr,
                            )
                            lean_val = _auditor_to_lean(expected_frac)
            except Exception:
                # Auditor is best-effort; never fail the formalizer call.
                pass

            result = {"entity": entity, "entity_id": ent_id, "property": prop, "property_id": prop_id,
                      "raw_value": raw, "lean_value": lean_val}
            # Pass through full output for dictionary merging
            if parsed.get("new_entries") or parsed.get("enrichments"):
                result["_formalizer_output"] = parsed
            return result

    # Fallback: deterministic encoding (legacy, no LLM available)
    ent_id = _to_lean_id(entity)
    prop_id = _to_lean_id(prop)
    # Try int, then float, then string
    try:
        num = int(raw)
        lean_val = f"Value.rat (mkRat {num} 1)"
    except (ValueError, TypeError):
        try:
            from fractions import Fraction
            from table_adapter import _parse_exact, _fraction_to_lean
            f = _parse_exact(raw)
            if f is not None:
                lean_val = _fraction_to_lean(f)
            else:
                lean_val = f'Value.str "{raw}"'
        except (ValueError, TypeError):
            lean_val = f'Value.str "{raw}"'
    return {"entity": entity, "entity_id": ent_id, "property": prop, "property_id": prop_id,
            "raw_value": raw, "lean_value": lean_val}


CS_FORMALIZER_SYSTEM = """\
You formalize a common sense / world knowledge fact into a Lean 4 proposition.

You receive: a natural language proposition confirmed by a common sense agent, the dictionary, and the reading guide.

Output JSON:
{
  "wprop": "<Lean WProp expression, e.g., HasProperty entity property value>",
  "entity_id": "...",
  "property_id": "...",
  "lean_value": "Value.str \"...\" or Value.rat (mkRat n d)",
  "reasoning": "..."
}

Use entity/property names from the dictionary. Use «guillemets» for special characters."""


def _formalize_cs_evidence(proposition, idx, llm_call=None, dictionary=None, reading_guide=""):
    """Formalize a common sense fact into Lean terms at Plausible grade."""
    if llm_call is not None:
        dict_text = dictionary.to_prompt() if dictionary else ""
        user = (
            f"COMMON SENSE FACT (confirmed by agent):\n"
            f"  \"{proposition}\"\n\n"
            f"DICTIONARY:\n{dict_text}\n\n"
            f"READING GUIDE:\n{reading_guide}\n\n"
            f"Output JSON."
        )
        resp = llm_call("cs_formalizer", CS_FORMALIZER_SYSTEM, user)
        parsed = _parse_json(resp)
        if parsed and parsed.get("wprop"):
            return {
                "entity": proposition, "entity_id": parsed.get("entity_id", "cs"),
                "property": "common_sense", "property_id": "common_sense",
                "raw_value": proposition, "lean_value": parsed.get("lean_value", ""),
                "wprop": parsed["wprop"],
                "grade": "Plausible", "source": "common_sense",
                "_formalizer_output": parsed,
            }

    # Fallback
    return {
        "entity": proposition, "entity_id": "cs",
        "property": "common_sense", "property_id": "common_sense",
        "raw_value": proposition, "lean_value": f'Value.str "{proposition}"',
        "grade": "Plausible", "source": "common_sense",
    }


NL_RENDERER_SYSTEM = """You render a formal verification result into a clear natural language answer.

You receive: the original claim, the verdict (VERIFIED/REFUTED/ABSTAIN), the formal goal, the evidence collected, and the dictionary.

Write a concise 1-3 sentence answer that:
- States the verdict clearly
- Explains what evidence was found
- Notes any caveats (e.g., ambiguous terms, interpretation choices)

Do NOT include Lean syntax. Write for a non-technical reader."""


def _render_nl_answer(claim, verdict, formal_goal, evidence, dictionary, llm_call):
    """Step 7: fresh LLM renders formal answer to NL."""
    if verdict not in ("VERIFIED", "REFUTED"):
        return "Could not verify this claim."

    if llm_call is None:
        ev_summary = ", ".join(f"{e['entity']}.{e['property']}={e['raw_value']}" for e in evidence)
        return f"{verdict}. Evidence: {ev_summary}."

    ev_lines = []
    for e in evidence:
        ev_lines.append(f"  \"{e['entity']}\".\"{e['property']}\" = \"{e['raw_value']}\"")

    dict_text = dictionary.to_prompt() if dictionary else ""
    user = (
        f"CLAIM: {claim}\n"
        f"VERDICT: {verdict}\n"
        f"FORMAL GOAL: {formal_goal}\n\n"
        f"EVIDENCE:\n" + "\n".join(ev_lines) + "\n\n"
        f"DICTIONARY CONTEXT:\n{dict_text}\n\n"
        f"Write a concise natural language answer."
    )
    try:
        return llm_call("renderer", NL_RENDERER_SYSTEM, user)
    except Exception:
        ev_summary = ", ".join(f"{e['entity']}.{e['property']}={e['raw_value']}" for e in evidence)
        return f"{verdict}. Evidence: {ev_summary}."


def _typecheck(lean_file):
    result = subprocess.run(["lake", "env", "lean", str(lean_file)],
                            cwd=LEAN_DIR, capture_output=True, text=True, timeout=120)
    return result.returncode == 0, result.stdout + result.stderr


def _extract_sigma_witness(proof_body: str) -> str:
    """Extract the outermost Sigma witness from the proof body.

    The solver writes: ``refine ⟨witness, ...⟩`` or ``exact ⟨witness, ...⟩``.
    The first value inside ⟨⟩ is the answer — it is the outermost
    Σ-bound variable of the goal type.

    Examples:
      'refine ⟨Value.rat (mkRat 5500000 1), ?_⟩' → '5500000'
      'refine ⟨Value.rat (mkRat 463 9), ?_⟩' → '463/9'
      'refine ⟨"manipur", Value.rat (mkRat 99 1), ?_⟩' → 'manipur'  (String witness)
      'refine ⟨«manipur», Value.rat (mkRat 99 1), ?_⟩' → 'manipur'  (legacy guillemet)

    Argmax/argmin goals have shape
    `Σ (e : String) (v : Value), ...`; the witness is a bare
    quoted-string literal. Legacy guillemet-wrapped or
    ``Entity.ofName "..."`` shapes are also handled for backward
    compatibility.

    The implementation deliberately does not fall back to scanning
    the whole proof_body for any ``mkRat N D`` (that scan once
    silently turned entity answers into the inner numeric value).
    """
    import re
    # Try two surface forms the solver may emit for the outermost
    # Sigma witness:
    #   (a) angle-bracket anonymous constructor:
    #       `refine ⟨witness, ...⟩` or `exact ⟨witness, ...⟩`
    #   (b) explicit Sigma.mk: `refine Sigma.mk witness ...`
    # Both resolve to the first value after the constructor head.
    m = re.search(r'[⟨<](.+?)[,⟩>]', proof_body)
    if not m:
        # Sigma.mk form: match the first balanced-paren argument.
        m = re.search(
            r'Sigma\.mk\s+(\(\s*Value\.[A-Za-z0-9_]+\s+\([^()]*\)\s*\)|'
            r'\(\s*Value\.[A-Za-z0-9_]+\s+[^()\s]+\s*\)|'
            r'Value\.[A-Za-z0-9_]+\s+\([^()]*\)|'
            r'Value\.[A-Za-z0-9_]+\s+[^()\s]+|'
            r'"[^"]*"|'
            r'\S+)',
            proof_body,
        )
        if not m:
            return ""
        witness_raw = m.group(1).strip()
        # Strip one outer layer of parens if present
        if witness_raw.startswith('(') and witness_raw.endswith(')'):
            witness_raw = witness_raw[1:-1].strip()
        witness = witness_raw
    else:
        witness = m.group(1).strip()
    # Value.int N → N
    m2 = re.search(r'Value\.int\s+(-?\d+)', witness)
    if m2:
        return m2.group(1)
    # Value.rat (mkRat N D) → N/D; also accept parenthesized
    # negative N: (mkRat (-N) D). First check for Rat arithmetic
    # expressions (e.g. `mkRat 4340 10 - mkRat 45 10`) which the
    # single-mkRat regex would otherwise mis-extract as just the
    # first operand. If the witness contains 2+ mkRat tokens,
    # evaluate the expression with Fraction arithmetic before
    # falling back to single-mkRat extraction.
    all_mkrats = re.findall(r'mkRat\s+\(?(-?\d+)\)?\s+(\d+)', witness)
    if len(all_mkrats) >= 2:
        try:
            from fractions import Fraction as _Fraction
            # Replace each mkRat N D with a placeholder F0, F1, ...
            frac_vars = {}
            counter = [0]
            def _stash_mkrat(m):
                i = counter[0]
                counter[0] += 1
                frac_vars[f"F{i}"] = _Fraction(int(m.group(1)), int(m.group(2)))
                return f"F{i}"
            py_expr = re.sub(r'mkRat\s+\(?(-?\d+)\)?\s+(\d+)', _stash_mkrat, witness)
            # Strip Value.rat / Value.int wrappers so only arithmetic + placeholders remain
            py_expr = re.sub(r'Value\.(?:rat|int)\s*', '', py_expr)
            import ast as _ast
            tree = _ast.parse(py_expr.strip(), mode='eval')
            def _safe_eval_frac(node):
                if isinstance(node, _ast.Expression):
                    return _safe_eval_frac(node.body)
                if isinstance(node, _ast.Name) and node.id in frac_vars:
                    return frac_vars[node.id]
                if isinstance(node, _ast.Constant) and isinstance(node.value, int):
                    return _Fraction(node.value)
                if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.USub):
                    return -_safe_eval_frac(node.operand)
                if isinstance(node, _ast.BinOp) and isinstance(node.op, (_ast.Add, _ast.Sub, _ast.Mult, _ast.Div)):
                    l, r = _safe_eval_frac(node.left), _safe_eval_frac(node.right)
                    if isinstance(node.op, _ast.Add): return l + r
                    if isinstance(node.op, _ast.Sub): return l - r
                    if isinstance(node.op, _ast.Mult): return l * r
                    return l / r
                raise ValueError(f"disallowed AST node: {type(node).__name__}")
            result = _safe_eval_frac(tree)
            if isinstance(result, _Fraction):
                return f"{result.numerator}/{result.denominator}"
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError):
            pass  # fall through to single-mkRat extraction
    m3 = re.search(r'mkRat\s+\(?(-?\d+)\)?\s+(\d+)', witness)
    if m3:
        return f"{m3.group(1)}/{m3.group(2)}"
    # Value.rat (mkRat <arith expr> D): evaluate the arithmetic
    # expression if it is a purely-integer, +/-/* AST (e.g.
    # '(235072 - 9197)'); emit N/D. Whitelist-only AST: no /, no
    # floats, no names, no calls. The prompt nudges the solver to
    # pre-simplify; this fallback handles rare slippage.
    m4 = re.search(r'mkRat\s+(\(.+?\))\s+(\d+)', witness)
    if m4:
        try:
            import ast as _ast
            tree = _ast.parse(m4.group(1), mode='eval')
            def _safe_eval(node):
                if isinstance(node, _ast.Expression):
                    return _safe_eval(node.body)
                if isinstance(node, _ast.Constant) and isinstance(node.value, int):
                    return node.value
                if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.USub):
                    return -_safe_eval(node.operand)
                if isinstance(node, _ast.BinOp) and isinstance(node.op, (_ast.Add, _ast.Sub, _ast.Mult)):
                    l, r = _safe_eval(node.left), _safe_eval(node.right)
                    if isinstance(node.op, _ast.Add): return l + r
                    if isinstance(node.op, _ast.Sub): return l - r
                    return l * r
                raise ValueError(f"disallowed AST node: {type(node).__name__}")
            n = _safe_eval(tree)
            return f"{n}/{m4.group(2)}"
        except (SyntaxError, ValueError, TypeError):
            pass
    # Bare Value.str "..." → unquoted string
    m4 = re.search(r'Value\.str\s+"([^"]*)"', witness)
    if m4:
        return m4.group(1)
    # Bare quoted string literal (e.g. `refine ⟨"manipur", ...⟩`).
    # Strip quotes for downstream comparison.
    m_str = re.match(r'^"([^"]*)"$', witness)
    if m_str:
        return m_str.group(1)
    # Bare integer
    if re.match(r'^-?\d+$', witness):
        return witness
    # Legacy guillemet-wrapped name. Return bare name.
    guil = re.match(r'^«([^»]*)»$', witness)
    if guil:
        return guil.group(1).strip()
    return witness


def _parse_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
        return None


def _render_table(table, max_rows=100):
    """Render table as text. Default 100 rows.

    Canonicalize column headers and cell values to one-line form
    before joining with " | ". Source tables can carry literal
    embedded newlines in headers; without canonicalization, the
    rendered TABLE would split across multiple display lines and
    fragmented tokens would reach the solver. Canonicalization
    here is for display only; the underlying table.columns list
    keeps the original strings, and tool-execution paths apply
    their own canonicalization for matching.
    """
    cols_disp = [_safe_comment_text(c) for c in table.columns]
    lines = [" | ".join(cols_disp)]
    lines.append("-" * len(lines[0]))
    for row in table.rows[:max_rows]:
        lines.append(" | ".join(_safe_comment_text(v) for v in row))
    if table.row_count > max_rows:
        lines.append(f"... ({table.row_count - max_rows} more rows)")
    return "\n".join(lines)
