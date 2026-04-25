"""Masked Table: deterministic, trusted structural summary of a
TableData for variant B / F formalizers.

The masked table shows column types and data anomalies but hides
numeric cell values. String cells are also hidden by default, except
for low-cardinality string columns whose distinct values are exposed
via ``string_col_categories`` (and only when no distinct value is
numeric-bearing, to prevent leaking numeric content like "17 (R&B)"
through a string column).

Single source of truth for:
- How columns are classified (50% numeric threshold).
- Which cells become anomalies (per-cell parse failure in a numeric
  column, including empty cells).
- How the formalizer prompt sees the table (via
  ``render_masked_for_prompt``).
- How the code-generated dictionary is seeded (via
  ``build_dictionary_from_headers``).

Design principles:
1. Immutable. ``MaskedCell`` and ``MaskedTable`` are frozen
   dataclasses whose container fields are tuples of pairs, not
   ``dict``. Lookups go through helper methods.
2. Deterministic. Same ``TableData`` in produces the same
   ``MaskedTable`` out; no LLM, no time-dependent state.
3. Parseability parity with the stats adapter (and only parseability
   parity).
4. No cache: every call rebuilds the masked table.
5. Prompt-safe strings: every table-derived string that flows into
   the prompt is run through ``_sanitize_for_prompt``.
6. No value bias: numeric values are never exposed; string values
   only when the column is low-cardinality and all distinct values
   are non-numeric-bearing.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction

from table_adapter import TableData, _parse_exact
from dictionary import Dictionary, DictEntry, Alias


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NUMERIC_THRESHOLD = 0.5               # matches table_adapter.schema()
_MAX_CATEGORIES = 5                    # max distinct values to expose
_MAX_ROWS_IN_RENDER = 100              # row display cap
_MAX_PROMPT_ATOM_LEN = 200             # sanitizer length cap

# Cell kinds (open enum)
KIND_INT = "int"
KIND_RAT = "rat"
KIND_STR = "str"
KIND_ANOMALY = "anomaly"
KIND_EMPTY = "empty"


# ---------------------------------------------------------------------------
# Sanitizers
# ---------------------------------------------------------------------------

def _is_numeric_bearing(s: str) -> bool:
    """A string is numeric-bearing if it parses via ``_parse_exact`` OR
    contains any digit character.

    The second condition catches annotated numbers like ``"17 (R&B)"``
    and ``"3 rated"`` that ``_parse_exact`` would reject but that still
    leak numeric content into the formalizer prompt.
    """
    if not isinstance(s, str):
        return True  # defensive: non-string is always suspicious
    if _parse_exact(s) is not None:
        return True
    return any(ch.isdigit() for ch in s)


def _sanitize_for_prompt(value: object, max_len: int = _MAX_PROMPT_ATOM_LEN) -> str:
    """Coerce a value into a prompt-safe string.

    Unlike ``feedback._safe_prompt_atom`` (which rejects), this helper
    COERCES: non-strings are cast via ``str()``, control characters
    become spaces, and over-long strings are truncated with an
    ellipsis. The rationale is that a benchmark cell with a stray
    newline should not crash the pipeline — it should render safely.

    Control characters in the ASCII control range 0x00-0x1F (except
    space 0x20 itself) and DEL 0x7F are replaced with spaces. All
    other characters pass through.
    """
    if not isinstance(value, str):
        value = str(value)
    if len(value) > max_len:
        value = value[: max_len - 3] + "..."
    chars = []
    for ch in value:
        o = ord(ch)
        if o < 0x20 or o == 0x7F:
            chars.append(" ")
        else:
            chars.append(ch)
    return "".join(chars)


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

def _classify_column(values: list[str]) -> str:
    """Return ``"int"``, ``"rat"``, or ``"str"`` for a column.

    Uses the 50% threshold from ``table_adapter.schema()``: a column
    is numeric-typed if more than 50% of rows (including empty cells,
    matching ``schema()`` behavior) parse via ``_parse_exact``. Among
    numeric columns, ``"rat"`` wins if any parsed value has a
    non-unit denominator; otherwise ``"int"``.

    Matches the pipeline's column typing so the masked-table
    classification agrees with ``schema()``.
    """
    if not values:
        return "str"
    numeric_count = 0
    any_rat = False
    for v in values:
        parsed = _parse_exact(v)
        if parsed is not None:
            numeric_count += 1
            if parsed.denominator != 1:
                any_rat = True
    frac = numeric_count / len(values)
    if frac > _NUMERIC_THRESHOLD:
        return "rat" if any_rat else "int"
    return "str"


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaskedCell:
    """A single masked cell. Numeric values are hidden behind the
    ``kind`` field. Non-parseable cells in numeric columns become
    ``kind="anomaly"`` with ``anomaly_value`` preserving the raw
    text. Empty cells in numeric columns become ``kind="empty"``.
    """
    kind: str
    anomaly_value: str = ""


@dataclass(frozen=True)
class MaskedTable:
    """Immutable structural summary of a TableData.

    All container fields are tuples of pairs (not dicts) so the
    dataclass is genuinely immutable. Callers use the helper methods
    ``col_type_of`` / ``categories_of`` / ``category_omission_reason_of``
    for lookup.

    ``anomalies`` is a tuple of ``(row_idx, col_name, kind, raw_value)``
    where ``kind`` is either ``"non_parseable"`` or ``"empty"``.

    ``category_omission_reasons`` records why a str column was omitted
    from ``string_col_categories``. Possible values:
    - ``"numeric_bearing"`` — at least one distinct value parses as a
      number or contains a digit.
    - ``"high_cardinality"`` — more than ``_MAX_CATEGORIES`` distinct
      non-empty values.
    Only str columns that were actually checked for categorization
    have an entry; int/rat columns do not.
    """
    table_id: str
    columns: tuple[str, ...]
    col_types: tuple[tuple[str, str], ...]
    cells: tuple[tuple[MaskedCell, ...], ...]
    row_count: int
    anomalies: tuple[tuple[int, str, str, str], ...]
    string_col_categories: tuple[tuple[str, tuple[str, ...]], ...]
    category_omission_reasons: tuple[tuple[str, str], ...] = ()
    caption: str = ""

    def col_type_of(self, col: str) -> str:
        for name, t in self.col_types:
            if name == col:
                return t
        raise KeyError(col)

    def categories_of(self, col: str) -> tuple[str, ...] | None:
        for name, vals in self.string_col_categories:
            if name == col:
                return vals
        return None

    def category_omission_reason_of(self, col: str) -> str | None:
        for name, reason in self.category_omission_reasons:
            if name == col:
                return reason
        return None

    def anomalies_for_column(self, col: str) -> tuple[tuple[int, str, str, str], ...]:
        return tuple(a for a in self.anomalies if a[1] == col)


# ---------------------------------------------------------------------------
# build_masked_table
# ---------------------------------------------------------------------------

def build_masked_table(table: TableData) -> MaskedTable:
    """Deterministic build of a MaskedTable from a TableData.

    Steps:
    1. Classify each column via ``_classify_column`` (50% threshold).
    2. Build per-cell ``MaskedCell`` values:
       - In int/rat columns: parse via ``_parse_exact``. Non-empty
         non-parseable → ``anomaly``. Empty → ``empty``. Parseable →
         ``int`` or ``rat`` (value hidden).
       - In str columns: every cell becomes ``str`` (even if empty).
         No value is ever exposed via ``MaskedCell``.
    3. Collect anomalies across all numeric columns.
    4. For each str column, compute distinct non-empty values. If
       cardinality ≤ ``_MAX_CATEGORIES`` AND no distinct value is
       numeric-bearing, expose them via ``string_col_categories``.
       Otherwise, omit the column from ``string_col_categories``.
    """
    columns = list(table.columns)
    n_cols = len(columns)
    rows = table.rows

    # Extract per-column value lists (handle short rows by padding
    # with empty strings so col_types classification is consistent
    # with the full row grid).
    col_values: list[list[str]] = [[] for _ in range(n_cols)]
    for row in rows:
        for j in range(n_cols):
            col_values[j].append(row[j] if j < len(row) else "")

    col_types_map: dict[str, str] = {}
    for j, col in enumerate(columns):
        col_types_map[col] = _classify_column(col_values[j])
    col_types_tuple = tuple((col, col_types_map[col]) for col in columns)

    # Build cells and collect anomalies
    anomalies_list: list[tuple[int, str, str, str]] = []
    cells_rows: list[tuple[MaskedCell, ...]] = []
    for i, row in enumerate(rows):
        row_cells: list[MaskedCell] = []
        for j, col in enumerate(columns):
            raw = row[j] if j < len(row) else ""
            col_type = col_types_map[col]
            if col_type == "str":
                if raw == "":
                    row_cells.append(MaskedCell(kind=KIND_EMPTY))
                else:
                    row_cells.append(MaskedCell(kind=KIND_STR))
            else:
                # int or rat column
                if raw == "":
                    row_cells.append(MaskedCell(kind=KIND_EMPTY))
                    anomalies_list.append((i, col, "empty", ""))
                else:
                    parsed = _parse_exact(raw)
                    if parsed is None:
                        row_cells.append(MaskedCell(kind=KIND_ANOMALY, anomaly_value=raw))
                        anomalies_list.append((i, col, "non_parseable", raw))
                    else:
                        row_cells.append(MaskedCell(kind=col_type))
        cells_rows.append(tuple(row_cells))

    # Compute string column categories + omission reasons
    string_categories: list[tuple[str, tuple[str, ...]]] = []
    omission_reasons: list[tuple[str, str]] = []
    for j, col in enumerate(columns):
        if col_types_map[col] != "str":
            continue
        distinct: list[str] = []
        seen: set[str] = set()
        overflow = False
        for v in col_values[j]:
            if v == "":
                continue
            if v not in seen:
                seen.add(v)
                distinct.append(v)
                if len(distinct) > _MAX_CATEGORIES:
                    overflow = True
                    break
        if overflow:
            omission_reasons.append((col, "high_cardinality"))
            continue
        if any(_is_numeric_bearing(v) for v in distinct):
            omission_reasons.append((col, "numeric_bearing"))
            continue
        string_categories.append((col, tuple(distinct)))

    return MaskedTable(
        table_id=table.table_id,
        columns=tuple(columns),
        col_types=col_types_tuple,
        cells=tuple(cells_rows),
        row_count=len(rows),
        anomalies=tuple(anomalies_list),
        string_col_categories=tuple(string_categories),
        category_omission_reasons=tuple(omission_reasons),
        caption=table.caption,
    )


# ---------------------------------------------------------------------------
# render_masked_for_prompt
# ---------------------------------------------------------------------------

def _cell_repr(cell: MaskedCell) -> str:
    """One-token rendering per masked cell kind."""
    if cell.kind == KIND_INT:
        return "<int>"
    if cell.kind == KIND_RAT:
        return "<rat>"
    if cell.kind == KIND_STR:
        return "str"
    if cell.kind == KIND_EMPTY:
        return "<empty>"
    if cell.kind == KIND_ANOMALY:
        safe = _sanitize_for_prompt(cell.anomaly_value, max_len=40)
        return f'"{safe}"'
    return "?"


def render_masked_for_prompt(
    masked: MaskedTable, max_rows: int = _MAX_ROWS_IN_RENDER
) -> str:
    """Render the masked table as a prompt-safe string block.

    Every table-derived string (column headers, caption, anomaly
    values, category values) passes through
    ``_sanitize_for_prompt``. Row display is truncated at
    ``max_rows``, but the 'Detected anomalies' section lists every
    anomaly regardless of row truncation.
    """
    safe_id = _sanitize_for_prompt(masked.table_id, max_len=80)
    safe_caption = _sanitize_for_prompt(masked.caption, max_len=120)
    safe_cols = [_sanitize_for_prompt(c, max_len=60) for c in masked.columns]

    lines = [
        "MASKED TABLE (types shown, numeric values hidden):",
        "",
        f"  Table ID: {safe_id}",
    ]
    if safe_caption:
        lines.append(f"  Caption: {safe_caption}")
    lines.append(f"  Columns ({len(masked.columns)}): " + " | ".join(safe_cols))

    # Types row
    type_tokens = [masked.col_type_of(c) for c in masked.columns]
    lines.append("  Types: " + " | ".join(type_tokens))
    lines.append(f"  Row count: {masked.row_count}")
    lines.append("")
    lines.append("  Cells (numeric values hidden; string cells masked):")

    # Row display (capped)
    shown_rows = min(len(masked.cells), max_rows)
    for i in range(shown_rows):
        row_cells = masked.cells[i]
        tokens = [_cell_repr(c) for c in row_cells]
        lines.append(f"    Row {i}: " + " | ".join(tokens))
    if masked.row_count > shown_rows:
        lines.append(f"    ... ({masked.row_count - shown_rows} more rows; see anomaly list below for any hidden anomalies)")
    lines.append("")

    # String column categories (only for columns that passed the
    # numeric-bearing exclusion)
    if masked.string_col_categories:
        lines.append("  String columns with low cardinality:")
        for col, vals in masked.string_col_categories:
            safe_col = _sanitize_for_prompt(col, max_len=60)
            safe_vals = ", ".join(_sanitize_for_prompt(v, max_len=40) for v in vals)
            lines.append(f"    {safe_col}: [{safe_vals}]")
        lines.append("")

    # Complete anomaly list — always full, regardless of row truncation
    if masked.anomalies:
        lines.append(f"  Detected anomalies ({len(masked.anomalies)}):")
        for row_idx, col, kind, raw in masked.anomalies:
            safe_col = _sanitize_for_prompt(col, max_len=60)
            if kind == "empty":
                lines.append(f"    - Row {row_idx} \"{safe_col}\": <empty cell in {masked.col_type_of(col)} column>")
            else:
                safe_raw = _sanitize_for_prompt(raw, max_len=80)
                lines.append(
                    f"    - Row {row_idx} \"{safe_col}\": \"{safe_raw}\" "
                    f"(non-parseable in {masked.col_type_of(col)} column)"
                )
    else:
        lines.append("  Detected anomalies: none")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dictionary seeding
# ---------------------------------------------------------------------------

def _lean_id(s: str) -> str:
    """Lazy wrapper for ``pipeline._to_lean_id`` to avoid a circular
    ``masked_table`` ↔ ``pipeline`` import at module load time."""
    from pipeline import _to_lean_id
    return _to_lean_id(s)


def _category_omission_reason(masked: MaskedTable, col: str) -> str:
    """Return a deterministic human-readable reason string for why
    ``col``'s distinct values were omitted from the dictionary entry.

    Used by ``build_dictionary_from_headers``. Returns one of:
    - ``"cardinality > 5"`` — the column has more than 5 distinct non-empty values
    - ``"at least one distinct value is numeric-bearing"`` — a distinct value parses as a number or contains a digit
    - ``"not categorized"`` — the column was not considered for categorization (e.g. not a str column)
    """
    reason_code = masked.category_omission_reason_of(col)
    if reason_code == "high_cardinality":
        return "cardinality > 5"
    if reason_code == "numeric_bearing":
        return "at least one distinct value is numeric-bearing"
    return "not categorized"


def build_dictionary_from_headers(masked: MaskedTable, base: Dictionary) -> Dictionary:
    """Code-only dictionary seeding. Deep-copies ``base`` (to
    preserve core predicates from ``init_core_dictionary()``) and
    adds one Property entry per column using ``pipeline._to_lean_id``
    for normalization.

    The Property entry's description includes:
    - The inferred column type (int/rat/str)
    - Count and samples of anomalies (if any)
    - Distinct non-numeric-bearing values for low-cardinality str
      columns (if ``string_col_categories`` exposed them)

    No LLM is called; the resulting dictionary is deterministic.
    """
    d = deepcopy(base)
    for col in masked.columns:
        col_type = masked.col_type_of(col)
        col_anomalies = masked.anomalies_for_column(col)

        desc_parts = [f"Table column '{_sanitize_for_prompt(col, max_len=60)}'. Inferred type: {col_type}."]
        if col_anomalies:
            sample_strs: list[str] = []
            for row_idx, _col, kind, raw in col_anomalies[:3]:
                if kind == "empty":
                    sample_strs.append(f"<empty> (row {row_idx})")
                else:
                    sample_strs.append(f"'{_sanitize_for_prompt(raw, max_len=40)}' (row {row_idx})")
            desc_parts.append(
                f"ANOMALIES: {len(col_anomalies)} non-parseable or empty value(s): "
                + ", ".join(sample_strs)
            )

        categories = masked.categories_of(col)
        if categories is not None:
            safe_cats = ", ".join(_sanitize_for_prompt(v, max_len=40) for v in categories)
            desc_parts.append(f"Distinct values: {safe_cats}")
        elif col_type == "str":
            # The category was omitted — report the specific reason
            # (cardinality > 5 OR at least one numeric-bearing value)
            # so the formalizer can reason about which filter is
            # feasible.
            reason = _category_omission_reason(masked, col)
            desc_parts.append(f"Distinct values omitted ({reason}).")

        lean_name = _lean_id(col)
        # Sanitize the alias surface_form so that column headers with
        # control characters (newlines, tabs, etc.) cannot reach
        # Dictionary.to_prompt() output unsanitized. The masked-table
        # render is already sanitized, but the dictionary prompt is a
        # separate interpolation path that must match.
        safe_surface = _sanitize_for_prompt(col, max_len=120)
        d.add_entry(DictEntry(
            name=lean_name,
            type="Property",
            description=" ".join(desc_parts),
            formalization=f"{lean_name} : String  -- column header (per R10 nominalist design)",
            aliases=[Alias(
                surface_form=safe_surface,
                context="column header",
                source="schema",
                grade_impact="no_cap",
            )],
        ))
    return d


def merge_user_dictionary(base: Dictionary, user: Dictionary) -> Dictionary:
    """Merge a user-supplied Dictionary into ``base`` without
    clobbering header-derived structural entries.

    - Entries from ``user`` whose ``name`` is already in ``base``:
      only their ``aliases`` are merged into the existing entry.
      The structural fields (``name``, ``type``, ``formalization``,
      ``description``) are preserved from ``base``.
    - Entries from ``user`` with a new ``name``: added wholesale.

    This preserves the "dictionary entries come from code" property
    while letting callers contribute domain synonyms or Convention
    entries.
    """
    out = deepcopy(base)
    for name, entry in user.entries.items():
        if name in out.entries:
            existing = out.entries[name]
            for alias in entry.aliases:
                existing.add_alias(alias)
        else:
            out.add_entry(deepcopy(entry))
    return out
