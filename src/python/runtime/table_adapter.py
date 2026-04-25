"""
TabFact-style table adapter (trusted tool).

Exact retrieval only. No fuzzy matching, no semantic
interpretation. Returns raw cell values with attestation metadata.

The table is treated as a tool, not an ontology; the adapter maps
storage (rows / columns) to world facts (entity / property / value).
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path


class CondUnknownOp(ValueError):
    """Raised when ``eval_cond`` / ``cond_to_lean`` encounter a
    ``Cond.op`` value not in ``_VALID_COND_OPS``. The stats
    dispatch catches this as ``invalid_cond`` and fails closed
    rather than silently degrading to ``Cond.all``.
    """


class CondUnknownColumn(ValueError):
    """Raised when ``Cond.property`` names a column that is not in the
    table. Previously ``eval_cond`` silently returned False for every
    row in this case, which let the adapter attest ``CountWhere p cond
    0`` for a non-existent property — violating the zero-verified-
    hallucination safety theorem. Fail-closed so the solver abstains
    or retries with a valid column."""


def _is_valid_identity(s: str) -> bool:
    """Heuristic: does this cell value look like a plausible row identifier?

    Conservative check: rejects clear data-corruption patterns
    (HTML/CSS formatting leakage) but accepts numeric-looking strings
    (columns like year/rank are legitimately numeric). Used in stats
    argmax/argmin to fail-closed on malformed source-table data rather
    than propagate corrupt identity to the kernel.

    Rejects:
    - Empty / whitespace-only strings
    - HTML attribute fragments: "align =", "style =", "colspan =", "rowspan ="
    - HTML/wiki tags: "<td", "<th", "<tr", "{|", "|}"

    Motivated by source tables that contain rows with HTML
    formatting leakage in the identity column (e.g., wiki-derived
    tables); without this filter, argmax could return such a row
    as the entity answer.
    """
    if not s or not s.strip():
        return False
    s_lower = s.strip().lower()
    html_patterns = [
        'align =', 'align=', 'style =', 'style=',
        'colspan =', 'colspan=', 'rowspan =', 'rowspan=',
        '<td', '<th', '<tr', '{|', '|}',
    ]
    for pat in html_patterns:
        if pat in s_lower:
            return False
    return True


def _canonicalize_col(s: str) -> str:
    """Shared column-name canonicalizer. Applied at table load
    time and at every column-name lookup so names survive
    different whitespace conventions (e.g. literal ``\\n`` inside
    multi-line headers vs. a fixture's space-collapsed form).
    Lowercases, strips surrounding whitespace, collapses any run
    of whitespace (including ``\\n`` / ``\\t``) to a single space,
    and normalizes whitespace surrounding common punctuation so
    "july 1 , 2013" and "july 1, 2013" canonicalize identically.
    """
    # Step 1: lowercase, strip, collapse whitespace runs to single space
    s = re.sub(r"\s+", " ", s.lower().strip())
    # Step 2: remove whitespace BEFORE commas, periods, colons, semicolons,
    # question marks, exclamations, and closing brackets/parens. Preserve
    # single space after. Handles "july 1 , 2013" -> "july 1, 2013".
    s = re.sub(r"\s+([,.;:?!\)\]])", r"\1", s)
    # Step 3: ensure single space AFTER comma/semicolon/colon if followed
    # by non-space (rare edge case). "a,b" -> "a, b". Leave other
    # punctuation untouched to avoid over-normalizing proper nouns.
    s = re.sub(r"([,;])([^\s])", r"\1 \2", s)
    return s


@dataclass
class TableData:
    """Canonical internal representation of a table."""
    table_id: str
    columns: list[str]
    rows: list[list[str]]  # rows[i][j] = cell value at row i, column j
    caption: str = ""

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def col_count(self) -> int:
        return len(self.columns)


@dataclass
class LookupResult:
    """Result of a table lookup. Either success with attested value, or error."""
    success: bool
    value: str | None = None
    row_idx: int | None = None
    col_name: str | None = None
    error: str | None = None
    # For multi-match
    all_matches: list[dict] | None = None


@dataclass
class SchemaResult:
    columns: list[str]
    row_count: int
    caption: str
    col_types: dict[str, str]  # inferred types: "int", "float", "str"


def load_tabfact_table(path: str | Path) -> TableData:
    """Load a TabFact table from #-delimited CSV. Applies the shared
    column canonicalizer so downstream lookups match fixture-
    normalized names even if the CSV has multi-whitespace headers."""
    path = Path(path)
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Empty table: {path}")

    columns = [_canonicalize_col(c) for c in lines[0].split("#")]
    seen: dict[str, int] = {}
    for i, c in enumerate(columns):
        if c in seen:
            raise ValueError(
                f"column collision after canonicalization in {path}: "
                f"positions {seen[c]} and {i} both canonicalize to {c!r}"
            )
        seen[c] = i
    rows = [line.split("#") for line in lines[1:]]

    return TableData(
        table_id=path.stem,
        columns=columns,
        rows=[[cell.strip() for cell in row] for row in rows],
        caption=path.stem.replace("-", " "),
    )


def _col_index(table: TableData, col_name: str) -> int | None:
    """Find column index by canonicalized name. Case-insensitive,
    whitespace-collapsed (see ``_canonicalize_col``). Table columns
    are canonicalized at load (see ``tablebench_to_tabledata``), so
    this only needs to canonicalize the lookup key."""
    target = _canonicalize_col(col_name)
    for i, c in enumerate(table.columns):
        if _canonicalize_col(c) == target:
            return i
    return None


def _parse_numeric(s: str) -> int | None:
    """Parse a cell value as integer. Returns None if not numeric.
    LEGACY: truncates decimals. Use _parse_exact for precise values.
    Kept for backward compatibility with non-stats code paths (e.g., cell_lookup)."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return int(float(s))
    except ValueError:
        return None


def schema(table: TableData) -> SchemaResult:
    """Return table schema with inferred column types."""
    col_types = {}
    for i, col in enumerate(table.columns):
        # Check if column is mostly numeric
        numeric_count = 0
        for row in table.rows:
            if i < len(row) and _parse_exact(row[i]) is not None:
                numeric_count += 1
        if table.row_count > 0 and numeric_count / table.row_count > 0.5:
            col_types[col] = "int"
        else:
            col_types[col] = "str"

    return SchemaResult(
        columns=table.columns,
        row_count=table.row_count,
        caption=table.caption,
        col_types=col_types,
    )


def cell_lookup(table: TableData, row: int, col: str) -> LookupResult:
    """Exact cell lookup by row index and column name. Trusted."""
    col_idx = _col_index(table, col)
    if col_idx is None:
        return LookupResult(
            success=False,
            error=f"Column '{col}' not found. Valid columns: {table.columns}",
        )
    if row < 0 or row >= table.row_count:
        return LookupResult(
            success=False,
            error=f"Row {row} out of range [0, {table.row_count})",
        )
    value = table.rows[row][col_idx]
    return LookupResult(success=True, value=value, row_idx=row, col_name=col)


def find_rows(table: TableData, col: str, value: str) -> LookupResult:
    """Find all rows where column exactly matches value (case-insensitive). Trusted."""
    col_idx = _col_index(table, col)
    if col_idx is None:
        return LookupResult(
            success=False,
            error=f"Column '{col}' not found. Valid columns: {table.columns}",
        )
    value_lower = value.lower().strip()
    matches = []
    for i, row in enumerate(table.rows):
        if i < len(table.rows) and col_idx < len(row):
            if row[col_idx].lower().strip() == value_lower:
                matches.append({"row_idx": i, "row_data": {c: row[j] for j, c in enumerate(table.columns) if j < len(row)}})

    if not matches:
        return LookupResult(
            success=False,
            error=f"No row where '{col}' = '{value}'. Available values in '{col}': {_sample_values(table, col_idx)}",
        )

    return LookupResult(
        success=True,
        value=matches[0]["row_data"].get(col, ""),
        row_idx=matches[0]["row_idx"],
        all_matches=matches,
    )


def column_values(table: TableData, col: str) -> LookupResult:
    """Return all values in a column. Trusted. Used for aggregation."""
    col_idx = _col_index(table, col)
    if col_idx is None:
        return LookupResult(
            success=False,
            error=f"Column '{col}' not found. Valid columns: {table.columns}",
        )
    values = []
    for i, row in enumerate(table.rows):
        if col_idx < len(row):
            values.append({"row_idx": i, "value": row[col_idx]})

    return LookupResult(
        success=True,
        all_matches=values,
    )


def _parse_numeric(s: str) -> int | None:
    """Parse a cell value as integer. Returns None if not numeric.
    LEGACY: truncates decimals. Use _parse_exact for precise values."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return int(float(s))
    except ValueError:
        return None


from fractions import Fraction

def _parse_exact(s: str) -> Fraction | None:
    """Parse a cell value as exact Fraction (strict mode). Preserves decimal precision.

    ``'3.5'`` → ``Fraction(7, 2)``, ``'100'`` → ``Fraction(100, 1)``,
    ``'2.356%'`` → ``Fraction(2356, 1000)``.

    Also normalizes benchmark-specific formatting quirks:
    - leading unary sign separated from the number by whitespace,
      e.g. ``'- 86.6'`` → ``-86.6``, ``'+ 42'`` → ``42``. TableBench
      emits this shape for some numeric columns; without the
      normalization, ``Fraction`` rejects the string and ``_classify_rows``
      silently skips the row, which was the Phase 2G Bug #4 silent-
      aggregation mode (e.g. ``MinWhere`` returned 21 instead of -86.6
      on the boiling-points table).

    For non-strict cell formats (e.g., ``'17 (R&B)'``, ``'3.4 km'``),
    use ``_parse_with_mode`` with ``parse_mode='leading_int'`` or
    ``'leading_number'`` instead. See specs/tier1-fixture-spec.md §1.1
    for parse_mode semantics.
    """
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    if not s:
        return None
    # Normalize unary sign with optional whitespace (but require a
    # digit after, so "- " or "-" alone don't parse as zero).
    m = re.match(r"^([+-])\s+(?=[\d.])", s)
    if m:
        s = m.group(1) + s[m.end():]
    try:
        return Fraction(s).limit_denominator(10**10)
    except (ValueError, ZeroDivisionError):
        return None


_LEADING_INT_RE = re.compile(r"^\s*([+-]?\d+)")
_LEADING_NUMBER_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)")


def _parse_with_mode(s: str, parse_mode: str = "strict") -> Fraction | None:
    """Parse a cell as Fraction under the given parse_mode.

    Per specs/tier1-fixture-spec.md §1.1:
      - "strict": only pure numeric (delegates to _parse_exact).
      - "leading_int": try strict first; else parse the leading integer
        (e.g., '17 (R&B)' → 17, '8 (annot)' → 8).
      - "leading_number": try strict first; else parse the leading
        decimal (e.g., '3.4 km' → 3.4, '99.5%' → 99.5 — though '%' is
        already stripped by strict).
    """
    parsed = _parse_exact(s)
    if parsed is not None:
        return parsed
    if not s or not s.strip():
        return None
    if parse_mode == "strict":
        return None
    if parse_mode == "leading_int":
        m = _LEADING_INT_RE.match(s)
        if m:
            try:
                return Fraction(int(m.group(1)))
            except (ValueError, ZeroDivisionError):
                return None
        return None
    if parse_mode == "leading_number":
        m = _LEADING_NUMBER_RE.match(s)
        if m:
            try:
                return Fraction(m.group(1)).limit_denominator(10**10)
            except (ValueError, ZeroDivisionError):
                return None
        return None
    # Unknown parse_mode → conservative fallback to strict (already None)
    return None


def _lean_int_lit(n: int) -> str:
    """Format an int as a Lean term, parenthesizing negative literals so
    they are not parsed as the minus operator applied to the next argument
    (e.g., `mkRat -81 25` parses as `(mkRat -) 81 25`)."""
    return f"({n})" if n < 0 else str(n)


def _fraction_to_lean(f: Fraction) -> str:
    """Convert Python Fraction to Lean Value expression.
    Always uses Value.rat for consistency — no int/rat mismatch."""
    return f"Value.rat (mkRat {_lean_int_lit(f.numerator)} {f.denominator})"


def _fraction_to_lean_paren(f: Fraction) -> str:
    """Parenthesized version for use as function arguments."""
    return f"(Value.rat (mkRat {_lean_int_lit(f.numerator)} {f.denominator}))"


# ============================================================
# Statistical Tool — trusted aggregation functions
# ============================================================

@dataclass
class StatsResult:
    success: bool
    operation: str = ""
    value: int | float | None = None
    count: int = 0
    details: str = ""
    error: str = ""
    entity: str = ""       # for argmax/argmin: which entity has the extreme value
    entity_col: str = ""   # which column identifies the entity
    row_idx: int = -1      # which row
    # Coverage metadata: populated by any stats call that scans a
    # numeric column through _classify_rows. None for pure
    # structural ops (row count, count_distinct on string cols)
    # that are not coverage-bounded. Consumers should check
    # `coverage is not None and coverage.is_partial` before
    # capping grade.
    coverage: "CoverageReport | None" = None


_VALID_COND_OPS = frozenset({
    "all", "eq", "gt", "lt", "gte", "lte", "contains", "and", "or", "not",
})


def _resolve_prop_col(columns: list[str], prop: str) -> int:
    """Return the column index matching ``prop`` via canonicalized
    lookup, or raise ``CondUnknownColumn``. Used by ``eval_cond`` so a
    typo or whitespace mismatch fails closed instead of quietly
    matching zero rows."""
    target = _canonicalize_col(prop)
    for i, c in enumerate(columns):
        if _canonicalize_col(c) == target:
            return i
    raise CondUnknownColumn(
        f"property {prop!r} not in columns "
        f"{[c for c in columns[:8]]}{'...' if len(columns) > 8 else ''}"
    )


def eval_cond(cond: dict, row: list[str], columns: list[str]) -> bool:
    """Evaluate a Cond against a table row. Returns True if the
    row satisfies the condition.

    Raises ``CondUnknownOp`` on unknown ``op`` values, and
    ``CondUnknownColumn`` when a ``Cond.property`` name does not
    resolve to any table column. (An earlier version silently
    returned False for every row in the unknown-column case, which
    let the adapter attest a CountWhere on a non-existent
    property.) Both subclass ``ValueError`` so existing
    ``except ValueError`` handlers keep working.
    """
    op = cond.get("op", "all")
    if op not in _VALID_COND_OPS:
        raise CondUnknownOp(
            f"unknown Cond op {op!r}; expected one of {sorted(_VALID_COND_OPS)}"
        )
    if op == "all":
        return True
    if op in ("eq", "gt", "lt", "gte", "lte"):
        prop = cond.get("property", "")
        col_idx = _resolve_prop_col(columns, prop)
        if col_idx >= len(row):
            return False
        cell = row[col_idx].strip()
        target = cond.get("value")
        def _parse_float(s):
            try:
                return float(str(s).strip().replace(",", "").replace("$", "").replace("%", ""))
            except (ValueError, TypeError):
                return None
        cell_num = _parse_float(cell)
        target_num = float(target) if isinstance(target, (int, float)) else _parse_float(str(target))
        if cell_num is not None and target_num is not None:
            if op == "eq":  return cell_num == target_num
            if op == "gt":  return cell_num > target_num
            if op == "lt":  return cell_num < target_num
            if op == "gte": return cell_num >= target_num
            if op == "lte": return cell_num <= target_num
        cell_str = cell.lower()
        target_str = str(target).strip().lower()
        if op == "eq": return cell_str == target_str
        return False
    if op == "contains":
        prop = cond.get("property", "")
        col_idx = _resolve_prop_col(columns, prop)
        if col_idx >= len(row):
            return False
        cell = row[col_idx].strip().lower()
        target = str(cond.get("value", "")).strip().lower()
        if not target:
            return False
        return target in cell
    if op == "and":
        return eval_cond(cond.get("left", {}), row, columns) and eval_cond(cond.get("right", {}), row, columns)
    if op == "or":
        return eval_cond(cond.get("left", {}), row, columns) or eval_cond(cond.get("right", {}), row, columns)
    # op == "not"
    return not eval_cond(cond.get("inner", {}), row, columns)


def _sample_skipped(skipped: list[tuple[int, str]]) -> tuple[str, ...]:
    """Return up to 5 sampled raw values from a skipped-rows list,
    with sanitization (coerce control chars to spaces, cap length)
    compatible with ``feedback.CoverageReport.sampled_skipped_values``.

    Matches the validation in ``feedback._safe_prompt_atom`` but
    coerces rather than raising, since benchmark cells can contain
    arbitrary text.
    """
    from feedback import _safe_prompt_atom
    result: list[str] = []
    for _, raw in skipped[:5]:
        # Coerce: replace control chars with spaces, cap length
        cleaned_chars: list[str] = []
        for ch in str(raw):
            o = ord(ch)
            if o < 0x20 or o == 0x7F:
                cleaned_chars.append(" ")
            else:
                cleaned_chars.append(ch)
        cleaned = "".join(cleaned_chars)[:200]
        # Apply the feedback.py strict check via try/except: if it
        # still fails, fall back to a safe placeholder rather than
        # crashing the whole stats call.
        try:
            _safe_prompt_atom(cleaned, "sample", 200)
        except ValueError:
            cleaned = "<unrepresentable>"
        result.append(cleaned)
    return tuple(result)


def _classify_rows(
    table: TableData,
    col_idx: int,
    cond: dict,
    column_name: str,
    parse_mode: str = "strict",
):
    """Shared helper for every coverage-aware stats function.

    Returns ``(used, coverage)`` where:
    - ``used`` is a list of ``(row_idx, Fraction)`` — rows that passed
      the Cond filter AND had a parseable numeric cell in the target
      column under the given ``parse_mode`` (default: "strict")
    - ``coverage`` is a ``feedback.CoverageReport`` describing eligible
      rows (passed Cond), used rows (subset parseable), and sampled
      skipped values

    Empty cells count as "skipped" so the no-silent-skipping
    discipline applies to empty-cell anomalies too.

    ``parse_mode`` selects the cell-parsing convention; the
    per-table tool curation determines which mode applies.
    Supported values:
    - "strict" (default): pure numeric only. Backward-compatible.
    - "leading_int": '17 (R&B)' → 17 (and pure numeric).
    - "leading_number": '3.4 km' → 3.4 (and pure numeric).
    """
    from feedback import CoverageReport

    used: list[tuple[int, Fraction]] = []
    skipped: list[tuple[int, str]] = []
    for i, row in enumerate(table.rows):
        if not eval_cond(cond, row, table.columns):
            continue
        cell = row[col_idx] if col_idx < len(row) else ""
        parsed = _parse_with_mode(cell, parse_mode)
        if parsed is None:
            skipped.append((i, cell))
        else:
            used.append((i, parsed))

    coverage = CoverageReport(
        column=column_name if column_name else "<row count>",
        rows_eligible=len(used) + len(skipped),
        rows_used=len(used),
        skipped_row_count=len(skipped),
        sampled_skipped_values=_sample_skipped(skipped),
    )
    return used, coverage


def stats_count_where(table: TableData, column: str, cond: dict) -> StatsResult:
    """Count rows satisfying ``cond``, independent of any aggregation
    column. Implements the ``CountWhere p cond v`` predicate semantic.

    ``column`` is retained as the ``CountWhere`` predicate's property
    label (the Lean WProp shape is ``CountWhere «column» cond value``),
    but it is NOT used to decide which rows count — the filter is
    entirely in ``cond``. A row with an empty or non-numeric cell in
    ``column`` still counts if ``cond`` matches.

    Never skips rows based on the target column's parseability, so
    coverage reports zeros for the skipped fields.
    """
    from feedback import CoverageReport
    if cond is None:
        cond = {"op": "all"}
    # Validate the nominal ``column`` label resolves to a real table
    # column. The WProp shape is ``CountWhere «column» cond value``;
    # attesting a property name that does not exist as a column would
    # be unsound. (The cond's ``property`` columns are validated
    # separately by ``eval_cond`` below.)
    if _col_index(table, column) is None:
        return StatsResult(
            success=False,
            operation="count_where",
            error=f"Column '{column}' not found. Valid: {table.columns}",
            coverage=CoverageReport(
                column=column if column else "<row count>",
                rows_eligible=0, rows_used=0, skipped_row_count=0,
            ),
        )
    matching = 0
    try:
        for row in table.rows:
            if eval_cond(cond, row, table.columns):
                matching += 1
    except CondUnknownColumn as e:
        # Property column in cond does not exist. Emit a
        # column_not_found error so the Stage-7 classifier maps it to
        # an actionable refinement signal and variant-F can retry.
        return StatsResult(
            success=False,
            operation="count_where",
            error=f"Column not found in condition: {e}",
            coverage=CoverageReport(
                column=column if column else "<row count>",
                rows_eligible=0, rows_used=0, skipped_row_count=0,
            ),
        )
    except ValueError as e:
        return StatsResult(
            success=False,
            operation="count_where",
            error=f"invalid_cond: {e}",
            coverage=CoverageReport(
                column=column if column else "<row count>",
                rows_eligible=0, rows_used=0, skipped_row_count=0,
            ),
        )
    coverage = CoverageReport(
        column=column if column else "<row count>",
        rows_eligible=matching,
        rows_used=matching,
        skipped_row_count=0,
    )
    return StatsResult(
        success=True,
        operation="count_where",
        value=matching,
        count=matching,
        details=f"Count where {cond}: {matching}",
        coverage=coverage,
    )


def cond_to_lean(cond: dict) -> str:
    """Convert a JSON condition to a Lean Cond expression."""
    op = cond.get("op", "all")
    if op == "all":
        return "Cond.all"
    if op in ("eq", "gt", "lt", "gte", "lte"):
        prop = cond.get("property", "")
        lean_prop = f"«{_canonicalize_col(prop)}»"
        target = cond.get("value")
        if isinstance(target, (int, float)):
            f = Fraction(target).limit_denominator(10**10)
            lean_val = _fraction_to_lean_paren(f)
        else:
            lean_val = f'(Value.str "{target}")'
        op_map = {"eq": "propEq", "gt": "propGt", "lt": "propLt", "gte": "propGte", "lte": "propLte"}
        return f"(Cond.{op_map[op]} {lean_prop} {lean_val})"
    if op == "contains":
        prop = cond.get("property", "")
        lean_prop = f"«{_canonicalize_col(prop)}»"
        target = str(cond.get("value", ""))
        lean_val = f'(Value.str "{target}")'
        return f"(Cond.propContains {lean_prop} {lean_val})"
    if op == "and":
        left = cond_to_lean(cond.get("left", {}))
        right = cond_to_lean(cond.get("right", {}))
        return f"(Cond.and {left} {right})"
    if op == "or":
        left = cond_to_lean(cond.get("left", {}))
        right = cond_to_lean(cond.get("right", {}))
        return f"(Cond.or {left} {right})"
    if op == "not":
        inner = cond_to_lean(cond.get("inner", {}))
        return f"(Cond.not {inner})"
    # Fail closed: refuse to silently degrade an unknown op to Cond.all.
    # That older behavior was a silent-failure mode in the Lean rendering
    # path parallel to eval_cond's silent-false rows bug.
    raise CondUnknownOp(
        f"cond_to_lean: unknown Cond op {op!r}; expected one of "
        f"{sorted(_VALID_COND_OPS)}"
    )


def stats_agg_where(table: TableData, operation: str, agg_col: str,
                    cond: dict | None = None,
                    parse_mode: str = "strict") -> StatsResult:
    """General aggregation with condition filtering. Trusted.

    Uses ``_classify_rows`` to compute eligible/used/skipped row
    counts, then aggregates over the used rows. Emits a
    ``CoverageReport`` on every call (success or failure) so
    callers can implement coverage-aware grading.

    ``parse_mode`` selects the cell-parsing convention; see
    ``_parse_with_mode``.
    """
    from feedback import CoverageReport

    col_idx = _col_index(table, agg_col)
    if col_idx is None:
        return StatsResult(
            success=False,
            error=f"Column '{agg_col}' not found. Valid: {table.columns}",
            coverage=CoverageReport(
                column=agg_col if agg_col else "<unknown>",
                rows_eligible=0, rows_used=0, skipped_row_count=0,
            ),
        )

    if cond is None:
        cond = {"op": "all"}

    try:
        used, coverage = _classify_rows(table, col_idx, cond, agg_col, parse_mode=parse_mode)
    except CondUnknownColumn as e:
        return StatsResult(
            success=False,
            error=f"Column not found in condition: {e}",
            coverage=CoverageReport(
                column=agg_col, rows_eligible=0, rows_used=0, skipped_row_count=0,
            ),
        )
    except ValueError as e:
        return StatsResult(
            success=False,
            error=f"invalid_cond: {e}",
            coverage=CoverageReport(
                column=agg_col, rows_eligible=0, rows_used=0, skipped_row_count=0,
            ),
        )

    if not used:
        return StatsResult(
            success=False,
            error=f"No matching numeric values for '{agg_col}' with given condition",
            coverage=coverage,
        )

    values = [v for _, v in used]
    cond_desc = f" where {cond}" if cond.get("op") != "all" else ""
    skipped_desc = (
        f"; {coverage.skipped_row_count} skipped"
        if coverage.is_partial else ""
    )

    if operation == "sum":
        total = sum(values, Fraction(0))
        return StatsResult(
            success=True, operation="sum", value=total, count=len(used),
            details=f"Sum of '{agg_col}'{cond_desc}: {float(total)} "
                    f"({len(used)} rows{skipped_desc})",
            coverage=coverage,
        )
    if operation == "count":
        return StatsResult(
            success=True, operation="count", value=len(used), count=len(used),
            details=f"Count of '{agg_col}'{cond_desc}: {len(used)}{skipped_desc}",
            coverage=coverage,
        )
    if operation == "mean":
        total = sum(values, Fraction(0))
        mean = Fraction(total, len(used))
        return StatsResult(
            success=True, operation="mean", value=mean, count=len(used),
            details=f"Mean of '{agg_col}'{cond_desc}: {float(mean)} "
                    f"(exact: {mean}{skipped_desc})",
            coverage=coverage,
        )
    if operation == "max":
        mx = max(values)
        return StatsResult(
            success=True, operation="max", value=mx, count=len(used),
            details=f"Max of '{agg_col}'{cond_desc}: {mx}{skipped_desc}",
            coverage=coverage,
        )
    if operation == "min":
        mn = min(values)
        return StatsResult(
            success=True, operation="min", value=mn, count=len(used),
            details=f"Min of '{agg_col}'{cond_desc}: {mn}{skipped_desc}",
            coverage=coverage,
        )
    return StatsResult(
        success=False,
        error=f"Unknown operation '{operation}' for stats_agg_where",
        coverage=coverage,
    )


def stats_topk_where(table: TableData, operation: str, agg_col: str,
                     k: int, cond: dict | None = None,
                     rank_by: str | None = None,
                     parse_mode: str = "strict") -> StatsResult:
    """Top-K / Bottom-K aggregation with optional cond filter. Trusted.

    Filters rows by ``cond``, ranks by ``rank_by`` (or ``agg_col`` if not
    given), selects the K largest (top) or smallest (bottom), and sums or
    averages the ``agg_col`` values of the selected rows. Implements the
    ``TopKSumWhere / TopKMeanWhere / TopKSumByWhere / TopKMeanByWhere /
    BottomKSumWhere / BottomKMeanWhere`` WProp shapes.

    ``parse_mode`` selects the cell-parsing convention for both
    rank_by and agg_col cells.
    """
    from feedback import CoverageReport

    agg_idx = _col_index(table, agg_col)
    if agg_idx is None:
        return StatsResult(
            success=False,
            error=f"Column '{agg_col}' not found. Valid: {table.columns}",
            coverage=CoverageReport(column=agg_col, rows_eligible=0,
                                    rows_used=0, skipped_row_count=0),
        )

    rank_col = rank_by if rank_by else agg_col
    rank_idx = _col_index(table, rank_col)
    if rank_idx is None:
        return StatsResult(
            success=False,
            error=f"Column '{rank_col}' not found. Valid: {table.columns}",
            coverage=CoverageReport(column=rank_col, rows_eligible=0,
                                    rows_used=0, skipped_row_count=0),
        )

    if cond is None:
        cond = {"op": "all"}
    try:
        k_int = int(k)
    except (TypeError, ValueError):
        k_int = 0
    if k_int <= 0:
        return StatsResult(
            success=False,
            error=f"K must be a positive integer, got {k!r}",
            coverage=CoverageReport(column=agg_col, rows_eligible=0,
                                    rows_used=0, skipped_row_count=0),
        )

    try:
        used: list[tuple[int, Fraction, Fraction]] = []
        skipped: list[tuple[int, str]] = []
        for i, row in enumerate(table.rows):
            if not eval_cond(cond, row, table.columns):
                continue
            rank_cell = row[rank_idx] if rank_idx < len(row) else ""
            agg_cell = row[agg_idx] if agg_idx < len(row) else ""
            rank_val = _parse_with_mode(rank_cell, parse_mode)
            agg_val = _parse_with_mode(agg_cell, parse_mode)
            if rank_val is None or agg_val is None:
                miss = rank_cell if rank_val is None else agg_cell
                skipped.append((i, miss))
            else:
                used.append((i, rank_val, agg_val))
    except CondUnknownColumn as e:
        return StatsResult(
            success=False,
            error=f"Column not found in condition: {e}",
            coverage=CoverageReport(column=agg_col, rows_eligible=0,
                                    rows_used=0, skipped_row_count=0),
        )
    except ValueError as e:
        return StatsResult(
            success=False,
            error=f"invalid_cond: {e}",
            coverage=CoverageReport(column=agg_col, rows_eligible=0,
                                    rows_used=0, skipped_row_count=0),
        )

    coverage = CoverageReport(
        column=agg_col,
        rows_eligible=len(used) + len(skipped),
        rows_used=len(used),
        skipped_row_count=len(skipped),
        sampled_skipped_values=_sample_skipped(skipped),
    )

    if not used:
        return StatsResult(
            success=False,
            error=f"No matching numeric values for '{agg_col}' with given condition",
            coverage=coverage,
        )

    is_bottom = operation.startswith("bottomk_")
    used.sort(key=lambda t: t[1], reverse=not is_bottom)
    # The L1 predicate `TopK*Cells col k cond v` claims
    # "sum/mean of the K largest" with exact k. Silently using
    # k_eff = min(k, len(used)) when fewer rows are eligible
    # would produce an L1 fact whose k field is the requested
    # value but whose actual selection used a smaller set. Fail
    # closed instead.
    if k_int > len(used):
        return StatsResult(
            success=False,
            error=(f"Requested k={k_int} exceeds eligible (parseable) row count "
                   f"({len(used)}) for '{agg_col}'. The L1 predicate {operation} "
                   f"requires EXACTLY k values; cannot honor with fewer."),
            coverage=coverage,
        )
    selected = used[:k_int]
    agg_vals = [av for _, _, av in selected]
    k_eff = k_int

    direction = "bottom" if is_bottom else "top"
    rank_desc = f" ranked by '{rank_col}'" if rank_by and rank_by != agg_col else ""
    cond_desc = f" where {cond}" if cond.get("op") != "all" else ""
    skipped_desc = (
        f"; {coverage.skipped_row_count} skipped"
        if coverage.is_partial else ""
    )

    if operation in ("topk_sum", "topk_sum_by", "bottomk_sum"):
        total = sum(agg_vals, Fraction(0))
        return StatsResult(
            success=True, operation=operation, value=total, count=k_eff,
            details=f"Sum of {direction}-{k_int} '{agg_col}'{rank_desc}{cond_desc}: "
                    f"{float(total)} ({k_eff} rows{skipped_desc})",
            coverage=coverage,
        )
    if operation in ("topk_mean", "topk_mean_by", "bottomk_mean"):
        total = sum(agg_vals, Fraction(0))
        mean = Fraction(total, k_eff)
        return StatsResult(
            success=True, operation=operation, value=mean, count=k_eff,
            details=f"Mean of {direction}-{k_int} '{agg_col}'{rank_desc}{cond_desc}: "
                    f"{float(mean)} (exact: {mean}; {k_eff} rows{skipped_desc})",
            coverage=coverage,
        )
    return StatsResult(
        success=False,
        error=f"Unknown operation '{operation}' for stats_topk_where",
        coverage=coverage,
    )


def stats_count(table: TableData, col: str, match_value: str | None = None) -> StatsResult:
    """Count rows. If match_value given, count rows where col == match_value. Trusted."""
    col_idx = _col_index(table, col)
    if col_idx is None:
        return StatsResult(success=False, error=f"Column '{col}' not found. Valid: {table.columns}")

    if match_value is None:
        return StatsResult(success=True, operation="count_all", value=table.row_count,
                          count=table.row_count, details=f"Total rows: {table.row_count}")

    match_lower = match_value.strip().lower()
    matches = [i for i, row in enumerate(table.rows)
               if col_idx < len(row) and match_lower in row[col_idx].strip().lower()]
    return StatsResult(success=True, operation="count_where", value=len(matches),
                      count=len(matches), details=f"Rows where '{col}' contains '{match_value}': {len(matches)}")


def stats_count_distinct(table: TableData, col: str) -> StatsResult:
    """Count distinct values in a column. Trusted."""
    col_idx = _col_index(table, col)
    if col_idx is None:
        return StatsResult(success=False, error=f"Column '{col}' not found. Valid: {table.columns}")

    distinct = set()
    for row in table.rows:
        if col_idx < len(row):
            distinct.add(row[col_idx].strip().lower())
    return StatsResult(success=True, operation="count_distinct", value=len(distinct),
                      count=len(distinct), details=f"Distinct values in '{col}': {len(distinct)} = {sorted(distinct)}")


def stats_sum(table: TableData, col: str, parse_mode: str = "strict") -> StatsResult:
    """Sum numeric values in a column. Trusted, coverage-aware
    (routes through ``_classify_rows`` under ``Cond.all``).

    ``parse_mode`` selects the cell-parsing convention; see
    ``_parse_with_mode``.
    """
    from feedback import CoverageReport
    col_idx = _col_index(table, col)
    if col_idx is None:
        return StatsResult(
            success=False, error=f"Column '{col}' not found. Valid: {table.columns}",
            coverage=CoverageReport(column=col, rows_eligible=0, rows_used=0, skipped_row_count=0),
        )
    used, coverage = _classify_rows(table, col_idx, {"op": "all"}, col, parse_mode=parse_mode)
    total = sum((v for _, v in used), Fraction(0))
    return StatsResult(
        success=True, operation="sum", value=total, count=len(used),
        details=f"Sum of '{col}': {total} ({len(used)} numeric values"
                f"{'; ' + str(coverage.skipped_row_count) + ' skipped' if coverage.is_partial else ''})",
        coverage=coverage,
    )


def stats_max(table: TableData, col: str, parse_mode: str = "strict") -> StatsResult:
    """Find max numeric value in a column and which row. Coverage-aware.

    ``parse_mode`` selects the cell-parsing convention.
    """
    from feedback import CoverageReport
    col_idx = _col_index(table, col)
    if col_idx is None:
        return StatsResult(
            success=False, error=f"Column '{col}' not found. Valid: {table.columns}",
            coverage=CoverageReport(column=col, rows_eligible=0, rows_used=0, skipped_row_count=0),
        )
    used, coverage = _classify_rows(table, col_idx, {"op": "all"}, col, parse_mode=parse_mode)
    if not used:
        return StatsResult(
            success=False, error=f"No numeric values in '{col}'",
            coverage=coverage,
        )
    max_val, max_row = max(used, key=lambda p: p[1])[1], max(used, key=lambda p: p[1])[0]
    return StatsResult(
        success=True, operation="max", value=max_val, count=len(used),
        details=f"Max of '{col}': {max_val} (row {max_row}"
                f"{'; ' + str(coverage.skipped_row_count) + ' skipped' if coverage.is_partial else ''})",
        coverage=coverage,
    )


def stats_min(table: TableData, col: str, parse_mode: str = "strict") -> StatsResult:
    """Find min numeric value in a column and which row. Coverage-aware.

    ``parse_mode`` selects the cell-parsing convention.
    """
    from feedback import CoverageReport
    col_idx = _col_index(table, col)
    if col_idx is None:
        return StatsResult(
            success=False, error=f"Column '{col}' not found. Valid: {table.columns}",
            coverage=CoverageReport(column=col, rows_eligible=0, rows_used=0, skipped_row_count=0),
        )
    used, coverage = _classify_rows(table, col_idx, {"op": "all"}, col, parse_mode=parse_mode)
    if not used:
        return StatsResult(
            success=False, error=f"No numeric values in '{col}'",
            coverage=coverage,
        )
    min_val, min_row = min(used, key=lambda p: p[1])[1], min(used, key=lambda p: p[1])[0]
    return StatsResult(
        success=True, operation="min", value=min_val, count=len(used),
        details=f"Min of '{col}': {min_val} (row {min_row}"
                f"{'; ' + str(coverage.skipped_row_count) + ' skipped' if coverage.is_partial else ''})",
        coverage=coverage,
    )


def stats_mean(table: TableData, col: str, scale: int = 100, parse_mode: str = "strict") -> StatsResult:
    """Legacy fixed-point mean helper. Coverage-aware.

    Prefer ``stats_agg_where(..., operation='mean', ...)`` which
    returns an exact ``Fraction`` — the legacy ``stats_mean`` is kept
    only for callers that still want a fixed-point integer value.

    ``parse_mode`` selects the cell-parsing convention.
    """
    from feedback import CoverageReport
    col_idx = _col_index(table, col)
    if col_idx is None:
        return StatsResult(
            success=False, error=f"Column '{col}' not found. Valid: {table.columns}",
            coverage=CoverageReport(column=col, rows_eligible=0, rows_used=0, skipped_row_count=0),
        )
    used, coverage = _classify_rows(table, col_idx, {"op": "all"}, col, parse_mode=parse_mode)
    if not used:
        return StatsResult(
            success=False, error=f"No numeric values in '{col}'",
            coverage=coverage,
        )
    total = sum((v for _, v in used), Fraction(0))
    count = len(used)
    mean_scaled = int(total * scale // count)
    remainder = int(total * scale - mean_scaled * count)
    return StatsResult(
        success=True, operation="mean", value=mean_scaled, count=count,
        details=f"Mean of '{col}': {mean_scaled}/{scale} = {mean_scaled/scale:.2f} "
                f"(sum={total}, count={count}, scale={scale}, remainder={remainder}"
                f"{'; ' + str(coverage.skipped_row_count) + ' skipped' if coverage.is_partial else ''})",
        coverage=coverage,
    )


def stats_argmax(table: TableData, val_col: str, entity_col: str | None = None,
                 parse_mode: str = "strict", condition: dict | None = None) -> StatsResult:
    """Find the entity with the maximum value in val_col.
    Coverage-aware.

    Skipped rows in ``val_col`` are counted in ``coverage`` so
    grade capping can detect silent skipping.

    ``parse_mode`` selects the cell-parsing convention for the
    measure column. ``condition`` (optional) restricts the search
    to rows matching the Cond (defaults to Cond.all).
    """
    from feedback import CoverageReport
    val_idx = _col_index(table, val_col)
    if val_idx is None:
        return StatsResult(
            success=False, error=f"Column '{val_col}' not found. Valid: {table.columns}",
            coverage=CoverageReport(column=val_col, rows_eligible=0, rows_used=0, skipped_row_count=0),
        )

    if entity_col:
        ent_idx = _col_index(table, entity_col)
        if ent_idx is None:
            return StatsResult(
                success=False, error=f"Entity column '{entity_col}' not found.",
                coverage=CoverageReport(column=val_col, rows_eligible=0, rows_used=0, skipped_row_count=0),
            )
    else:
        ent_idx = 0

    cond = condition if condition else {"op": "all"}
    used, coverage = _classify_rows(table, val_idx, cond, val_col, parse_mode=parse_mode)
    if not used:
        return StatsResult(
            success=False, error=f"No numeric values in '{val_col}'",
            coverage=coverage,
        )
    # Runtime identity-cell validator. Filter out rows whose
    # identity-column cell is malformed (HTML leakage, formatting
    # fragments) so argmax cannot return a corrupt entity like
    # "align = left|total".
    used_valid = [
        (i, v) for i, v in used
        if i < len(table.rows) and ent_idx < len(table.rows[i])
        and _is_valid_identity(table.rows[i][ent_idx])
    ]
    if not used_valid:
        return StatsResult(
            success=False,
            error=(f"No rows with valid identity cells in '{val_col}' — "
                   f"all candidate rows had HTML/formatting leakage in entity column '{table.columns[ent_idx]}'. "
                   f"Source-table data quality issue."),
            coverage=coverage,
        )
    max_row_idx, max_val = max(used_valid, key=lambda p: p[1])
    max_entity = table.rows[max_row_idx][ent_idx] if ent_idx < len(table.rows[max_row_idx]) else ""
    return StatsResult(
        success=True, operation="argmax", value=max_val,
        entity=max_entity, entity_col=table.columns[ent_idx], row_idx=max_row_idx,
        details=f"Argmax of '{val_col}': {max_entity} = {max_val} (row {max_row_idx}"
                f"{'; ' + str(coverage.skipped_row_count) + ' skipped' if coverage.is_partial else ''})",
        coverage=coverage,
    )


def stats_argmin(table: TableData, val_col: str, entity_col: str | None = None,
                 parse_mode: str = "strict", condition: dict | None = None) -> StatsResult:
    """Find the entity with the minimum value in val_col. Coverage-aware.

    ``parse_mode`` selects the cell-parsing convention. ``condition``
    (optional) restricts the search to rows matching the Cond.
    """
    from feedback import CoverageReport
    val_idx = _col_index(table, val_col)
    if val_idx is None:
        return StatsResult(
            success=False, error=f"Column '{val_col}' not found. Valid: {table.columns}",
            coverage=CoverageReport(column=val_col, rows_eligible=0, rows_used=0, skipped_row_count=0),
        )

    if entity_col:
        ent_idx = _col_index(table, entity_col)
        if ent_idx is None:
            return StatsResult(
                success=False, error=f"Entity column '{entity_col}' not found.",
                coverage=CoverageReport(column=val_col, rows_eligible=0, rows_used=0, skipped_row_count=0),
            )
    else:
        ent_idx = 0

    cond = condition if condition else {"op": "all"}
    used, coverage = _classify_rows(table, val_idx, cond, val_col, parse_mode=parse_mode)
    if not used:
        return StatsResult(
            success=False, error=f"No numeric values in '{val_col}'",
            coverage=coverage,
        )
    # Runtime identity-cell validator (same motivation as
    # stats_argmax): prevent malformed identity rows from being
    # returned as the argmin result.
    used_valid = [
        (i, v) for i, v in used
        if i < len(table.rows) and ent_idx < len(table.rows[i])
        and _is_valid_identity(table.rows[i][ent_idx])
    ]
    if not used_valid:
        return StatsResult(
            success=False,
            error=(f"No rows with valid identity cells in '{val_col}' — "
                   f"all candidate rows had HTML/formatting leakage in entity column '{table.columns[ent_idx]}'. "
                   f"Source-table data quality issue."),
            coverage=coverage,
        )
    min_row_idx, min_val = min(used_valid, key=lambda p: p[1])
    min_entity = table.rows[min_row_idx][ent_idx] if ent_idx < len(table.rows[min_row_idx]) else ""
    return StatsResult(
        success=True, operation="argmin", value=min_val,
        entity=min_entity, entity_col=table.columns[ent_idx], row_idx=min_row_idx,
        details=f"Argmin of '{val_col}': {min_entity} = {min_val} (row {min_row_idx}"
                f"{'; ' + str(coverage.skipped_row_count) + ' skipped' if coverage.is_partial else ''})",
        coverage=coverage,
    )


def stats_top_n(table: TableData, val_col: str, n: int = 5,
                entity_col: str | None = None, bottom: bool = False) -> StatsResult:
    """Return top-N (or bottom-N) entities sorted by val_col. Trusted."""
    val_idx = _col_index(table, val_col)
    if val_idx is None:
        return StatsResult(success=False, error=f"Column '{val_col}' not found. Valid: {table.columns}")

    if entity_col:
        ent_idx = _col_index(table, entity_col)
        if ent_idx is None:
            return StatsResult(success=False, error=f"Entity column '{entity_col}' not found.")
    else:
        ent_idx = 0

    entries = []
    for i, row in enumerate(table.rows):
        if val_idx < len(row):
            num = _parse_exact(row[val_idx])
            if num is not None:
                ent = row[ent_idx] if ent_idx < len(row) else f"row_{i}"
                entries.append((num, ent, i))

    entries.sort(key=lambda x: x[0], reverse=not bottom)
    top = entries[:n]
    direction = "bottom" if bottom else "top"
    entities = [e[1] for e in top]
    values = [e[0] for e in top]
    return StatsResult(success=True, operation=f"{direction}_{n}", value=values[0] if values else None,
                      count=len(top),
                      details=f"{direction.title()} {n} by '{val_col}': " +
                              ", ".join(f"{e} ({v})" for v, e, _ in top))


def stats_median(table: TableData, col: str, scale: int = 100) -> StatsResult:
    """Median of numeric values in a column, returned as fixed-point. Trusted."""
    col_idx = _col_index(table, col)
    if col_idx is None:
        return StatsResult(success=False, error=f"Column '{col}' not found. Valid: {table.columns}")

    values = []
    for row in table.rows:
        if col_idx < len(row):
            n = _parse_exact(row[col_idx])
            if n is not None:
                values.append(n)
    if not values:
        return StatsResult(success=False, error=f"No numeric values in '{col}'")

    values.sort()
    n = len(values)
    if n % 2 == 1:
        median_scaled = values[n // 2] * scale
    else:
        median_scaled = (values[n // 2 - 1] + values[n // 2]) * scale // 2

    return StatsResult(success=True, operation="median", value=median_scaled,
                      count=n,
                      details=f"Median of '{col}': {median_scaled}/{scale} = {median_scaled/scale:.2f} ({n} values)")


def stats_computed_argmax(table: TableData, num_col: str, den_col: str,
                          entity_col: str | None = None, scale: int = 10000) -> StatsResult:
    """Find entity with highest ratio num_col/den_col. Trusted.
    Returns the ratio as fixed-point (num*scale/den). For "Medal Efficiency = gold/total"."""
    num_idx = _col_index(table, num_col)
    den_idx = _col_index(table, den_col)
    if num_idx is None:
        return StatsResult(success=False, error=f"Column '{num_col}' not found. Valid: {table.columns}")
    if den_idx is None:
        return StatsResult(success=False, error=f"Column '{den_col}' not found. Valid: {table.columns}")

    ent_idx = _col_index(table, entity_col) if entity_col else 0

    best_ratio = None
    best_entity = ""
    best_row = -1
    best_num = 0
    best_den = 0
    for i, row in enumerate(table.rows):
        if num_idx < len(row) and den_idx < len(row):
            n = _parse_exact(row[num_idx])
            d = _parse_exact(row[den_idx])
            if n is not None and d is not None and d != 0:
                ratio = n * scale // d
                if best_ratio is None or ratio > best_ratio:
                    best_ratio = ratio
                    best_entity = row[ent_idx] if ent_idx < len(row) else f"row_{i}"
                    best_row = i
                    best_num = n
                    best_den = d
    if best_ratio is None:
        return StatsResult(success=False, error=f"No valid ratios in '{num_col}'/'{den_col}'")
    return StatsResult(success=True, operation="computed_argmax", value=best_ratio,
                      entity=best_entity, row_idx=best_row,
                      details=f"Argmax of '{num_col}'/'{den_col}': {best_entity} = "
                              f"{best_num}/{best_den} = {best_ratio}/{scale} (row {best_row})")


def stats_computed_argmin(table: TableData, num_col: str, den_col: str,
                          entity_col: str | None = None, scale: int = 10000) -> StatsResult:
    """Find entity with lowest ratio num_col/den_col. Trusted."""
    num_idx = _col_index(table, num_col)
    den_idx = _col_index(table, den_col)
    if num_idx is None:
        return StatsResult(success=False, error=f"Column '{num_col}' not found. Valid: {table.columns}")
    if den_idx is None:
        return StatsResult(success=False, error=f"Column '{den_col}' not found. Valid: {table.columns}")

    ent_idx = _col_index(table, entity_col) if entity_col else 0

    best_ratio = None
    best_entity = ""
    best_row = -1
    for i, row in enumerate(table.rows):
        if num_idx < len(row) and den_idx < len(row):
            n = _parse_exact(row[num_idx])
            d = _parse_exact(row[den_idx])
            if n is not None and d is not None and d != 0:
                ratio = n * scale // d
                if best_ratio is None or ratio < best_ratio:
                    best_ratio = ratio
                    best_entity = row[ent_idx] if ent_idx < len(row) else f"row_{i}"
                    best_row = i
    if best_ratio is None:
        return StatsResult(success=False, error=f"No valid ratios in '{num_col}'/'{den_col}'")
    return StatsResult(success=True, operation="computed_argmin", value=best_ratio,
                      entity=best_entity, row_idx=best_row,
                      details=f"Argmin of '{num_col}'/'{den_col}': {best_entity} = {best_ratio}/{scale} (row {best_row})")


def stats_lag_diff(table: TableData, val_col: str, sort_col: str,
                   entity_col: str | None = None) -> StatsResult:
    """Compute consecutive differences sorted by sort_col. Trusted.
    Returns the max difference, its position, and the full diff list.
    For "year with largest increase" type questions."""
    val_idx = _col_index(table, val_col)
    sort_idx = _col_index(table, sort_col)
    if val_idx is None:
        return StatsResult(success=False, error=f"Column '{val_col}' not found. Valid: {table.columns}")
    if sort_idx is None:
        return StatsResult(success=False, error=f"Column '{sort_col}' not found. Valid: {table.columns}")

    ent_idx = _col_index(table, entity_col) if entity_col else sort_idx

    # Collect (sort_key, value, entity) tuples
    entries = []
    for i, row in enumerate(table.rows):
        if val_idx < len(row) and sort_idx < len(row):
            v = _parse_exact(row[val_idx])
            s = _parse_exact(row[sort_idx])
            if v is not None and s is not None:
                ent = row[ent_idx] if ent_idx < len(row) else str(s)
                entries.append((s, v, ent, i))

    entries.sort(key=lambda x: x[0])

    if len(entries) < 2:
        return StatsResult(success=False, error=f"Need at least 2 rows for lag diff")

    # Compute consecutive differences
    diffs = []
    max_diff = None
    max_diff_entity = ""
    max_diff_row = -1
    for j in range(1, len(entries)):
        diff = entries[j][1] - entries[j-1][1]
        diffs.append((entries[j][2], diff, entries[j][3]))
        if max_diff is None or diff > max_diff:
            max_diff = diff
            max_diff_entity = entries[j][2]
            max_diff_row = entries[j][3]

    diff_summary = ", ".join(f"{ent}:{d}" for ent, d, _ in diffs[:10])
    return StatsResult(success=True, operation="lag_diff", value=max_diff,
                      entity=max_diff_entity, row_idx=max_diff_row,
                      count=len(diffs),
                      details=f"Lag diff of '{val_col}' sorted by '{sort_col}': "
                              f"max change = {max_diff} at {max_diff_entity}. Diffs: {diff_summary}")


def stats_pct_change(table: TableData, val_col: str, sort_col: str,
                     scale: int = 10000) -> StatsResult:
    """Compute max percentage change between consecutive rows. Trusted.
    pct_change = (new - old) * scale / old. For "largest percentage increase"."""
    val_idx = _col_index(table, val_col)
    sort_idx = _col_index(table, sort_col)
    if val_idx is None:
        return StatsResult(success=False, error=f"Column '{val_col}' not found. Valid: {table.columns}")
    if sort_idx is None:
        return StatsResult(success=False, error=f"Column '{sort_col}' not found. Valid: {table.columns}")

    ent_idx = sort_idx

    entries = []
    for i, row in enumerate(table.rows):
        if val_idx < len(row) and sort_idx < len(row):
            v = _parse_exact(row[val_idx])
            s = _parse_exact(row[sort_idx])
            if v is not None and s is not None:
                ent = row[ent_idx] if ent_idx < len(row) else str(s)
                entries.append((s, v, ent, i))

    entries.sort(key=lambda x: x[0])

    if len(entries) < 2:
        return StatsResult(success=False, error=f"Need at least 2 rows for pct change")

    max_pct = None
    max_entity = ""
    max_row = -1
    for j in range(1, len(entries)):
        old = entries[j-1][1]
        new = entries[j][1]
        if old != 0:
            pct = (new - old) * scale // old
            if max_pct is None or pct > max_pct:
                max_pct = pct
                max_entity = entries[j][2]
                max_row = entries[j][3]

    if max_pct is None:
        return StatsResult(success=False, error="No valid percentage changes")
    return StatsResult(success=True, operation="pct_change", value=max_pct,
                      entity=max_entity, row_idx=max_row,
                      details=f"Max pct change of '{val_col}' by '{sort_col}': "
                              f"{max_pct}/{scale} = {max_pct/scale:.2%} at {max_entity}")


def _sample_values(table: TableData, col_idx: int, n: int = 5) -> list[str]:
    """Sample up to n distinct values from a column."""
    seen = set()
    samples = []
    for row in table.rows:
        if col_idx < len(row):
            v = row[col_idx].strip()
            if v not in seen:
                seen.add(v)
                samples.append(v)
                if len(samples) >= n:
                    break
    return samples
