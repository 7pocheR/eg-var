"""EG-VAR formalizer feedback: sanitized signals for variant-F
retries.

Defines the data structures that variant F uses to pass refinement
signals from the solver loop back to the formalizer. Validates every
field at construction time and renders a safe, deterministic prompt
fragment per signal kind.

Runtime enforcement:
- ``RefinementSignal.kind`` must be one of ``VALID_KINDS``.
- Exactly one companion field must be populated per kind.
- ``ToolFailure.tool`` / ``reason`` / ``operation`` must be members
  of their respective closed enumerations.
- Every interpolated string field is run through
  ``_safe_prompt_atom``, which rejects non-strings, over-long values,
  and any control character.
- ``CoverageReport`` counts must be of exact type ``int``.
- ``render()`` uses explicit ``raise`` statements so validation
  cannot be disabled by ``python -O``.

Trust model: the constructor is the runtime safety boundary, but the
*provenance* discipline (constructing only inside the solver loop
from trusted tool-adapter outcomes) is convention, not enforced at
the type level. Together these bound the residual-risk surface to
adversarial cell values that would still fail length/charset checks
or prompt-injection payloads in fields whose values originate
upstream of the table adapter.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Closed enumerations — runtime-enforced by ``_require_member``
# ---------------------------------------------------------------------------

# The three actionable refinement signals. Any other abstention cause
# — kernel rejection, solver timeout, stagnation, parse errors — is
# non-actionable at the formalizer level and MUST NOT construct a
# ``RefinementSignal``.
VALID_KINDS: frozenset[str] = frozenset({
    "TOOL_FAILURE",
    "COVERAGE_GAP",
    "DICTIONARY_MISMATCH",
})

VALID_TOOLS: frozenset[str] = frozenset({"stats", "lookup"})

VALID_REASONS: frozenset[str] = frozenset({
    "column_not_found",
    "column_ambiguous",
    "no_numeric_values",
    "entity_not_found",
    "entity_ambiguous",
})

# A closed list of operations the stats/lookup dispatch is allowed
# to report. Matches the ``operation`` arg accepted by the stats
# adapter. Adding a new operation requires touching this file and
# the dispatch code that constructs a ToolFailure with that
# operation; this is the intended audit surface.
#
# 'count' is kept here for backward-compat audit only: the live
# dispatcher's REMOVED_OPS gate hard-rejects it before any
# ToolFailure is constructed.
VALID_OPERATIONS: frozenset[str] = frozenset({
    "lookup",
    "sum",
    "mean",
    "count",  # legacy — not reachable in live dispatch (see REMOVED_OPS)
    "count_where",
    "count_distinct",
    "max",
    "min",
    "median",
    "argmax",
    "argmin",
    "top_n",
    "topk_sum",
    "topk_mean",
    "topk_sum_by",
    "topk_mean_by",
    "bottomk_sum",
    "bottomk_mean",
    "computed_argmax",
    "computed_argmin",
    "lag_diff",
    "pct_change",
})

# Re-export ``Literal`` types for static type-checking; they are NOT
# runtime guards. Use ``_require_member`` at construction time to
# enforce membership.
RefinementKind = Literal[
    "TOOL_FAILURE", "COVERAGE_GAP", "DICTIONARY_MISMATCH",
]
ToolFailureReason = Literal[
    "column_not_found",
    "column_ambiguous",
    "no_numeric_values",
    "entity_not_found",
    "entity_ambiguous",
]

# ---------------------------------------------------------------------------
# Primitive sanitizers
# ---------------------------------------------------------------------------

# Allowed characters for interpolated atoms: printable ASCII plus
# Unicode letters/marks/numbers/punctuation/symbols and normal space
# separators (category Zs), plus the explicit extras below
# (guillemets, curly quotes). This admits legitimate column names
# like "boiling points in ˚C" (U+02DA, category Sk) while rejecting
# the categories an attacker could use to break prompt integrity:
# controls, format controls (bidi overrides, zero-width joiners,
# BOM), surrogates, private-use, unassigned codepoints, and
# line/paragraph separators.
_EXTRA_ALLOWED_CHARS: frozenset[str] = frozenset({"«", "»", "\u2019", "\u2018"})

_MAX_COLUMN_LEN = 200
_MAX_IDENT_LEN = 200
_MAX_OPERATION_LEN = 32
_MAX_SAMPLE_LEN = 200
_MAX_SAMPLE_COUNT = 5


def _is_allowed_char(ch: str) -> bool:
    """Return True if ``ch`` is safe for direct prompt interpolation.

    Allowed:
    - printable ASCII (0x20..0x7E)
    - Unicode categories L* (letters), M* (combining marks), N*
      (numbers), P* (punctuation), S* (symbols), Zs (space separators)
    - the explicit extras in ``_EXTRA_ALLOWED_CHARS`` (guillemets,
      curly quotes)

    Rejected:
    - any C{c,f,s,o,n} character (controls, format, surrogates,
      private-use, unassigned) — blocks \\n \\t \\r bidi overrides,
      zero-width joiners/spaces, BOM, DEL, etc.
    - Zl (line separator) and Zp (paragraph separator)
    """
    if ch in _EXTRA_ALLOWED_CHARS:
        return True
    o = ord(ch)
    # Fast path: printable ASCII.
    if 0x20 <= o <= 0x7E:
        return True
    # Explicit reject for control range + DEL.
    if o < 0x20 or o == 0x7F:
        return False
    import unicodedata
    cat = unicodedata.category(ch)
    # Reject anything that could break prompt structure or slip past
    # visible sanitization.
    if cat in ("Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"):
        return False
    # Allow visible printable Unicode.
    if cat[0] in ("L", "M", "N", "P", "S"):
        return True
    if cat == "Zs":
        return True
    return False


def _safe_prompt_atom(value: object, field_name: str, max_len: int) -> str:
    """Validate and return a string safe for direct prompt interpolation.

    Rejects:
    - non-``str`` types (including ``bytes``, ``int``, subclasses)
    - strings longer than ``max_len``
    - strings containing any control character (CR, LF, TAB, VT, FF,
      NUL, BEL, ESC, DEL, etc.)
    - strings containing any character not in the printable ASCII
      range or the small allowed Unicode set (guillemets, curly
      quotes)

    Passed values are returned unchanged (no escaping). The purpose is
    to ensure that whatever reaches the prompt is a short, visible
    ASCII/guillemet string — so that an attacker cannot inject newlines
    that could be interpreted as prompt section breaks, or control
    characters that could confuse downstream tooling.
    """
    if type(value) is not str:
        raise ValueError(
            f"{field_name} must be of type str, got {type(value).__name__}"
        )
    if len(value) > max_len:
        raise ValueError(
            f"{field_name} exceeds max length {max_len}: got {len(value)} chars"
        )
    for i, ch in enumerate(value):
        if not _is_allowed_char(ch):
            raise ValueError(
                f"{field_name} contains disallowed character at index {i}: "
                f"U+{ord(ch):04X}"
            )
    return value


def _require_member(value: object, allowed: frozenset[str], field_name: str) -> str:
    """Validate that ``value`` is a string and is a member of ``allowed``.

    This is the runtime check that backs the ``Literal[...]`` type
    hints. Python does not enforce ``Literal`` at runtime, so every
    closed-enum field on every public dataclass in this module goes
    through this helper in its ``__post_init__``.
    """
    if type(value) is not str:
        raise ValueError(
            f"{field_name} must be of type str, got {type(value).__name__}"
        )
    if value not in allowed:
        raise ValueError(
            f"{field_name}={value!r} not in allowed set {sorted(allowed)!r}"
        )
    return value


def _require_plain_int(value: object, field_name: str) -> int:
    """Validate that ``value`` is a plain, non-negative ``int``.

    Rejects ``bool`` (which is an ``int`` subclass in Python), ``float``,
    numpy ints, and anything else that is not exactly ``int``.
    """
    if type(value) is not int:
        raise ValueError(
            f"{field_name} must be of exact type int, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value}")
    return value


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolFailure:
    """A sanitized report of a trusted-tool call that failed."""
    tool: str                      # validated against VALID_TOOLS
    operation: str                 # validated against VALID_OPERATIONS
    reason: str                    # validated against VALID_REASONS
    column: str | None = None      # None for entity-level failures; else _safe_prompt_atom

    def __post_init__(self) -> None:
        _require_member(self.tool, VALID_TOOLS, "ToolFailure.tool")
        _require_member(self.operation, VALID_OPERATIONS, "ToolFailure.operation")
        _require_member(self.reason, VALID_REASONS, "ToolFailure.reason")
        if self.operation != self.operation.strip():
            # Redundant with VALID_OPERATIONS check but keeps future
            # additions honest.
            raise ValueError("ToolFailure.operation must not have leading/trailing whitespace")
        if self.column is not None:
            _safe_prompt_atom(self.column, "ToolFailure.column", _MAX_COLUMN_LEN)


@dataclass(frozen=True)
class CoverageReport:
    """Row-level coverage metadata for an aggregation tool call.

    ``rows_eligible`` counts rows satisfying the ``Cond`` used in
    the call. ``rows_used`` is the subset that had a parseable
    aggregate value. ``sampled_skipped_values`` contains up to 5
    raw cell values from skipped rows, each validated by
    ``_safe_prompt_atom``.

    A stats evidence item is full-coverage iff
    ``skipped_row_count == 0``.
    """
    column: str
    rows_eligible: int
    rows_used: int
    skipped_row_count: int
    sampled_skipped_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Canonicalize column whitespace before the safety check so column
        # names containing '\n' (TableBench tables sometimes wrap header
        # labels with line breaks, e.g. '% of\nparty list votes') do not
        # crash CoverageReport. The safety property (no control chars in
        # final field) still holds because we collapse all whitespace runs
        # to a single space.
        if isinstance(self.column, str) and any(c in self.column for c in "\n\r\t"):
            import re as _re
            object.__setattr__(self, "column", _re.sub(r"\s+", " ", self.column).strip())
        _safe_prompt_atom(self.column, "CoverageReport.column", _MAX_COLUMN_LEN)
        _require_plain_int(self.rows_eligible, "CoverageReport.rows_eligible")
        _require_plain_int(self.rows_used, "CoverageReport.rows_used")
        _require_plain_int(self.skipped_row_count, "CoverageReport.skipped_row_count")
        if self.rows_used + self.skipped_row_count != self.rows_eligible:
            raise ValueError(
                f"CoverageReport invariant violated: "
                f"rows_used ({self.rows_used}) + skipped_row_count "
                f"({self.skipped_row_count}) != rows_eligible ({self.rows_eligible})"
            )
        # Require sampled_skipped_values to be a tuple (not a list),
        # then sanitize every element.
        if type(self.sampled_skipped_values) is not tuple:
            raise ValueError(
                f"CoverageReport.sampled_skipped_values must be a tuple, got "
                f"{type(self.sampled_skipped_values).__name__}"
            )
        if len(self.sampled_skipped_values) > _MAX_SAMPLE_COUNT:
            raise ValueError(
                f"CoverageReport.sampled_skipped_values capped at {_MAX_SAMPLE_COUNT}, "
                f"got {len(self.sampled_skipped_values)}"
            )
        for i, sample in enumerate(self.sampled_skipped_values):
            _safe_prompt_atom(
                sample,
                f"CoverageReport.sampled_skipped_values[{i}]",
                _MAX_SAMPLE_LEN,
            )

    @property
    def is_partial(self) -> bool:
        """True iff this aggregation had eligible rows it could not use."""
        return self.skipped_row_count > 0


@dataclass(frozen=True)
class RefinementSignal:
    """The ONLY structure a variant-F retry may consume."""
    kind: str                                  # validated against VALID_KINDS
    tool_failure: ToolFailure | None = None
    coverage: CoverageReport | None = None
    missing_identifier: str | None = None      # validated via _safe_prompt_atom

    def __post_init__(self) -> None:
        _require_member(self.kind, VALID_KINDS, "RefinementSignal.kind")

        populated = sum(
            1
            for x in (self.tool_failure, self.coverage, self.missing_identifier)
            if x is not None
        )
        if populated != 1:
            raise ValueError(
                f"RefinementSignal must have exactly one of tool_failure, "
                f"coverage, missing_identifier set; got {populated}"
            )
        expected_field = {
            "TOOL_FAILURE": self.tool_failure is not None,
            "COVERAGE_GAP": self.coverage is not None,
            "DICTIONARY_MISMATCH": self.missing_identifier is not None,
        }
        if not expected_field[self.kind]:
            raise ValueError(
                f"RefinementSignal kind={self.kind!r} does not match populated field"
            )
        if self.missing_identifier is not None:
            _safe_prompt_atom(
                self.missing_identifier,
                "RefinementSignal.missing_identifier",
                _MAX_IDENT_LEN,
            )

    def render(self) -> str:
        """Render this signal into a deterministic prompt fragment.

        Uses a fixed template per ``kind``. Because every interpolated
        field was validated by ``_safe_prompt_atom`` or
        ``_require_member`` at construction time, the rendered output
        cannot contain newlines, tabs, control characters, or arbitrary
        Unicode. Uses explicit ``raise`` rather than ``assert`` so
        validation is not elided by ``python -O``.
        """
        if self.kind == "TOOL_FAILURE":
            tf = self.tool_failure
            if tf is None:
                raise RuntimeError("TOOL_FAILURE signal missing tool_failure field")
            col_fragment = f" on column '{tf.column}'" if tf.column else ""
            return (
                "RETRY HINT — tool failure:\n"
                f"  Your previous formalization required a {tf.tool} {tf.operation} call{col_fragment}, "
                f"which failed with reason '{tf.reason}'.\n"
                "  Consider an alternative formalization at a lower abstraction level "
                "(e.g., individual HasProperty lookups + explicit arithmetic) "
                "or adjust the Cond filter so that the failing rows are excluded.\n"
                "  This is a TOOL-capability signal only. No information about the "
                "solver's proof attempts is available."
            )

        if self.kind == "COVERAGE_GAP":
            cov = self.coverage
            if cov is None:
                raise RuntimeError("COVERAGE_GAP signal missing coverage field")
            samples = ", ".join(f"'{v}'" for v in cov.sampled_skipped_values[:5])
            samples_line = f"  Sampled skipped values: {samples}\n" if samples else ""
            return (
                "RETRY HINT — coverage gap:\n"
                f"  Your previous formalization asked for an aggregation over column "
                f"'{cov.column}' that covered {cov.rows_used}/{cov.rows_eligible} of "
                f"the Cond-eligible rows. {cov.skipped_row_count} row(s) were skipped "
                "because their values could not be parsed.\n"
                f"{samples_line}"
                "  Consider adding an explicit Cond filter to exclude the unparseable "
                "rows, or producing an alternative formalization that handles them. "
                "This is a COVERAGE signal only. No proof-level information is "
                "available."
            )

        if self.kind == "DICTIONARY_MISMATCH":
            ident = self.missing_identifier
            if ident is None:
                raise RuntimeError(
                    "DICTIONARY_MISMATCH signal missing missing_identifier field"
                )
            return (
                "RETRY HINT — dictionary mismatch:\n"
                f"  Your previous formalization referenced identifier '{ident}' which "
                "was not resolvable in the solver's view of the dictionary. "
                "Reuse only identifiers that are already defined as Property or "
                "Entity entries, or add a new_entries declaration in your output "
                "before referencing a new identifier.\n"
                "  This is a DICTIONARY signal only. No proof-level information is "
                "available."
            )

        # Unreachable: __post_init__ enforces ``kind in VALID_KINDS``.
        raise RuntimeError(f"unhandled RefinementKind in render(): {self.kind}")
