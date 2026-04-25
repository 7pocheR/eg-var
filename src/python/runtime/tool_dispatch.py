"""Shared per-table menu enforcement.

Single source of truth for `is_action_in_per_table_menu` and
`off_menu_message`, used by both the live solver loop in `pipeline` and
the baseline agent.

Per-action-class enforcement rules:

  stats action:
    Resolve via pipeline._resolve_tool_instance (per-op-kind matching
    with column canonicalization). On-menu iff the resolver returns a
    non-None instance.

  lookup action:
    On-menu iff the table has at least one cell_lookup tool. The
    adapter validates the actual (entity, property) pair separately
    (adapter NOT FOUND is not an enforcement reject).

  lookup_batch action:
    Recurse per entry; reject whole batch with first off-menu entry's
    index (atomic from solver's perspective). Per-entry adapter
    NOT FOUND failures stay per-entry, not enforcement reject.

  other action shapes (give_up, ask_common_sense, eval, etc.):
    Out of scope for menu enforcement; accept.
"""
from __future__ import annotations
from typing import Any


def is_action_in_per_table_menu(
    table_id: str, action: dict
) -> tuple[bool, str | None]:
    """Policy oracle: is `action` on the per-table menu for `table_id`?

    Returns (on_menu, reason).
      - on_menu=True  -> reason=None
      - on_menu=False -> reason is a short human-readable string
        explaining why; goes into the rejection message.
    """
    a = action.get("action", "")

    if a == "stats":
        op = action.get("operation", "")
        col = action.get("column", "")
        # Reuse the live runtime's resolver so canonicalization and
        # column-matching rules stay in lockstep across both paths.
        from pipeline import _resolve_tool_instance
        resolved = _resolve_tool_instance(table_id, op, col)
        if resolved is None:
            return False, (f"stats operation {op!r} on column {col!r} does not match "
                           f"any per-table tool instance")
        return True, None

    if a == "lookup":
        # Universal cell_lookup design: a single cell_lookup tool
        # admits any measure_col, so the enforcement check is just
        # "does the table have any cell_lookup tool?" The adapter
        # validates the specific (entity, property) pair separately.
        from pipeline import _load_per_table_formalizations
        tf = _load_per_table_formalizations().get(table_id, {}) or {}
        has_cl = any(t.get("library_ref") == "cell_lookup"
                     for t in tf.get("tools", []) or [])
        if not has_cl:
            return False, ("table has no cell_lookup tool in its per-table menu "
                           "(structural — this table doesn't admit single-entry lookup)")
        return True, None

    if a == "lookup_batch":
        lookups = action.get("lookups")
        if not isinstance(lookups, list) or not lookups:
            return False, "lookups list is empty or not a list"
        for i, L in enumerate(lookups):
            if not isinstance(L, dict):
                return False, f"entry [{i}] is not a dict"
            # Recurse via single-lookup check
            inner = {"action": "lookup", **L}
            ok, reason = is_action_in_per_table_menu(table_id, inner)
            if not ok:
                return False, f"entry [{i}]: {reason}"
        return True, None

    # Other action types (give_up, ask_common_sense, eval, code, etc.):
    # out of scope for menu enforcement. Default accept.
    return True, None


def off_menu_message(table_id: str, action: dict) -> str:
    """Canonical rejection text for off-menu calls.

    Both the live solver loop (under enforce_menu=True) and the
    baseline agent emit this text verbatim.
    """
    on_menu, reason = is_action_in_per_table_menu(table_id, action)
    if on_menu:
        # Defensive: callers should only invoke this for off-menu actions.
        return (f"INTERNAL ERROR: off_menu_message called for an on-menu action "
                f"(table_id={table_id!r}, action={action!r}). The caller's "
                f"is_action_in_per_table_menu check is inconsistent with this call.")

    a = action.get("action", "?")
    return (f"ERROR: action {a!r} is OFF the per-table menu for this table "
            f"(enforce_menu=True rejects this). Reason: {reason}. "
            f"Either choose a different action / column / property that IS in the "
            f"per-table menu, or accept that this table's curator commitment "
            f"genuinely does not admit the operation you tried.")
