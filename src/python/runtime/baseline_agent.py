"""D.I baseline runner: LLM-with-tools, no kernel.

Production multi-turn agent for the D.I-strict and D.I-loose
variants of the baseline ladder. Mirrors the EG-VAR solver loop's
tool surface (same per-table menu via tool_dispatch); the
differentiator is kernel-off (no Lean compilation, no L1/L2 lifts,
no formalization). The agent emits a free-text answer; the runner
judges via answers_match.

Trace contract: tool-call steps carry `raw_action` (full JSON) and
structured `outcome` (success, n_evidence_added, optional
per_entry_status). Non-tool steps omit those fields (presence-test
convention) so downstream consumers can read this trace and the
EG-VAR solver trace identically.
"""
from __future__ import annotations
import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from table_adapter import TableData


@dataclass
class BaselineResult:
    """Result of a single baseline_agent run on one (claim, table)."""
    answer: str | None
    steps: list[dict] = field(default_factory=list)
    n_tool_calls: int = 0
    elapsed_s: float = 0.0
    terminated_reason: str = "exhausted"  # answered | max_steps | stagnation | exception | parse_storm
    # Diagnostic buckets:
    n_off_menu_calls: int = 0
    n_invalid_json: int = 0
    n_unknown_actions: int = 0
    n_tool_errors: int = 0
    answered_without_tool: bool = False
    last_observation: str | None = None
    final_action: dict | None = None
    error: str | None = None
    enforce_menu: bool = True


# -------- prompt construction --------

_ACTION_DOCS_VERBATIM = """\
Action shapes (output exactly ONE JSON object per turn — nothing else;
do NOT wrap in markdown fences; do NOT simulate the runtime's feedback):

{"action": "lookup", "entity": "<name as it appears in the table>", "property": "<column name as it appears in the header>"}
  -> Use the ORIGINAL name from the table, not a normalized form.

{"action": "lookup_batch", "lookups": [{"entity": "...", "property": "..."}, ...]}
  -> Multiple lookups in one step. Same rules as single lookup.
  -> Use this when you need many values to save round-trips.

{"action": "stats", "operation": "<op>", "column": "<col>", ...optional params...}
  Operations: sum, max, min, mean, count_distinct, count_where, argmax, argmin,
              topk_sum, topk_mean, bottomk_sum, bottomk_mean,
              topk_sum_by, topk_mean_by (require `rank_by`).

  REMOVED operations (will be HARD-REJECTED with the canonical replacement):
    count, count_rows → use `count_where` with {op:'all'} (drop-in equivalent).
    median           → NO direct replacement; median ≠ mean.
    top_n            → topk_sum/topk_mean/bottomk_sum/bottomk_mean ONLY for
                       aggregated VALUE; legacy top_n returned an entity LIST.
    computed_argmax, computed_argmin, lag_diff, pct_change → reformulate
                       via multiple lookups + arithmetic.

  Optional params:
    - condition: JSON Cond filter applied before aggregation. Format:
        {"op": "<eq|gt|lt|gte|lte|contains>", "property": "<col>", "value": <literal>}
        Compound: {"op": "and"|"or", "left": <cond>, "right": <cond>}
                  {"op": "not", "inner": <cond>}
        IMPORTANT: the `property` field is REQUIRED for non-trivial ops
        (it names the column the FILTER is on, often same as `column`
        but explicit). Omitting it produces wrong results without an error.

        NEGATION: there is NO `neq` / `not_eq` / `ne` operator. To
        express `!=`, wrap an `eq` in `not`:
          {"op": "not", "inner": {"op": "eq", "property": "X", "value": Y}}

    - n: integer (k) for topk_*/bottomk_*. Runtime fails closed if k > eligible rows.
    - rank_by: REQUIRED for topk_*_by (column to rank by, DISTINCT from
               `column` — the aggregation target). Non-_by topk variants
               REJECT rank_by; use _by suffix if you need a separate rank column.
    - entity_column: identity column for argmax/argmin (defaults to per-table identity).

  count_where: counts ROWS satisfying the condition. The `column` argument
    labels the result for traceability but does NOT affect which rows count
    — the condition alone decides.

  argmax/argmin: returns ENTITY NAME + value (per the per-table identity column).
    For conditional argmax, pass the `condition` filter.

  Examples:
    count rows with age > 30:
      {"action": "stats", "operation": "count_where", "column": "age",
       "condition": {"op": "gt", "property": "age", "value": 30}}
    sum gdp where un budget > 2:
      {"action": "stats", "operation": "sum", "column": "gdp",
       "condition": {"op": "gt", "property": "un budget", "value": 2}}
    argmax of PoW excluding the 'Total' footer row:
      {"action": "stats", "operation": "argmax", "column": "PoW",
       "condition": {"op": "not", "inner": {"op": "eq", "property": "Place", "value": "Total"}}}
    topk_sum_by — sum of top-3 GDPs ranked by 2010 population:
      {"action": "stats", "operation": "topk_sum_by", "column": "gdp",
       "rank_by": "2010 population", "n": 3}

{"action": "answer", "value": "<bare answer — see ANSWER FORMAT below>"}
  -> Terminates the loop. The judge does exact + numeric-equivalence matching.
"""


_ANSWER_FORMAT_INSTRUCTION = """\
ANSWER FORMAT (load-bearing — the judge is strict):

When you emit `{"action": "answer", "value": ...}`, the `value` MUST be
ONLY the bare answer:
  - For numeric answers: just the number, no commas, no units.
    e.g. `4351` (NOT `"4,351 thousand"`).
  - For entity/text answers: just the entity name as it appears in the table.
    e.g. `oman` (NOT `"Oman has the highest..."`).
  - For date/time answers: the date or time as it appears in the table.

Do NOT wrap the answer in a sentence. Do NOT add explanations.
Surrounding prose will be marked WRONG even when your underlying answer
is right.
"""


def _build_system_prompt() -> str:
    return (
        "You are a question-answering agent for tabular data. You see the "
        "table directly in the prompt AND you have access to a per-table "
        "tool menu. You may either:\n"
        "  - read the table directly to answer simple lookups, OR\n"
        "  - call a tool when computation is hard (aggregations, top-k, filtering).\n\n"
        + _ANSWER_FORMAT_INSTRUCTION
        + "\n"
        + _ACTION_DOCS_VERBATIM
        + "\n"
        + "When you have enough information, emit the `answer` action."
    )


def _build_user_prompt(claim: str, table_text: str, tool_menu: str,
                        prior_steps: list[dict]) -> str:
    parts = [
        f"QUESTION: {claim}",
        "",
        f"TABLE:",
        table_text,
        "",
        f"TOOL MENU (per-table, available operations):",
        tool_menu,
    ]
    if prior_steps:
        parts.append("")
        parts.append("PRIOR ACTIONS THIS TURN (oldest first):")
        for i, s in enumerate(prior_steps[-12:]):  # last 12 to bound context
            ra = s.get("raw_action")
            obs = s.get("feedback") or s.get("obs") or "(no obs)"
            if ra:
                parts.append(f"  [step {i}] {json.dumps(ra)[:200]}")
                parts.append(f"           → {str(obs)[:200]}")
            else:
                parts.append(f"  [step {i}] {s.get('action', '?')}: {str(obs)[:120]}")
    parts.append("")
    parts.append("Emit your next action JSON.")
    return "\n".join(parts)


# -------- robust JSON action parser (handles multi-block + fences) --------

def _parse_action(text: str) -> dict | None:
    """Extract the FIRST valid JSON object from solver text. Tolerates
    markdown fences and multi-block responses. Brace-matched scan
    (greedy regex would span unrelated braces and fail to parse)."""
    text = text.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


# -------- action execution --------

def _execute_action(action: dict, table: TableData,
                     enforce_menu: bool = True) -> tuple[str, dict, str]:
    """Execute one action. Returns (feedback_str, outcome_dict, kind).
    kind in {tool, answer, parse_error, unknown}.

    Reuses pipeline / tool_dispatch helpers as the single source of
    truth so trace shapes align with run_solver_loop.
    """
    from pipeline import (
        _execute_lookup, _resolve_lookup_identity, _resolve_tool_instance,
        REMOVED_OPS, removed_op_message, check_topk_rank_by, TOPK_ALL_OPS,
        PER_ENTRY_STATUS_OK, PER_ENTRY_STATUS_NOT_FOUND,
        PER_ENTRY_STATUS_OFF_MENU, PER_ENTRY_STATUS_MALFORMED,
    )
    from tool_dispatch import is_action_in_per_table_menu, off_menu_message
    from table_adapter import (
        stats_count_distinct, stats_sum, stats_max, stats_min, stats_mean,
        stats_argmax, stats_argmin, stats_agg_where, stats_topk_where,
        stats_count_where,
    )

    a_type = action.get("action", "")

    # Answer action terminates the loop.
    if a_type == "answer":
        val = str(action.get("value", ""))
        return val, {"success": True, "n_evidence_added": 0}, "answer"

    # Menu enforcement check (D.I-strict variant).
    if enforce_menu and a_type in ("stats", "lookup", "lookup_batch"):
        on_menu, _reason = is_action_in_per_table_menu(table.table_id, action)
        if not on_menu:
            msg = off_menu_message(table.table_id, action)
            outcome = {"success": False, "n_evidence_added": 0}
            if a_type == "lookup_batch":
                n = max(len(action.get("lookups", []) or []), 1)
                outcome["per_entry_status"] = [PER_ENTRY_STATUS_OFF_MENU] * n
                outcome["n_total"] = n
                outcome["n_succeeded"] = 0
            return msg, outcome, "tool"

    if a_type == "lookup":
        ent = action.get("entity", "")
        prop = action.get("property", "")
        id_col = _resolve_lookup_identity(table.table_id)
        try:
            r = _execute_lookup(table, ent, prop, identity_col=id_col)
        except Exception as e:
            # _execute_lookup normally fails closed to None; this
            # branch labels rare adapter raises with the shared
            # vocabulary's ADAPTER_ERROR rather than a bare ERROR.
            return (f"ERROR ({PER_ENTRY_STATUS_ADAPTER_ERROR}): "
                    f"lookup raised {type(e).__name__}: {e}",
                    {"success": False, "n_evidence_added": 0}, "tool")
        if r is None:
            return (f"NOT FOUND: entity={ent!r} property={prop!r}",
                    {"success": False, "n_evidence_added": 0}, "tool")
        return (f"OK: {ent}.{r['property']} = {r['raw_value']}",
                {"success": True, "n_evidence_added": 1}, "tool")

    if a_type == "lookup_batch":
        lookups = action.get("lookups")
        if not isinstance(lookups, list) or not lookups:
            return ("ERROR: 'lookups' list is empty",
                    {"success": False, "n_evidence_added": 0,
                     "per_entry_status": [PER_ENTRY_STATUS_MALFORMED],
                     "n_total": 1, "n_succeeded": 0},
                    "tool")
        out = []
        per_entry: list[str] = []
        n_added = 0
        id_col = _resolve_lookup_identity(table.table_id)
        for i, L in enumerate(lookups):
            if not isinstance(L, dict):
                out.append(f"  [{i}] MALFORMED: entry not a dict")
                per_entry.append(PER_ENTRY_STATUS_MALFORMED)
                continue
            ent, prop = L.get("entity", ""), L.get("property", "")
            try:
                r = _execute_lookup(table, ent, prop, identity_col=id_col)
            except Exception as e:
                # ADAPTER_ERROR is distinct from NOT_FOUND in the
                # shared per-entry status vocabulary.
                out.append(f"  [{i}] ADAPTER_ERROR: {type(e).__name__}: {e}")
                per_entry.append(PER_ENTRY_STATUS_ADAPTER_ERROR)
                continue
            if r is None:
                out.append(f"  [{i}] NOT FOUND: {ent!r}.{prop!r}")
                per_entry.append(PER_ENTRY_STATUS_NOT_FOUND)
            else:
                out.append(f"  [{i}] OK: {ent}.{r['property']} = {r['raw_value']}")
                per_entry.append(PER_ENTRY_STATUS_OK)
                n_added += 1
        n_total = len(per_entry)
        n_succ = sum(1 for s in per_entry if s == PER_ENTRY_STATUS_OK)
        feedback = (f"BATCH ({n_total} lookups, {n_succ} succeeded):\n"
                    + "\n".join(out))
        return (feedback,
                {"success": n_succ > 0, "n_evidence_added": n_added,
                 "n_total": n_total, "n_succeeded": n_succ,
                 "per_entry_status": per_entry},
                "tool")

    if a_type == "stats":
        op = action.get("operation", "")
        col = action.get("column", "")
        cond = action.get("condition")
        if op in REMOVED_OPS:
            return (removed_op_message(op),
                    {"success": False, "n_evidence_added": 0}, "tool")
        resolved = _resolve_tool_instance(table.table_id, op, col)
        parse_mode = resolved["config"].get("parse_mode", "strict") if resolved else "strict"
        entity_col = action.get("entity_column")
        if resolved and resolved.get("identity_column"):
            entity_col = resolved["identity_column"]
        try:
            if op == "count_where":
                sr = stats_count_where(table, col, cond or {"op": "all"})
            elif op in TOPK_ALL_OPS:
                err, rank_by = check_topk_rank_by(op, action.get("rank_by"))
                if err is not None:
                    return (err, {"success": False, "n_evidence_added": 0}, "tool")
                n_val = action.get("n", 5)
                sr = stats_topk_where(table, op, col, n_val, cond, rank_by, parse_mode=parse_mode)
            elif cond and op in ("sum", "max", "min", "mean"):
                sr = stats_agg_where(table, op, col, cond, parse_mode=parse_mode)
            elif op == "mean":
                sr = stats_agg_where(table, "mean", col, {"op": "all"}, parse_mode=parse_mode)
            elif op == "count_distinct":
                sr = stats_count_distinct(table, col)
            elif op == "sum":
                sr = stats_sum(table, col, parse_mode=parse_mode)
            elif op == "max":
                sr = stats_max(table, col, parse_mode=parse_mode)
            elif op == "min":
                sr = stats_min(table, col, parse_mode=parse_mode)
            elif op == "argmax":
                sr = stats_argmax(table, col, entity_col, parse_mode=parse_mode, condition=cond)
            elif op == "argmin":
                sr = stats_argmin(table, col, entity_col, parse_mode=parse_mode, condition=cond)
            else:
                return (f"ERROR: unknown operation {op!r}",
                        {"success": False, "n_evidence_added": 0}, "tool")
        except Exception as e:
            return (f"ERROR: stats raised {type(e).__name__}: {e}",
                    {"success": False, "n_evidence_added": 0}, "tool")
        if not sr.success:
            return (f"ERROR: {sr.error}",
                    {"success": False, "n_evidence_added": 0}, "tool")
        return (f"OK: {op}({col}) = {sr.value}. {sr.details or ''}",
                {"success": True, "n_evidence_added": 1}, "tool")

    return (f"ERROR: unknown action {a_type!r}",
            {"success": False, "n_evidence_added": 0}, "unknown")


# -------- main loop --------

def run_baseline_agent(
    *, claim: str, table: TableData, tf: dict | None,
    backend, max_steps: int = 20, stagnation_threshold: int = 6,
    enforce_menu: bool = True,
) -> BaselineResult:
    """Multi-turn baseline agent loop. Returns BaselineResult.

    Args:
      claim: natural-language question.
      table: TableData (full table visible in prompt + tools).
      tf: per-table formalization dict (for the tool menu). May be None
          (no menu rendered; rare for Tier 1).
      backend: LLMBackend with .call(role, system, user) → str.
      max_steps: hard cap on solver turns.
      stagnation_threshold: give up after N consecutive non-progress steps.
      enforce_menu: True for D.I-strict; False for D.I-loose.
    """
    from pipeline import _render_table, _summarize_table_tools

    table_text = _render_table(table)
    tool_menu = "\n".join(_summarize_table_tools(tf)) if tf else "(no per-table menu)"
    sys_prompt = _build_system_prompt()

    res = BaselineResult(answer=None, enforce_menu=enforce_menu)
    consecutive_failures = 0
    consecutive_parse_errors = 0

    t0 = time.time()
    for step_no in range(max_steps):
        if consecutive_failures >= stagnation_threshold:
            res.terminated_reason = "stagnation"
            res.steps.append({"action": "stagnation",
                              "feedback": f"give up after {consecutive_failures} consecutive failures"})
            break
        if consecutive_parse_errors >= 3:
            res.terminated_reason = "parse_storm"
            res.steps.append({"action": "parse_storm",
                              "feedback": "give up after 3 consecutive parse errors"})
            break

        user_prompt = _build_user_prompt(claim, table_text, tool_menu, res.steps)
        try:
            raw = backend.call("solver", sys_prompt, user_prompt)
        except Exception as e:
            res.terminated_reason = "exception"
            res.error = f"backend: {type(e).__name__}: {e}"
            res.steps.append({"action": "exception", "feedback": res.error})
            break

        action = _parse_action(raw)
        if action is None:
            res.n_invalid_json += 1
            consecutive_parse_errors += 1
            res.steps.append({"action": "parse_error",
                              "feedback": "ERROR: could not parse JSON action.",
                              "raw_response": raw[:300]})
            continue
        consecutive_parse_errors = 0

        feedback, outcome, kind = _execute_action(action, table, enforce_menu=enforce_menu)
        res.last_observation = feedback[:200]
        res.final_action = action

        if kind == "answer":
            res.answer = feedback
            res.terminated_reason = "answered"
            res.answered_without_tool = (res.n_tool_calls == 0)
            # Presence-test contract: non-tool steps omit raw_action /
            # outcome so tool-step diffs stay clean. The action JSON
            # is preserved on the result via final_action.
            res.steps.append({"action": "answer", "feedback": "(final answer)",
                              "value": str(action.get("value", ""))})
            break

        if kind == "tool":
            res.n_tool_calls += 1
            # Separate off_menu from generic tool errors so the
            # headline categorization stays meaningful; per-rejection
            # subclass detail (REMOVED_OPS, rank_by, NOT FOUND) lives
            # in feedback.
            if not outcome.get("success"):
                if "OFF the per-table menu" in feedback:
                    res.n_off_menu_calls += 1
                else:
                    res.n_tool_errors += 1
            if outcome.get("success"):
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            res.steps.append({"action": action.get("action", ""),
                              "raw_action": action, "outcome": outcome,
                              "feedback": feedback[:500]})
        else:
            # Unknown action: not a tool call, omit raw_action /
            # outcome to honor the presence-test contract. The
            # action JSON is preserved under result.final_action.
            res.n_unknown_actions += 1
            consecutive_failures += 1
            res.steps.append({"action": "unknown", "feedback": feedback,
                              "attempted_action_kind": action.get("action", "?")})

    if res.answer is None and res.terminated_reason == "exhausted":
        res.terminated_reason = "max_steps"
    res.elapsed_s = round(time.time() - t0, 2)
    return res
