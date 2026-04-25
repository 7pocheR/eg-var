"""
EG-VAR Dictionary: rich accumulated formal language specification.

Event-sourced: each entry has an immutable base; enrichments are
appended. Shared across all memoryless formalizer instances.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import json

Source = Literal["core", "schema", "tool_confirmed", "formalizer_proposed", "interpreted"]
GradeCap = Literal["no_cap", "Supported", "Plausible"]
EntryType = Literal["Predicate", "Entity", "Property", "Value", "Relation", "Convention"]


@dataclass
class Alias:
    surface_form: str
    context: str
    source: Source
    grade_impact: GradeCap


@dataclass
class DictEntry:
    name: str
    type: EntryType
    description: str
    formalization: str
    aliases: list[Alias] = field(default_factory=list)
    disambiguation: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)

    def add_alias(self, alias: Alias):
        """Enrich: add a new alias to this entry."""
        self.aliases.append(alias)

    def add_disambiguation(self, rule: str):
        """Enrich: add a disambiguation note."""
        self.disambiguation.append(rule)

    def best_grade_cap(self) -> GradeCap:
        """The best (least restrictive) grade cap among confirmed aliases."""
        for a in self.aliases:
            if a.grade_impact == "no_cap" and a.source in ("core", "tool_confirmed", "schema"):
                return "no_cap"
        for a in self.aliases:
            if a.grade_impact == "Supported":
                return "Supported"
        return "Plausible"

    def to_prompt_text(self) -> str:
        """Render this entry for inclusion in a formalizer prompt."""
        lines = [
            f"[{self.type}] {self.name}",
            f"  Description: {self.description}",
            f"  Formal: {self.formalization}",
        ]
        if self.aliases:
            alias_strs = [f'"{a.surface_form}" ({a.context}, {a.source})' for a in self.aliases]
            lines.append(f"  Aliases: {'; '.join(alias_strs)}")
        if self.disambiguation:
            lines.append(f"  Disambiguation: {'; '.join(self.disambiguation)}")
        return "\n".join(lines)


class Dictionary:
    """The accumulated formal language dictionary for a EG-VAR session."""

    def __init__(self):
        self.entries: dict[str, DictEntry] = {}
        self._event_log: list[dict] = []  # event-sourced history

    def add_entry(self, entry: DictEntry) -> bool:
        """Add a new entry. Returns False if name already exists (use enrich instead)."""
        if entry.name in self.entries:
            return False
        self.entries[entry.name] = entry
        self._event_log.append({"action": "add", "name": entry.name, "type": entry.type})
        return True

    def enrich(self, name: str, alias: Alias | None = None, disambiguation: str | None = None):
        """Enrich an existing entry with new alias or disambiguation."""
        if name not in self.entries:
            return
        if alias:
            self.entries[name].add_alias(alias)
            self._event_log.append({"action": "add_alias", "name": name, "alias": alias.surface_form})
        if disambiguation:
            self.entries[name].add_disambiguation(disambiguation)
            self._event_log.append({"action": "add_disambiguation", "name": name, "rule": disambiguation})

    def get(self, name: str) -> DictEntry | None:
        return self.entries.get(name)

    def find_by_alias(self, surface_form: str) -> list[DictEntry]:
        """Find entries that have a matching alias (case-insensitive)."""
        surface_lower = surface_form.lower().strip()
        results = []
        for entry in self.entries.values():
            for alias in entry.aliases:
                if alias.surface_form.lower().strip() == surface_lower:
                    results.append(entry)
                    break
        return results

    def to_prompt(self) -> str:
        """Render the full dictionary for a formalizer prompt."""
        sections = {
            "Predicate": [],
            "Entity": [],
            "Property": [],
            "Value": [],
            "Relation": [],
            "Convention": [],
        }
        for entry in self.entries.values():
            sections.setdefault(entry.type, []).append(entry.to_prompt_text())

        lines = ["=== EG-VAR DICTIONARY ===", ""]
        for section, entries in sections.items():
            if entries:
                lines.append(f"--- {section}s ---")
                for e in entries:
                    lines.append(e)
                    lines.append("")
        return "\n".join(lines)

    def entry_count(self) -> int:
        return len(self.entries)

    def to_json(self) -> str:
        """Serialize dictionary to JSON for persistence."""
        data = {
            "entries": {
                name: {
                    "name": e.name,
                    "type": e.type,
                    "description": e.description,
                    "formalization": e.formalization,
                    "aliases": [{"surface_form": a.surface_form, "context": a.context,
                                 "source": a.source, "grade_impact": a.grade_impact}
                                for a in e.aliases],
                    "disambiguation": e.disambiguation,
                    "related": e.related,
                }
                for name, e in self.entries.items()
            },
            "event_log": self._event_log,
        }
        return json.dumps(data, indent=2)


def init_core_dictionary() -> Dictionary:
    """Initialize dictionary with core predicates (fixed, never change)."""
    d = Dictionary()

    d.add_entry(DictEntry(
        name="HasProperty",
        type="Predicate",
        description="Asserts that an entity has a specific value for a property. "
                    "Usage: HasProperty \"<entity>\" \"<property>\" <value>. "
                    "Example: HasProperty \"brazil\" \"points\" (Value.rat (mkRat 15 1)) means 'Brazil has 15 points.'",
        formalization="HasProperty : String → String → Value → WProp  -- per R10 nominalist design",
    ))

    d.add_entry(DictEntry(
        name="Relation",
        type="Predicate",
        description="Asserts a named relationship between two entities. "
                    "Usage: Relation \"<subject>\" <relname> \"<object>\". "
                    "Example: Relation \"brazil\" rel_in_group \"group_a\".",
        formalization="Relation : String → RelName → String → WProp  -- per R10",
    ))

    d.add_entry(DictEntry(
        name="ValueGt",
        type="Predicate",
        description="Greater-than comparison between two numeric values. Kernel-checked.",
        formalization="ValueGt : Value → Value → Prop",
    ))

    d.add_entry(DictEntry(
        name="ValueLt",
        type="Predicate",
        description="Less-than comparison between two numeric values. Kernel-checked.",
        formalization="ValueLt : Value → Value → Prop",
    ))

    d.add_entry(DictEntry(
        name="ValueEq",
        type="Predicate",
        description="Equality between two values. Kernel-checked.",
        formalization="ValueEq : Value → Value → Prop",
    ))

    # Core conventions — advisory encoding standards for formalizers
    d.add_entry(DictEntry(
        name="temporal_encoding",
        type="Convention",
        description="When a column contains dates (e.g., '5 february 1999'), extract the year "
                    "as Value.rat (mkRat year 1) for year-level comparisons. Use full string Value.str for exact "
                    "date matching.\n"
                    "  Example: 'competed between 1961 and 1967' →\n"
                    "    Informal: HasProperty(entity, year, v) AND ValueGte(v, 1961) AND ValueLte(v, 1967)\n"
                    "    Lean type: PProd (Evidence Grade.Verified (HasProperty e year v)) "
                    "(PProd (PLift (ValueGte v (Value.rat (mkRat 1961 1)))) (PLift (ValueLte v (Value.rat (mkRat 1967 1)))))\n"
                    "  Example: 'both recorded on the same date' →\n"
                    "    Informal: HasProperty(e1, date, d1) AND HasProperty(e2, date, d2) AND ValueEq(d1, d2)\n"
                    "    Lean type: PProd (Evidence Grade.Verified (HasProperty e1 date d1)) "
                    "(PProd (Evidence Grade.Verified (HasProperty e2 date d2)) (PLift (ValueEq d1 d2)))",
        formalization="-- convention, no Lean type",
    ))

    d.add_entry(DictEntry(
        name="numeric_formulas",
        type="Convention",
        description="When a cell contains a formula (e.g., '70 + 71 + 68 + 72 = 281'), "
                    "the number after '=' is the canonical numeric value.\n"
                    "  Example: raw cell '70 + 71 + 68 + 72 = 281' → Value.rat (mkRat 281 1)\n"
                    "When a cell contains a compound value (e.g., 'american airlines center 20301'), "
                    "extract the number if the property is about count/attendance.\n"
                    "  Example: raw cell 'us airways center 18422' → Value.rat (mkRat 18422 1) (for attendance)",
        formalization="-- convention, no Lean type",
    ))

    # World-level aggregate predicates (all take Cond for filtering).
    # Per R10: column / entity slots are String literals, not identifier-typed atoms.
    d.add_entry(DictEntry(
        name="SumWhere",
        type="Predicate",
        description="Sum of a column with optional filter.\n"
                    "  SumWhere \"population\" Cond.all (Value.rat (mkRat 5500000 1)) = 'total population is 5.5M'\n"
                    "  SumWhere \"gdp\" (Cond.propGt \"un budget\" (Value.rat (mkRat 2 1))) (Value.rat (mkRat 7700143 1)) = 'GDP where budget>2'",
        formalization="SumWhere : String → Cond → Value → WProp  -- per R10",
    ))

    d.add_entry(DictEntry(
        name="CountWhere",
        type="Predicate",
        description="Count of rows with optional filter.\n"
                    "  CountWhere \"season\" Cond.all (Value.rat (mkRat 10 1)) = 'there are 10 seasons'",
        formalization="CountWhere : String → Cond → Value → WProp  -- per R10",
    ))

    d.add_entry(DictEntry(
        name="MeanWhere",
        type="Predicate",
        description="Mean of a column, exact as Rat. Convenience for SumWhere/CountWhere.\n"
                    "  MeanWhere \"age\" Cond.all (Value.rat (mkRat 463 9)) = 'average age is 463/9'",
        formalization="MeanWhere : String → Cond → Value → WProp  -- per R10",
    ))

    d.add_entry(DictEntry(
        name="ArgmaxWhere",
        type="Predicate",
        description="Entity (named by answer_name) with the maximum value of a column.\n"
                    "  ArgmaxWhere \"population\" Cond.all \"tokyo\" (Value.rat (mkRat 15000000 1)) = 'tokyo has max population'\n"
                    "Per R10.6 this is a level-3 (entity-anchored) predicate; using it requires the per-table\n"
                    "formalization to commit to entity-anchoring (rows-as-entities + identity column).",
        formalization="ArgmaxWhere : String → Cond → String → Value → WProp  -- per R10",
    ))

    d.add_entry(DictEntry(
        name="EntitySum",
        type="Predicate",
        description="Sum of all property values for a single entity.\n"
                    "  EntitySum \"brazil\" (Value.rat (mkRat 100 1)) = 'sum of all brazil values is 100'",
        formalization="EntitySum : String → Value → WProp  -- per R10",
    ))

    d.add_entry(DictEntry(
        name="question_answer_types",
        type="Convention",
        description="For questions, use the appropriate predicate and embed the known answer.\n\n"
                    "  INTEGER SUM: 'What is the total population?' (gold=5500000) →\n"
                    "    lean_goal_type: Evidence Grade.Verified (SumWhere population (Value.rat (mkRat 5500000 1)))\n"
                    "    No Sigma needed — answer is embedded directly.\n\n"
                    "  AVERAGE with decimal: 'What is the average X?' (gold=10.6) →\n"
                    "    lean_goal_type: Σ (s : Int), Σ (c : Int), PProd (Evidence Grade.Verified "
                    "(SumWhere col (Value.rat (mkRat s 1)))) (PProd (Evidence Grade.Verified "
                    "(CountWhere col (Value.rat (mkRat c 1)))) (PLift (s * 10 / c = 106)))\n"
                    "    Integer division handles rounding to match the gold answer's decimal places.\n"
                    "    Cross-multiply: avg=10.6 → sum*10 = count*106. Sigma binds Int witnesses.\n\n"
                    "  ARGMAX: 'Which city has the largest population?' →\n"
                    "    lean_goal_type: Σ (e : String), Σ (v : Value), Evidence Grade.Verified "
                    "(ArgmaxWhere \"population\" Cond.all e v)\n\n"
                    "  BINARY CLAIM: 'brazil scored more points' → no Σ, just PProd + PLift\n"
                    "    lean_goal_type: PProd (Evidence ...) (PLift (ValueGt ...))\n",
        formalization="-- convention, no Lean type",
    ))

    d.add_entry(DictEntry(
        name="numeric_types",
        type="Convention",
        description="ALL numeric values use Value.rat (mkRat n d). No Value.int.\n"
                    "  Integer N → Value.rat (mkRat N 1).  Decimal a/b → Value.rat (mkRat a b).\n"
                    "  Tool results: always Value.rat regardless of whether the column is integer or decimal.\n"
                    "  Cond thresholds: Value.rat (mkRat 2 1) or Value.rat (mkRat 2 1) for comparisons.\n"
                    "  DECIMAL COMPUTATIONS: use exact Rat arithmetic.\n"
                    "    Average: mkRat sum count = mkRat expected_num expected_den\n"
                    "    Example: avg = 463/9 → mkRat 463 9 (exact, auto-normalized)\n"
                    "    Multiplication: mkRat a b * mkRat c d = mkRat (a*c) (b*d)\n"
                    "    All exact, no truncation, no scaling. Proofs use native_decide.\n"
                    "  Integer arithmetic: +, -, *, / work via decide.\n"
                    "  DECOMPOSITION: total - part = remainder (by decide)",
        formalization="-- convention, no Lean type",
    ))

    return d


def init_table_dictionary(d: Dictionary, columns: list[str], col_types: dict[str, str],
                          caption: str = "", reading_guide: str = "") -> Dictionary:
    """Add table schema entries to an existing dictionary."""
    # Add session context
    if caption or reading_guide:
        d.add_entry(DictEntry(
            name="_session_context",
            type="Convention",
            description=f"Table caption: {caption}. {reading_guide}",
            formalization="-- context only, no Lean type",
        ))

    # Add one Property entry per column.
    # `name` is a String tag (the column header). The "Property"
    # type field is documentation/categorization for the formalizer,
    # not a Lean type.
    for col in columns:
        col_type = col_types.get(col, "str")
        d.add_entry(DictEntry(
            name=col,
            type="Property",
            description=f"Table column '{col}'. Inferred type: {col_type}.",
            formalization=f"{col} : String  -- column header (per R10 nominalist design)",
            aliases=[Alias(
                surface_form=col,
                context="column header",
                source="schema",
                grade_impact="no_cap",
            )],
        ))

    return d
