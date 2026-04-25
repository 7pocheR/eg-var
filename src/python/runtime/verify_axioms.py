"""
Axiom verifier: whitelist-compliance check for generated .lean files.

Enforces the operational replay gate of the kernel-trust theorem:
every axiom the proof depends on must belong to one of these buckets.

  (1) FIXED_PRELUDE_AXIOMS - declared in EGVar/Basic.lean,
      EGVar/TableL1.lean, EGVar/WorldOntology.lean, etc. These are
      the EG-VAR calculus itself (mkVerified, downcast, WProp,
      Evidence, Attested_T, ClaimTag_T, ToolId, ...).

  (2) STANDARD_LEAN_AXIOMS - Lean 4 core axioms pulled in by any
      tactic-heavy proof (Classical.choice, Quot.sound, propext).

  (3) RUNTIME_LEAF_SCHEMAS - per-observation axioms emitted by the
      runtime during tool execution. Names match the regex
      patterns enumerated below. These are the only axioms that
      vary per-claim.

  (4) PER_TABLE_LIFT_PREFIX - per-source lift axioms from the
      audited fixture Lambda(s). Names match lift_<hex>_<op>.

A proof is whitelist-compliant iff every axiom in its dependency
closure falls into one of these buckets AND the required
kernel-trust axioms (mkVerified, claimOfTag) appear somewhere. Any
sorry* variant is flagged. Unknown axioms cause a FAIL with the
offending axiom listed.
"""
from __future__ import annotations
import subprocess
import re
from pathlib import Path

LEAN_DIR = Path(__file__).resolve().parent.parent.parent.parent / "src" / "lean"

# (1) Fixed EG-VAR prelude — declared in src/lean/EGVar/*.lean. Must be
#     reflected in the trust artifact: changes here require review.
FIXED_PRELUDE_AXIOMS = {
    # Basic.lean — calculus core
    "WProp", "WAnd", "WOr", "WNot", "Evidence", "Grade", "downcast",
    "ToolId", "Query_T", "Payload_T", "Attested_T",
    "WorldRec", "interp_T", "ClaimOf", "ClaimTag_T", "claimWProp",
    "claimOfTag", "mkVerified", "CSQuery", "CSAttestation", "mkPlausible",
    "Timestamp", "At",
    # TableL1.lean — L1 predicate shells (WProp constructors)
    "RowHandle", "CellAt", "Cond",
    "SumCells", "CountCells", "CountDistinctCells", "MeanCells",
    "MaxCellValue", "MinCellValue", "MaxCell", "MinCell",
    "TopKSumCells", "TopKMeanCells", "TopKSumByCells", "TopKMeanByCells",
    "BottomKSumCells", "BottomKMeanCells",
    # WorldOntology.lean / Rules.lean / Ontology.lean — L2 predicates
    "HasProperty", "Value",
    "SumWhere", "CountWhere", "CountDistinctWhere", "MeanWhere",
    "MaxWhere", "MinWhere", "ArgmaxWhere", "ArgminWhere",
    "TopKSumWhere", "TopKMeanWhere",
    "TopKSumByWhere", "TopKMeanByWhere",
    "BottomKSumWhere", "BottomKMeanWhere",
    "Entity",
}

# (2) Standard Lean 4 core axioms pulled in by any non-trivial tactic proof.
STANDARD_LEAN_AXIOMS = {"Classical.choice", "Quot.sound", "propext"}

# (3) Runtime-emitted observation-leaf schemas (regex-based). Every
#     obs<i>_* axiom appearing in a generated .lean must match one
#     of these patterns. The regex anchors _N_ where N is digits.
RUNTIME_LEAF_PATTERNS = [
    re.compile(r"^obs\d+_q$"),          # Query_T axiom
    re.compile(r"^obs\d+_p$"),          # Payload_T axiom
    re.compile(r"^obs\d+_att$"),        # Attested_T axiom
    re.compile(r"^obs\d+_tag$"),        # ClaimTag_T axiom
    re.compile(r"^obs\d+_shape$"),      # claimWProp equality axiom
    re.compile(r"^obs\d+_l1$"),         # direct L1 Evidence Verified axiom
    re.compile(r"^obs\d+_row$"),        # RowHandle axiom
    re.compile(r"^obs\d+_cell$"),       # CellAt Evidence axiom
    re.compile(r"^obs\d+_id$"),         # identity-cell Evidence axiom
    re.compile(r"^obs\d+_max$"),        # MaxCell/MinCell evidence axiom
    re.compile(r"^obs\d+_min$"),
    # tool ID axioms emitted once per module
    re.compile(r"^tableTool$"),
    re.compile(r"^statsTool$"),
    # content-stream axioms (for CSAttestation paths)
    re.compile(r"^cs\d+_q$"),
    re.compile(r"^cs\d+_att$"),
]

# (4) Per-source lift axioms. Names emitted as lift_<source_hash>_<suffix>
#     from tier1_table_formalizations.json.
PER_TABLE_LIFT_PATTERN = re.compile(r"^lift_[0-9a-f]+_.+$")

# Minimum axioms that must appear for a proof to be structurally using
# the EG-VAR evidence discipline. Not all proofs invoke mkVerified
# (lift-only paths can route through direct Evidence axioms plus lifts);
# however, any trust-artifact chain MUST include either mkVerified OR a
# runtime-emitted Evidence axiom alongside a per-table lift. We check this
# at the bucket level rather than requiring a single named axiom.

FORBIDDEN_AXIOMS = {"sorry", "sorryAx", "Lean.Axiom.sorry"}


def classify_axiom(name: str) -> str:
    """Return the whitelist bucket for an axiom name, or 'UNKNOWN'."""
    if name in FORBIDDEN_AXIOMS:
        return "FORBIDDEN"
    if name in FIXED_PRELUDE_AXIOMS:
        return "prelude"
    if name in STANDARD_LEAN_AXIOMS:
        return "lean_std"
    for pat in RUNTIME_LEAF_PATTERNS:
        if pat.match(name):
            return "runtime_leaf"
    if PER_TABLE_LIFT_PATTERN.match(name):
        return "per_table_lift"
    return "UNKNOWN"


def check_axioms(lean_file: Path, def_name: str = "claim_proof") -> dict:
    """Run #print axioms on a definition and verify whitelist compliance.

    A proof passes iff every axiom in its dependency closure is in a
    whitelist bucket, no FORBIDDEN axiom is present, and the closure
    contains at least one trust-artifact axiom (either mkVerified via
    the attestation path OR at least one runtime_leaf plus one
    per_table_lift).
    """
    # Append #print axioms to a temp copy if not already present
    content = lean_file.read_text()
    if f"#print axioms {def_name}" not in content:
        content += f"\n#print axioms {def_name}\n"
        temp = lean_file.with_suffix(".check.lean")
        temp.write_text(content)
    else:
        temp = lean_file

    result = subprocess.run(
        ["lake", "env", "lean", str(temp)],
        cwd=LEAN_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )

    output = result.stdout + result.stderr

    # Parse axiom list from `'claim_proof' depends on axioms: [a1, a2, ...]`
    # which may span multiple lines.
    axioms: set[str] = set()
    in_list = False
    for line in output.splitlines():
        if "depends on axioms" in line:
            in_list = True
            m = re.search(r"\[(.+)\]", line)
            if m:
                for a in m.group(1).split(","):
                    axioms.add(a.strip().rstrip(","))
                in_list = False
        elif in_list:
            line = line.strip().rstrip(",").strip()
            if line == "]" or line == "":
                in_list = False
            elif line.startswith("["):
                axioms.add(line.lstrip("[").rstrip(",").strip())
            else:
                axioms.add(line.rstrip("]").strip())

    # Classify every axiom
    bucketed: dict[str, list[str]] = {
        "prelude": [], "lean_std": [], "runtime_leaf": [],
        "per_table_lift": [], "FORBIDDEN": [], "UNKNOWN": [],
    }
    for a in sorted(axioms):
        if a:
            bucketed[classify_axiom(a)].append(a)

    # Trust-artifact discipline: at least one runtime_leaf AND one
    # per_table_lift OR mkVerified (via the attestation path) must be
    # present. mkVerified alone is sufficient; lift-only paths need
    # both a runtime leaf and a per_table_lift.
    has_mk_verified = "mkVerified" in axioms
    has_runtime_leaf = len(bucketed["runtime_leaf"]) > 0
    has_per_table_lift = len(bucketed["per_table_lift"]) > 0
    trust_ok = has_mk_verified or (has_runtime_leaf and has_per_table_lift)

    passed = (
        not bucketed["FORBIDDEN"]
        and not bucketed["UNKNOWN"]
        and trust_ok
    )

    if temp != lean_file:
        temp.unlink()

    return {
        "def_name": def_name,
        "axioms": sorted(axioms),
        "axiom_count": len(axioms),
        "buckets": {k: v for k, v in bucketed.items() if v},
        "forbidden": bucketed["FORBIDDEN"],
        "unknown": bucketed["UNKNOWN"],
        "trust_ok": trust_ok,
        "has_mk_verified": has_mk_verified,
        "has_runtime_leaf": has_runtime_leaf,
        "has_per_table_lift": has_per_table_lift,
        "pass": passed,
    }


def main():
    import sys, json
    generated_dir = Path(__file__).resolve().parent.parent.parent.parent / "results" / "generated"

    files = sorted(generated_dir.glob("*.lean"))
    if not files:
        print("No generated files found.")
        return

    all_pass = True
    counts = {"pass": 0, "fail": 0}
    for f in files:
        def_name = "claim_proof"
        result = check_axioms(f, def_name)

        if result["pass"]:
            counts["pass"] += 1
            print(f"PASS  {f.name}  ({result['axiom_count']} axioms; "
                  f"mkVerified={result['has_mk_verified']}, "
                  f"lifts={len(result['buckets'].get('per_table_lift', []))}, "
                  f"leaves={len(result['buckets'].get('runtime_leaf', []))})")
        else:
            counts["fail"] += 1
            all_pass = False
            print(f"FAIL  {f.name}")
            if result["forbidden"]:
                print(f"      FORBIDDEN axioms: {result['forbidden']}")
            if result["unknown"]:
                print(f"      UNKNOWN axioms: {result['unknown']}")
            if not result["trust_ok"]:
                print(f"      TRUST GAP: "
                      f"mkVerified={result['has_mk_verified']}, "
                      f"runtime_leaf={result['has_runtime_leaf']}, "
                      f"per_table_lift={result['has_per_table_lift']}")

    print(f"\nSummary: {counts['pass']} pass / {counts['fail']} fail "
          f"/ {counts['pass']+counts['fail']} total")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
