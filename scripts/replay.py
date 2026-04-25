#!/usr/bin/env python3
"""Replay: typecheck each fixture in ``results/replay_fixtures``
against the Lean library (``src/lean/EGVar/``) and compare the outcome
to the expected verdict in ``manifest.json``.

Usage:
    python scripts/replay.py
    # exits 0 on success, 1 on any mismatch
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEAN_DIR = PROJECT_ROOT / "src" / "lean"
FIXTURE_DIR = PROJECT_ROOT / "results" / "replay_fixtures"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"

# Lean's error format used for regex-based detection. We accept the
# common prefix "error:" at line start or after a file:line:col prefix.
# Return code is authoritative; this regex is a conservative
# supplementary check.
_LEAN_ERROR_RE = re.compile(r"(?:^|\n)[^:\n]*error:", re.MULTILINE)


def typecheck(lean_file: Path, timeout: int = 120) -> tuple[bool, str]:
    """Run ``lake env lean`` on a file. Returns (accepted, combined_output).

    The Lean CLI's return code is the authoritative success signal.
    The ``error:`` regex is a supplementary check that catches the
    (rare) case of a zero return code with an error printed to stdout.
    """
    try:
        result = subprocess.run(
            ["lake", "env", "lean", str(lean_file)],
            cwd=LEAN_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            return False, output
        if _LEAN_ERROR_RE.search(output):
            return False, output
        return True, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except FileNotFoundError as e:
        return False, f"lake not on PATH: {e}"


def _validate_fixture_path(fixture_rel: object) -> Path:
    """Resolve a relative fixture path inside the project.

    Rejects: non-strings, absolute paths, paths with ``..``, paths that
    resolve outside ``PROJECT_ROOT``. Returns an absolute ``Path``
    anchored inside the project on success.
    """
    if not isinstance(fixture_rel, str):
        raise ValueError(f"fixture path must be a string, got {type(fixture_rel).__name__}")
    if not fixture_rel:
        raise ValueError("fixture path must be non-empty")
    p = Path(fixture_rel)
    if p.is_absolute():
        raise ValueError(f"fixture path must be relative, got {fixture_rel!r}")
    if ".." in p.parts:
        raise ValueError(f"fixture path must not contain '..', got {fixture_rel!r}")
    resolved = (PROJECT_ROOT / p).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as e:
        raise ValueError(
            f"fixture path resolves outside project root: {fixture_rel!r}"
        ) from e
    return resolved


def _load_manifest() -> list[dict]:
    """Load and validate ``manifest.json``. Raises ``ValueError`` on any
    schema violation.
    """
    if not MANIFEST_PATH.exists():
        raise ValueError(f"manifest not found at {MANIFEST_PATH}")
    try:
        parsed = json.loads(MANIFEST_PATH.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"manifest is not valid JSON: {e}") from e
    if not isinstance(parsed, dict) or "fixtures" not in parsed:
        raise ValueError("manifest must be an object with a 'fixtures' list")
    fixtures = parsed["fixtures"]
    if not isinstance(fixtures, list):
        raise ValueError("manifest['fixtures'] must be a list")
    for i, f in enumerate(fixtures):
        if not isinstance(f, dict):
            raise ValueError(f"fixture {i} must be an object")
        for required in ("label", "fixture", "expected"):
            if required not in f:
                raise ValueError(f"fixture {i} missing required field {required!r}")
        if f["expected"] not in ("accept", "reject"):
            raise ValueError(
                f"fixture {i} has invalid expected={f['expected']!r} "
                "(must be 'accept' or 'reject')"
            )
        if not isinstance(f["label"], str):
            raise ValueError(f"fixture {i} label must be a string")
        # Validate the fixture path eagerly so malformed manifests fail
        # fast (before running Lean on anything).
        _validate_fixture_path(f["fixture"])
    return fixtures


def main() -> int:
    try:
        fixtures = _load_manifest()
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            "Manifest must list each fixture and its expected verdict.",
            file=sys.stderr,
        )
        return 1

    passed = 0
    failed: list[tuple[str, str]] = []

    print(f"Regression replay: {len(fixtures)} fixtures", file=sys.stderr)
    print(f"  Lean library: {LEAN_DIR.relative_to(PROJECT_ROOT)}", file=sys.stderr)
    print("", file=sys.stderr)

    for f in fixtures:
        label = f["label"]
        fixture_path = _validate_fixture_path(f["fixture"])
        expected = f["expected"]

        if not fixture_path.exists():
            failed.append((label, f"missing: {fixture_path}"))
            print(f"  [{label}] MISSING {fixture_path}", file=sys.stderr)
            continue

        accepted, output = typecheck(fixture_path)
        got = "accept" if accepted else "reject"

        if got == expected:
            passed += 1
            print(f"  [{label}] {got} (as expected)", file=sys.stderr)
        else:
            failed.append((label, f"expected {expected}, got {got}"))
            print(
                f"  [{label}] MISMATCH expected={expected} got={got}",
                file=sys.stderr,
            )
            for line in output.splitlines()[:5]:
                print(f"       {line}", file=sys.stderr)

    print("", file=sys.stderr)
    print(f"Passed: {passed}/{len(fixtures)}", file=sys.stderr)
    if failed:
        print(f"Failed: {len(failed)}", file=sys.stderr)
        for label, reason in failed:
            print(f"  - {label}: {reason}", file=sys.stderr)
        return 1
    print("OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
