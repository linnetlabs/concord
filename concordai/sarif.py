"""SARIF 2.1.0 export -- concord findings in the format GitHub code-scanning ingests.

`concord lint --sarif` and `concord radar --sarif` emit a SARIF log; upload it with
github/codeql-action/upload-sarif and leaks / value contradictions show up inline in the
PR "Code scanning" tab, annotated on the offending lines. Pure and deterministic -- the
SARIF is just a dict serialised to JSON, no dependency.
"""
from __future__ import annotations

import json

from . import __version__

_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
_INFO_URI = "https://github.com/linnetlabs/concord"

_RULES = {
    "concord/leak": {
        "id": "concord/leak", "name": "CodenameLeak",
        "shortDescription": {"text": "A banned or internal term reached a public file."},
        "helpUri": _INFO_URI, "defaultConfiguration": {"level": "error"},
    },
    "concord/contradiction": {
        "id": "concord/contradiction", "name": "ValueContradiction",
        "shortDescription": {"text": "The same fact is stated with conflicting values."},
        "helpUri": _INFO_URI, "defaultConfiguration": {"level": "warning"},
    },
}


def _loc(uri: str, line) -> dict:
    return {"physicalLocation": {"artifactLocation": {"uri": uri},
                                 "region": {"startLine": max(1, int(line or 1))}}}


def _result(rule_id: str, level: str, text: str, locations: list, fingerprints=None) -> dict:
    r = {"ruleId": rule_id, "level": level, "message": {"text": text}, "locations": locations}
    if fingerprints:
        r["partialFingerprints"] = fingerprints
    return r


def lint_results(findings) -> list:
    """concordai.lint.Finding[] -> SARIF results (one per leak, on its file:line)."""
    out = []
    for f in findings:
        level = "error" if f.severity == "error" else "warning"
        out.append(_result(
            "concord/leak", level,
            f"{f.term_id}: {f.reason} ('{(f.snippet or '').strip()[:80]}')",
            [_loc(f.file, f.line)],
            {"concord/loc": f"{f.file}:{f.line}:{f.col}", "concord/term": str(f.term_id)}))
    return out


def radar_results(conflicts) -> list:
    """radar conflict dicts -> SARIF results (the contradiction sits on BOTH passages)."""
    out = []
    for c in conflicts:
        vs = " vs ".join(c.get("clash") or []) or "conflicting values"
        canon = ""
        if c.get("canonical"):
            canon = f"; likely canonical: {c['canonical_file']} ({c.get('canonical_reason', '')})"
        locs = [_loc(c["a"]["file"], c["a"]["line"]), _loc(c["b"]["file"], c["b"]["line"])]
        out.append(_result(
            "concord/contradiction", "warning",
            f"Same topic, conflicting values: {vs}{canon}", locs,
            {"concord/pair": f"{c['a']['file']}:{c['a']['line']}|{c['b']['file']}:{c['b']['line']}"}))
    return out


def sarif_log(results: list, rule_ids) -> dict:
    rules = [_RULES[r] for r in rule_ids if r in _RULES]
    return {
        "version": "2.1.0",
        "$schema": _SCHEMA,
        "runs": [{
            "tool": {"driver": {"name": "Concord", "version": __version__,
                                "informationUri": _INFO_URI, "rules": rules}},
            "results": results,
        }],
    }


def dumps(results: list, rule_ids) -> str:
    return json.dumps(sarif_log(results, rule_ids), indent=2)
