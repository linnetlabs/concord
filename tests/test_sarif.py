"""SARIF 2.1.0 export for lint findings and radar contradictions."""
from __future__ import annotations

import json

from concordai import sarif
from concordai.lint import Finding


def test_lint_results_map_to_leak_rule():
    findings = [Finding(term_id="PROJECT_X", severity="error", reason="internal codename",
                        file="docs/public.md", line=12, col=3, snippet="ship PROJECT_X soon")]
    res = sarif.lint_results(findings)
    assert len(res) == 1
    r = res[0]
    assert r["ruleId"] == "concord/leak"
    assert r["level"] == "error"
    loc = r["locations"][0]["physicalLocation"]
    assert loc["artifactLocation"]["uri"] == "docs/public.md"
    assert loc["region"]["startLine"] == 12


def test_radar_results_carry_both_sides_and_canonical():
    conflicts = [{
        "clash": ["$49", "$39"],
        "a": {"file": "pricing.md", "line": 4},
        "b": {"file": "faq.md", "line": 9},
        "canonical": "a", "canonical_file": "pricing.md", "canonical_reason": "faq.md is the stale side",
    }]
    res = sarif.radar_results(conflicts)
    assert len(res) == 1 and res[0]["ruleId"] == "concord/contradiction"
    assert len(res[0]["locations"]) == 2          # the conflict sits on both passages
    assert "pricing.md" in res[0]["message"]["text"]    # canonical surfaced
    assert "$49 vs $39" in res[0]["message"]["text"]


def test_sarif_log_shape_is_valid():
    log = sarif.sarif_log(sarif.radar_results([]), ["concord/contradiction"])
    assert log["version"] == "2.1.0"
    assert log["$schema"].endswith("sarif-2.1.0.json")
    driver = log["runs"][0]["tool"]["driver"]
    assert driver["name"] == "Concord"
    assert any(rule["id"] == "concord/contradiction" for rule in driver["rules"])


def test_dumps_is_json():
    s = sarif.dumps(sarif.lint_results([]), ["concord/leak"])
    parsed = json.loads(s)                          # round-trips
    assert parsed["runs"][0]["results"] == []


def test_region_startline_is_at_least_one():
    findings = [Finding(term_id="x", severity="warn", reason="r", file="f", line=0, col=0, snippet="")]
    res = sarif.lint_results(findings)
    assert res[0]["locations"][0]["physicalLocation"]["region"]["startLine"] == 1  # SARIF is 1-based
