"""The shareable HTML report renders the lint + radar sections."""
from __future__ import annotations

from concordai import report
from concordai.lint import Finding


def test_report_renders_sections_and_branding():
    findings = [Finding("codename-x", "error", "internal codename", "web/a.html", 5, 1, "<h1>X</h1>", "exact")]
    conflicts = [{"clash": ["$49", "$39"], "subject": ["cloud"],
                  "a": {"file": "a.md", "line": 1, "text": "Cloud is $49", "values": ["$49"]},
                  "b": {"file": "b.md", "line": 1, "text": "Cloud is $39", "values": ["$39"]}}]
    html = report.build("myrepo", "2026-06-09", findings, conflicts, None)
    assert "Consistency report" in html and "myrepo" in html
    assert "web/a.html:5" in html                    # the leak
    assert "contradiction" in html.lower()           # the radar
    assert "a.md:1" in html and "$49" in html
    assert "by Linnet Labs" in html


def test_report_uses_verdicts_when_present():
    conflicts = [{"clash": ["n>=4", "n>=8"], "subject": ["respondents"],
                  "a": {"file": "a.md", "line": 1, "text": "n>=4", "values": ["n>=4"]},
                  "b": {"file": "b.md", "line": 1, "text": "n>=8", "values": ["n>=8"]}}]
    verdicts = [{"real": True, "canonical": "n >= 8", "change": "a", "why": "stricter"}]
    html = report.build("r", "2026", [], conflicts, verdicts)
    assert "confirmed" in html.lower() and "canonical" in html.lower()
    # a verdict marked not-real drops it -> the empty-state message shows
    dropped = report.build("r", "2026", [], conflicts, [{"real": False}])
    assert "No contradictions found" in dropped
