"""M0 lint behaviour: banned terms in public files are caught; internal/data are exempt."""
from __future__ import annotations

import pathlib

import concordai
from concordai.lint import lint_repo
from concordai.rules import load_ruleset
from concordai.visibility import classify

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
EXAMPLE_RULES = pathlib.Path(concordai.__file__).parent / "rules.example.yaml"


def _findings():
    rules = load_ruleset(EXAMPLE_RULES)
    return lint_repo(FIXTURES, rules, scope=("public",))


def test_public_leaks_are_caught():
    ids = {f.term_id for f in _findings()}
    assert {"codename-bluebird", "retired-name-falcon", "retired-price"} <= ids


def test_only_public_files_flagged():
    files = {f.file for f in _findings()}
    assert files == {"web/landing.html"}
    assert not any(f.startswith("strategy/") or f.startswith("data/") for f in files)


def test_clean_lines_not_flagged():
    for f in _findings():
        assert "totally clean" not in f.snippet


def test_visibility_classification():
    vis = load_ruleset(EXAMPLE_RULES).visibility
    assert classify("web/landing.html", vis) == "public"
    assert classify("strategy/plan.md", vis) == "internal"
    assert classify("data/payload.json", vis) == "data"
    assert classify("random/thing.md", vis) == "unknown"


def test_line_and_column_are_reported():
    f = next(f for f in _findings() if f.term_id == "codename-bluebird")
    assert f.line == 4 and f.col > 0
