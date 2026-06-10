"""Auto-resolve mechanics: value matching (whitespace/operator/boundary) + file edit."""
from __future__ import annotations

import re

from concordai.verify import _flex, apply_fix


def test_flex_is_whitespace_and_operator_tolerant():
    assert re.search(_flex("n>=4"), "require n >= 4 here", re.I)
    assert re.search(_flex("n>=8"), "the floor is n ≥ 8.", re.I)
    assert re.search(_flex("$49"), "it costs $49 today")


def test_flex_respects_number_boundary():
    assert not re.search(_flex("$49"), "it costs $490 today")  # no partial match


def test_apply_fix_replaces_on_the_change_side(tmp_path):
    (tmp_path / "p.md").write_text("Aggregate results require n >= 4 here.\n", encoding="utf-8")
    conflict = {"clash": ["n>=4", "n>=8"],
                "a": {"file": "p.md", "line": 1, "values": ["n>=4"]},
                "b": {"file": "o.md", "line": 1, "values": ["n>=8"]}}
    res = apply_fix(str(tmp_path), "a", "n >= 8", conflict, dry_run=True)
    assert res and res[2].endswith("n >= 4 here.") and res[3].endswith("n >= 8 here.")
    assert "n >= 4" in (tmp_path / "p.md").read_text()  # dry run didn't write

    apply_fix(str(tmp_path), "a", "n >= 8", conflict, dry_run=False)
    assert "n >= 8" in (tmp_path / "p.md").read_text() and "n >= 4" not in (tmp_path / "p.md").read_text()
