"""init makes the private ruleset + built index uncommittable."""
from __future__ import annotations

from concordai.bootstrap import ensure_gitignore, scaffold_rules


def test_scaffold_and_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("node_modules/\n", encoding="utf-8")

    rules_path, created = scaffold_rules(tmp_path)
    assert created and rules_path.exists()

    added = ensure_gitignore(tmp_path)
    assert "rules.yaml" in added and ".concord/" in added

    gi = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert "node_modules/" in gi          # preserved the user's lines
    for line in ("rules.yaml", "*.local.yaml", ".concord/"):
        assert line in gi


def test_init_is_idempotent(tmp_path):
    scaffold_rules(tmp_path)
    ensure_gitignore(tmp_path)
    assert ensure_gitignore(tmp_path) == []        # nothing added second time
    # rules.yaml not clobbered on a second scaffold
    _, created = scaffold_rules(tmp_path)
    assert created is False
