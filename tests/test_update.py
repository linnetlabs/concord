"""Incremental update: changed files re-embed, deleted files drop, matrix stays aligned."""
from __future__ import annotations

import numpy as np

from concordai.index import Index
from concordai.gitdiff import _parse_name_status


class FakeEmb:
    """Deterministic 4-d embedder so the splice logic is testable without a model."""

    def embed(self, texts, kind="passage"):
        return np.array(
            [[float(len(t)), float(t.count("a")), 1.0, 2.0] for t in texts], dtype="float32"
        )


def test_update_replaces_changed_and_drops_deleted(tmp_path):
    (tmp_path / "a.md").write_text("alpha apple\n\nsecond para a a a", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta banana", encoding="utf-8")

    idx = Index.build(tmp_path, ruleset=None, embedder=FakeEmb())
    assert {p.file for p in idx.passages} == {"a.md", "b.md"}
    assert idx.matrix.shape[0] == len(idx.passages)

    (tmp_path / "a.md").write_text("alpha changed", encoding="utf-8")
    (tmp_path / "b.md").unlink()
    idx.update(tmp_path, changed=["a.md"], deleted=["b.md"], embedder=FakeEmb())

    assert {p.file for p in idx.passages} == {"a.md"}        # deleted file gone
    assert len(idx.passages) == 1                            # a re-chunked to its new content
    assert idx.matrix.shape[0] == len(idx.passages)          # matrix still aligned


def test_parse_name_status_handles_renames_and_deletes():
    changed, deleted = [], []
    _parse_name_status("M\tdocs/x.md\nD\tdocs/y.md\nR100\told.md\tnew.md", changed, deleted)
    assert "docs/x.md" in changed and "new.md" in changed     # rename -> new path is the target
    assert "docs/y.md" in deleted


def test_save_load_roundtrips_meta(tmp_path):
    idx = Index.build(tmp_path / "empty" if False else tmp_path, ruleset=None, embedder=FakeEmb()) \
        if list(tmp_path.glob("*.md")) else Index([], None)
    idx.meta = {"model": "e5-small", "commit": "abc123"}
    idx.save(tmp_path)
    again = Index.load(tmp_path)
    assert again.meta["model"] == "e5-small" and again.meta["commit"] == "abc123"
