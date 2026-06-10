"""The semantic find() path: single-query ranking and multi-query max-merge."""
from __future__ import annotations

import numpy as np

from concordai.find import find
from concordai.index import Index


class FakeEmb:
    """Deterministic 3-d embedder keyed on two marker tokens (no real model)."""

    def embed(self, texts, kind="passage"):
        return np.array(
            [[t.count("alpha"), t.count("bravo"), 1.0] for t in texts], dtype="float32"
        )


def _index(tmp_path):
    (tmp_path / "a.md").write_text("alpha alpha alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("bravo bravo bravo", encoding="utf-8")
    return Index.build(tmp_path, ruleset=None, embedder=FakeEmb())


def test_semantic_find_ranks_nearest_first(tmp_path):
    idx = _index(tmp_path)
    hits = find("alpha", root=tmp_path, channels=("semantic",), index=idx, embedder=FakeEmb())
    assert hits and hits[0].match_type == "semantic"
    assert hits[0].file == "a.md"


def test_multi_query_max_merge_surfaces_both(tmp_path):
    idx = _index(tmp_path)
    hits = find(["alpha", "bravo"], root=tmp_path, channels=("semantic",),
                index=idx, embedder=FakeEmb(), top=10)
    assert {"a.md", "b.md"} <= {h.file for h in hits}  # each phrasing's best surfaces
