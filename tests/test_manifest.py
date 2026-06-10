"""Content-hash manifest detects changes without git (the non-git fallback)."""
from __future__ import annotations

import numpy as np

from concordai import manifest
from concordai.index import Index


class FakeEmb:
    def embed(self, texts, kind="passage"):
        return np.array([[float(len(t)), 1.0] for t in texts], dtype="float32")


def test_scan_and_diff(tmp_path):
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    m0 = manifest.scan(tmp_path)
    assert set(m0) == {"a.md", "b.md"}

    (tmp_path / "a.md").write_text("alpha edited", encoding="utf-8")  # change
    (tmp_path / "b.md").unlink()                                       # delete
    (tmp_path / "c.md").write_text("gamma", encoding="utf-8")          # add
    changed, deleted = manifest.diff(m0, manifest.scan(tmp_path))
    assert set(changed) == {"a.md", "c.md"}
    assert deleted == ["b.md"]


def test_touch_without_edit_is_not_a_change(tmp_path):
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    m0 = manifest.scan(tmp_path)
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")  # rewrite identical content
    changed, deleted = manifest.diff(m0, manifest.scan(tmp_path))
    assert changed == [] and deleted == []  # content hash unchanged


def test_index_save_writes_manifest_and_update_uses_it(tmp_path):
    (tmp_path / "a.md").write_text("alpha apple", encoding="utf-8")
    idx = Index.build(tmp_path, ruleset=None, embedder=FakeEmb())
    idx.save(tmp_path)
    assert (tmp_path / ".concord" / "manifest.json").exists()

    reloaded = Index.load(tmp_path)
    assert "a.md" in reloaded.manifest

    (tmp_path / "a.md").write_text("alpha changed", encoding="utf-8")
    changed, deleted = manifest.diff(reloaded.manifest, manifest.scan(tmp_path))
    assert changed == ["a.md"]
