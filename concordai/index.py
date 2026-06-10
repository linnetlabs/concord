"""Semantic index — embed passages once, search by cosine.

Small-corpus design: passages + a dense matrix persisted to `.concord/`. At repo
scale (a few thousand passages) a numpy matmul is instant, so there is no faiss /
external vector store to operate. Build once; every semantic query reuses it.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

from .chunk import Passage, chunk_file, chunk_repo, _TEXT_EXTS
from .embed import Embedder, get_embedder
from .rules import Ruleset
from .visibility import classify

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

_DIR = ".concord"


class Index:
    def __init__(self, passages: List[Passage], matrix=None, meta=None, manifest=None):
        self.passages = passages
        self.matrix = matrix  # (N, D) float32, rows L2-normalised
        self.meta = meta or {}  # {"model": ..., "commit": ...} for incremental update
        self.manifest = manifest or {}  # {rel_path: content_hash} for non-git change detection

    @classmethod
    def build(
        cls,
        root: "str | Path",
        ruleset: Optional[Ruleset] = None,
        embedder: Optional[Embedder] = None,
    ) -> "Index":
        if np is None:
            raise RuntimeError("index requires the embeddings extra: pip install \"concord-ai[embeddings]\"")
        passages = list(chunk_repo(root, ruleset, prose=True))
        embedder = embedder or get_embedder()
        if not passages:
            return cls(passages, None)  # nothing indexable — empty index, not a crash
        vecs = np.asarray(embedder.embed([p.text for p in passages], kind="passage"), dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return cls(passages, vecs / norms)

    def search(self, query_vec, top: int = 50) -> List[Tuple[Passage, float]]:
        if np is None or self.matrix is None:
            return []
        q = np.asarray(query_vec, dtype="float32")
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self.matrix @ q
        order = np.argsort(-sims)[:top]
        return [(self.passages[i], float(sims[i])) for i in order]

    # --- incremental update -------------------------------------------------
    def update(self, root, changed, deleted, embedder: Embedder) -> "Index":
        """Re-embed only `changed` files and drop `deleted` ones, in place.

        Passages from any changed/deleted file are removed (with their matrix rows),
        the changed files are re-chunked and re-embedded, and the results appended.
        This is what makes update-on-commit cheap: cost scales with the diff, not the
        corpus. The caller supplies the file lists (see gitdiff.changed_files)."""
        if np is None:
            raise RuntimeError("update requires the embeddings extra: pip install \"concord-ai[embeddings]\"")
        drop = set(changed) | set(deleted)
        keep = [i for i, p in enumerate(self.passages) if p.file not in drop]
        passages = [self.passages[i] for i in keep]
        matrix = self.matrix[keep] if self.matrix is not None else None

        fresh: List[Passage] = []
        for rel in changed:
            fp = Path(root) / rel
            if fp.suffix.lower() in _TEXT_EXTS and fp.exists():
                fresh.extend(chunk_file(fp, rel=rel, prose=True))
        if fresh:
            vecs = np.asarray(embedder.embed([p.text for p in fresh], kind="passage"), dtype="float32")
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
            matrix = vecs if (matrix is None or matrix.shape[0] == 0) else np.vstack([matrix, vecs])
            passages = passages + fresh

        self.passages, self.matrix = passages, matrix
        return self

    # --- persistence --------------------------------------------------------
    def save(self, root: "str | Path", meta=None) -> None:
        d = Path(root) / _DIR
        d.mkdir(exist_ok=True)
        # Self-ignore: the built index is a derived cache, never committed — and we
        # do this without touching the user's root .gitignore.
        (d / ".gitignore").write_text("*\n", encoding="utf-8")
        (d / "passages.json").write_text(
            json.dumps([asdict(p) for p in self.passages], ensure_ascii=False),
            encoding="utf-8",
        )
        if meta:
            self.meta = {**self.meta, **meta}
        (d / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False), encoding="utf-8")
        from . import manifest as _manifest
        self.manifest = _manifest.scan(root)
        (d / "manifest.json").write_text(json.dumps(self.manifest, ensure_ascii=False), encoding="utf-8")
        if np is not None and self.matrix is not None:
            np.save(d / "matrix.npy", self.matrix)

    @classmethod
    def load(cls, root: "str | Path") -> "Index":
        d = Path(root) / _DIR
        passages = [Passage(**p) for p in json.loads((d / "passages.json").read_text("utf-8"))]
        matrix = np.load(d / "matrix.npy") if (np is not None and (d / "matrix.npy").exists()) else None
        meta = json.loads((d / "meta.json").read_text("utf-8")) if (d / "meta.json").exists() else {}
        manifest = json.loads((d / "manifest.json").read_text("utf-8")) if (d / "manifest.json").exists() else {}
        return cls(passages, matrix, meta=meta, manifest=manifest)
