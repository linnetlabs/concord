"""Recall-complete sweep (find_all): recovers a scattered facet a top-k would miss."""
from __future__ import annotations

import numpy as np

from concordai.chunk import Passage
from concordai.find import find_all


class _Emb:
    def __init__(self, qv):
        self._qv = qv

    def embed(self, texts, kind="query"):
        return np.asarray([self._qv], dtype="float32")


class _Idx:
    def __init__(self, vecs, passages):
        self.matrix = np.asarray(vecs, dtype="float32")
        self.passages = passages
        self.meta = {}


def _norm(v):
    v = np.asarray(v, dtype="float32")
    return v / (np.linalg.norm(v) + 1e-9)


def test_find_all_recovers_a_scattered_facet():
    # facet A (pricing) -- on-topic with the query; facet B (security) -- off-topic.
    vecs = [_norm([1, 0, 0, 0]), _norm([0.98, 0.05, 0, 0]),   # A: high cosine to query
            _norm([0, 1, 0, 0]), _norm([0, 0.98, 0.05, 0])]   # B: ~orthogonal (low cosine)
    passages = [
        Passage("pricing.md", 1, 1, "Skyline costs $49 per seat per month", "public"),
        Passage("faq.md", 1, 1, "the price is $49 per seat billed monthly", "public"),
        Passage("security.md", 1, 1, "data is encrypted at rest with AES-256", "public"),
        Passage("trust.md", 1, 1, "we encrypt everything in transit and at rest", "public"),
    ]
    idx = _Idx(vecs, passages)
    hits = find_all("how much does it cost", index=idx, embedder=_Emb([1, 0, 0, 0]),
                    floor=0.55, patience=5)
    files = {h.file for h in hits}
    # both on-topic pricing passages kept...
    assert {"pricing.md", "faq.md"} <= files
    # ...AND at least one off-topic security passage recovered purely because it is a NEW facet
    assert files & {"security.md", "trust.md"}
    assert len({h.facet for h in hits if h.facet}) >= 2  # spanned >1 facet


def test_find_all_falls_back_to_exact_without_index(tmp_path):
    (tmp_path / "a.md").write_text("the secret codeword is bluebird here", encoding="utf-8")
    hits = find_all("bluebird", root=tmp_path)   # no index -> exact channel
    assert any(h.match_type == "exact" and "bluebird" in h.text for h in hits)
