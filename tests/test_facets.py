"""auto_k picks the facet count from structure: distinct blobs -> that many; one blob -> 1."""
from __future__ import annotations

import numpy as np

from concordai.cluster import auto_k, facet_labels


def test_auto_k_finds_three_blobs():
    rng = np.random.RandomState(0)
    blobs = [rng.normal(c, 0.1, (15, 3)) for c in ([5, 0, 0], [0, 5, 0], [0, 0, 5])]
    V = np.vstack(blobs).astype("float32")
    assert auto_k(V) == 3


def test_auto_k_single_blob_is_one_facet():
    rng = np.random.RandomState(1)
    V = rng.normal([3, 3, 3], 0.3, (30, 3)).astype("float32")  # one direction, no real split
    assert auto_k(V) == 1


def test_facet_labels_count_matches_structure():
    rng = np.random.RandomState(2)
    V = np.vstack([rng.normal([5, 0], 0.1, (10, 2)), rng.normal([0, 5], 0.1, (10, 2))]).astype("float32")
    texts = ["alpha apple orchard"] * 10 + ["beta banana grove"] * 10
    labs = facet_labels(texts, V)  # k auto
    assert len(labs) == 20 and len(set(labs)) == 2
