"""Adaptive read-depth: elbow cutoff, patience-streak stopping, MMR diversity."""
from __future__ import annotations

import numpy as np

from concordai.retrieve import adaptive_take, elbow_cutoff, mmr


def test_elbow_cuts_at_largest_gap():
    assert elbow_cutoff([0.9, 0.88, 0.87, 0.50, 0.48, 0.47]) == 3  # gap between rank 3 and 4


def test_elbow_edge_cases():
    assert elbow_cutoff([0.9], min_keep=1) == 1
    assert elbow_cutoff([], min_keep=1) == 0


def test_patience_stops_after_consecutive_irrelevant():
    items = ["a", "b", "c", "d", "e"]
    judge = lambda x: x in {"a", "b"}  # noqa: E731 -- c,d,e irrelevant
    assert adaptive_take(items, [0] * 5, judge=judge, patience=2) == ["a", "b"]


def test_single_miss_does_not_stop():
    items = ["a", "b", "c", "d"]
    judge = lambda x: x in {"a", "c", "d"}  # noqa: E731 -- only b misses (1 < patience)
    assert adaptive_take(items, [0] * 4, judge=judge, patience=2) == ["a", "c", "d"]


def test_no_judge_falls_back_to_elbow():
    assert adaptive_take(["a", "b", "c", "d"], [0.9, 0.89, 0.4, 0.3]) == ["a", "b"]


def test_mmr_picks_nearest_then_diverse():
    q = np.array([1, 0], dtype="float32")
    vecs = np.array([[1, 0], [0.99, 0.01], [0, 1]], dtype="float32")  # x1, near-dup x2, orthogonal y
    out = mmr(q, vecs, ["x1", "x2", "y"], lambda_=0.3, k=2)
    assert out[0] == "x1"   # nearest to query
    assert out[1] == "y"    # diversity beats the near-duplicate
