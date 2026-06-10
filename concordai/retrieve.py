"""Adaptive read-depth — decide how far down a similarity ranking to read.

The token-efficiency core. Fixed top-k either over-reads easy queries or
under-reads broad ones; these functions right-size the context per query.

Three judges, in cost order:
  - elbow      : pure geometry, zero model calls (default)
  - patience   : read until N consecutive items are judged irrelevant
  - (callable) : inject an LLM- or human-relevance judge for the marginal band

`mmr` re-ranks to avoid reading near-duplicate restatements of the same point.
All functions operate on plain similarity scores / vectors — no I/O.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

try:
    import numpy as np
except ImportError:  # numpy is a [semantic] extra; degrade gracefully
    np = None  # type: ignore


def elbow_cutoff(scores: Sequence[float], min_keep: int = 1) -> int:
    """Index *after* the largest drop in a descending score list (the elbow).

    Returns how many items to keep. Reads adapt to the query: a sharp drop after
    2 items keeps 2; a gentle decline keeps more.
    """
    s = list(scores)
    if len(s) <= min_keep:
        return len(s)
    gaps = [(s[i] - s[i + 1], i + 1) for i in range(len(s) - 1)]
    _, cut = max(gaps, key=lambda g: g[0])
    return max(cut, min_keep)


def adaptive_take(
    ranked: List,
    scores: Sequence[float],
    judge: Optional[Callable[[object], bool]] = None,
    patience: int = 2,
    max_k: Optional[int] = None,
) -> List:
    """Walk a ranked list, stopping when relevance dies.

    Without a `judge`, the geometric elbow decides (no model calls). With one,
    read until `patience` consecutive items are judged irrelevant — never stop on
    a single off-topic neighbour, which is brittle to false neighbours.
    """
    if max_k is not None:
        ranked = ranked[:max_k]
        scores = list(scores)[:max_k]
    if judge is None:
        return ranked[: elbow_cutoff(scores)]

    kept, misses = [], 0
    for item in ranked:
        if judge(item):
            kept.append(item)
            misses = 0
        else:
            misses += 1
            if misses >= patience:
                break
    return kept


def mmr(
    query_vec,
    cand_vecs,
    candidates: List,
    lambda_: float = 0.7,
    k: Optional[int] = None,
) -> List:
    """Maximal marginal relevance: balance similarity-to-query against novelty.

    Reads more *new* information per token by suppressing near-duplicate passages.
    """
    if np is None:
        raise RuntimeError("mmr requires the embeddings extra: pip install \"concord-ai[embeddings]\"")
    cand_vecs = np.asarray(cand_vecs, dtype="float32")
    q = np.asarray(query_vec, dtype="float32")
    sim_q = cand_vecs @ q
    k = k or len(candidates)
    selected: List[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: sim_q[i])
        else:
            sel = cand_vecs[selected]
            def score(i):
                novelty = float((cand_vecs[i] @ sel.T).max())
                return lambda_ * sim_q[i] - (1 - lambda_) * novelty
            best = max(remaining, key=score)
        selected.append(best)
        remaining.remove(best)
    return [candidates[i] for i in selected]
