"""Passage clustering — an annotated topic map over the index.

Flat top-k retrieval under-retrieves on "find ALL X" queries: it stops at a cliff of
near-duplicates and misses the scattered rest. A topic map fixes that two ways:
  - coverage: route a broad query to its cluster and return the whole neighbourhood
  - scoping:  run contradiction checks within one topic, O(cluster^2) not O(corpus^2)

Two levels — MiniBatchKMeans leaf clusters, then agglomerative super-clusters over the
leaf centroids — give an annotated hierarchy. Labels are tf-idf top terms (deterministic,
no model). This operates at the passage level (verbatim prose), not the entity level.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Clustering:
    leaf_of: "np.ndarray"          # (N,) leaf id per passage
    leaf_labels: List[str]         # per leaf
    leaf_centroids: "np.ndarray"   # (K, D)
    super_of_leaf: "np.ndarray"    # (K,) super id per leaf
    super_labels: List[str]        # per super
    sizes: List[int]               # passages per leaf

    @property
    def k(self) -> int:
        return len(self.leaf_labels)


def _labels_for(texts, assignments, n_clusters, top=4) -> List[str]:
    """Rough deterministic cluster labels = distinctive tf-idf terms (a keyword bag).

    Honest about what it is: a vague keyword summary, fine for orientation. Clean,
    human-readable topic NAMES come from `concord topics --samples` + a driver naming
    each cluster — deterministic heading-based labelling was tried and mislabelled
    themes, so it's not shipped.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(stop_words="english", max_features=4000, token_pattern=r"[A-Za-z][A-Za-z-]{2,}")
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return [f"cluster {i}" for i in range(n_clusters)]
    terms = np.array(vec.get_feature_names_out())
    out: List[str] = []
    for c in range(n_clusters):
        rows = np.where(assignments == c)[0]
        if len(rows) == 0:
            out.append(f"cluster {c}")
            continue
        mean = np.asarray(X[rows].mean(axis=0)).ravel()
        out.append(" · ".join(terms[mean.argsort()[::-1][:top]]))
    return out


def cluster(matrix, texts, k_leaves: int = 40, n_super: int = 8) -> Clustering:
    from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

    matrix = np.asarray(matrix, dtype="float32")
    n = matrix.shape[0]
    k_leaves = max(2, min(k_leaves, n))
    km = MiniBatchKMeans(n_clusters=k_leaves, random_state=0, n_init=3)
    leaf_of = km.fit_predict(matrix)
    leaf_labels = _labels_for(texts, leaf_of, k_leaves)
    sizes = [int((leaf_of == c).sum()) for c in range(k_leaves)]

    n_super = max(1, min(n_super, k_leaves))
    if n_super < k_leaves:
        super_of_leaf = AgglomerativeClustering(n_clusters=n_super).fit_predict(km.cluster_centers_)
    else:
        super_of_leaf = np.arange(k_leaves)
    super_labels = _labels_for(texts, super_of_leaf[leaf_of], n_super)

    return Clustering(leaf_of, leaf_labels, km.cluster_centers_, super_of_leaf, super_labels, sizes)


# Cosine-distance below which two passages are the "same facet". Tuned empirically
# for e5 geometry (it packs related prose tightly): a sweep on real windows put the
# clean knee at 0.15 — a near-duplicate query collapses to 1 facet, a topically
# spread one fans out (~3 for a broad GDPR query). This models facet COUNT = topical
# breadth, not geometric separability (silhouette/elbow over-split near-dup windows;
# they failed the single-blob ground-truth test). Lower it to split finer.
FACET_DISTANCE = 0.15


def _agglo_labels(vecs, distance_threshold: float = FACET_DISTANCE):
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    V = np.asarray(vecs, dtype="float32")
    n = len(V)
    if n < 3:
        return np.zeros(n, dtype=int)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold,
        metric="cosine", linkage="average",
    ).fit_predict(V)


def auto_k(vecs, distance_threshold: float = FACET_DISTANCE) -> int:
    """Number of facets in a window = clusters separated by more than the cosine
    threshold. A near-duplicate window -> 1; a topically spread one -> several."""
    return len(set(_agglo_labels(vecs, distance_threshold)))


def facet_labels(texts, vecs, distance_threshold: float = FACET_DISTANCE) -> List[str]:
    """Per-item facet label by clustering a small RESULT window (not the corpus).

    Organises what flat retrieval already found, so a driver can read best-per-facet
    and notice when a multi-facet query's later facets are being crowded out of the
    top by near-duplicates of the first. Facet count is auto-selected by topical
    spread (cosine-distance threshold), not a fixed or separability-tuned k.
    """
    import numpy as np

    n = len(texts)
    labels = _agglo_labels(vecs, distance_threshold)
    if len(set(labels)) <= 1:
        return [_labels_for(texts, np.zeros(n, dtype=int), 1, top=3)[0]] * n
    labs = _labels_for(texts, labels, len(set(labels)), top=3)  # agglo ids are 0..k-1
    return [labs[int(labels[i])] for i in range(n)]


def route(query_vec, clustering: Clustering, top: int = 1) -> List[int]:
    """EXPERIMENTAL — nearest leaf cluster(s) to a query.

    Measured to UNDERPERFORM flat retrieval (a centroid is a blurry average; specific
    queries match specific passages better). Kept for exploration only — not a
    retrieval path. Use flat find()/read for actually locating content.
    """
    q = np.asarray(query_vec, dtype="float32")
    q = q / (np.linalg.norm(q) or 1.0)
    cent = clustering.leaf_centroids
    cent = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9)
    return list(np.argsort(-(cent @ q))[:top])
