"""The `find()` primitive -- exact and semantic channels, one ranked result.

Every mode in Concord is a view over `find`:
  - the lint        = find(scope="public", channels=("exact",)) per banned term
  - lookup / read   = find(<question>) then adaptive-read the ranked passages
  - consistency     = find(<a passage's text>) to surface near-duplicate wording

Exact hits score 1.0 and are tagged "exact"; semantic hits carry their cosine
and are tagged "semantic". Exact always works; semantic activates when an index
(and the [semantic] extra) is present.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .chunk import chunk_repo
from .index import Index
from .rules import Ruleset


@dataclass
class Hit:
    file: str
    line: int
    text: str
    score: float
    match_type: str  # "exact" | "semantic"
    visibility: str = "unknown"
    facet: Optional[str] = None  # set by find_all (which facet/cluster this hit covers)


def _exact_channel(query: str, root: Path, ruleset: Optional[Ruleset], scope) -> List[Hit]:
    rx = re.compile(re.escape(query), re.IGNORECASE)
    hits: List[Hit] = []
    for p in chunk_repo(root, ruleset):
        if scope and p.visibility not in scope:
            continue
        for offset, line in enumerate(p.text.splitlines()):
            if rx.search(line):
                hits.append(Hit(p.file, p.start_line + offset, line.strip()[:200], 1.0, "exact", p.visibility))
    return hits


def find(
    query: "str | Sequence[str]",
    root: "str | Path" = ".",
    ruleset: Optional[Ruleset] = None,
    channels: Sequence[str] = ("exact", "semantic"),
    scope: Optional[Sequence[str]] = None,
    top: int = 50,
    index: Optional[Index] = None,
    embedder=None,
) -> List[Hit]:
    """Return ranked hits, merging the requested channels.

    `query` may be a single string or several phrasings (multi-query). Semantic
    scores are MAX-merged across phrasings, so a passage matching ANY phrasing ranks
    high -- measured to recover recall a single phrasing misses. Pair multi-query with
    a patience-walk over the ranked list, not a fixed `top`, or the phrasings crowd
    each other out of a small window.

    `scope` optionally restricts to visibility categories (e.g. ("public",) for the
    leak-paraphrase check). Exact hits sort first (score 1.0), semantic by cosine.
    """
    root = Path(root)
    scope = set(scope) if scope else None
    qlist = [query] if isinstance(query, str) else list(query)
    hits: List[Hit] = []

    if "exact" in channels:
        seen = set()
        for q in qlist:
            for h in _exact_channel(q, root, ruleset, scope):
                if (h.file, h.line) not in seen:
                    seen.add((h.file, h.line))
                    hits.append(h)

    if "semantic" in channels:
        try:
            import numpy as np  # only needed for the semantic channel
            idx = index or Index.load(root)
            from .embed import get_embedder
            emb = embedder or get_embedder(getattr(idx, "meta", {}).get("model"))
            if idx.matrix is not None:
                qvs = np.asarray(emb.embed(qlist, kind="query"), dtype="float32")
                qn = qvs / (np.linalg.norm(qvs, axis=1, keepdims=True) + 1e-9)
                sims = (idx.matrix @ qn.T).max(axis=1)  # best phrasing per passage
                for i in np.argsort(-sims)[:top]:
                    p = idx.passages[int(i)]
                    if scope and p.visibility not in scope:
                        continue
                    hits.append(Hit(p.file, p.start_line, p.text[:200], float(sims[int(i)]), "semantic", p.visibility))
        except (FileNotFoundError, RuntimeError):
            pass  # no index / no ML backend -- exact channel still returned

    hits.sort(key=lambda h: (-h.score, h.match_type != "exact", h.file, h.line))
    return hits[:top]


def find_all(
    query: "str | Sequence[str]",
    root: "str | Path" = ".",
    ruleset: Optional[Ruleset] = None,
    scope: Optional[Sequence[str]] = None,
    pool: int = 150,
    patience: int = 5,
    floor: float = 0.55,
    index: Optional[Index] = None,
    embedder=None,
) -> List[Hit]:
    """Recall-oriented retrieval for "find ALL X" sweeps.

    A fixed top-k with an aggressive cutoff under-retrieves a scattered answer: it
    fills with near-duplicate restatements of the first facet and misses the rest.
    Instead, this clusters a generous candidate `pool` into facets and walks the
    ranking, keeping a hit while it is on-topic (cosine >= `floor`) OR introduces a
    NEW facet, stopping only after `patience` consecutive hits add neither. The result
    is recall-complete at the cost of more tokens -- the trade the README's caveat names.
    Falls back to the exact channel when there is no semantic index.
    """
    root = Path(root)
    scope = set(scope) if scope else None
    try:
        import numpy as np
        idx = index or Index.load(root)
        from .embed import get_embedder
        emb = embedder or get_embedder(getattr(idx, "meta", {}).get("model"))
        if idx.matrix is None:
            raise FileNotFoundError
    except (FileNotFoundError, RuntimeError):
        return find(query, root, ruleset, channels=("exact",), scope=scope, top=pool)

    qlist = [query] if isinstance(query, str) else list(query)
    qvs = np.asarray(emb.embed(qlist, kind="query"), dtype="float32")
    qn = qvs / (np.linalg.norm(qvs, axis=1, keepdims=True) + 1e-9)
    sims = (idx.matrix @ qn.T).max(axis=1)
    order = [int(i) for i in np.argsort(-sims)[:pool]]
    cand = [i for i in order if not (scope and idx.passages[i].visibility not in scope)]
    if not cand:
        return []

    facets: List[Optional[str]]
    try:
        from . import cluster as _cluster
        facets = _cluster.facet_labels([idx.passages[i].text for i in cand], idx.matrix[cand])
    except Exception:  # noqa: BLE001 -- facets are the stopping signal; degrade to pure floor walk
        facets = [None] * len(cand)

    kept: List[Hit] = []
    seen: set = set()
    misses = 0
    for n, i in enumerate(cand):
        s = float(sims[i])
        fac = facets[n] if n < len(facets) else None
        new_facet = fac is not None and fac not in seen
        if s >= floor or new_facet:
            p = idx.passages[i]
            kept.append(Hit(p.file, p.start_line, p.text[:200], s, "semantic", p.visibility, fac))
            if fac is not None:
                seen.add(fac)
            misses = 0
        else:
            misses += 1
            if misses >= patience:
                break
    return kept
