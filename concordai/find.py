"""The `find()` primitive — exact and semantic channels, one ranked result.

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
    high — measured to recover recall a single phrasing misses. Pair multi-query with
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
            pass  # no index / no ML backend — exact channel still returned

    hits.sort(key=lambda h: (-h.score, h.match_type != "exact", h.file, h.line))
    return hits[:top]
