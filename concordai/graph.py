"""Graph + consistency facade -- the biome-facing API.

Concord already knows the corpus (passages, visibility), its cross-references (links.py),
its contradictions (radar.py), and its git history (activity.py). This module joins them
into one stable artifact other tools can consume without reaching into submodules:

    concord.graph(root)        -> {nodes, edges}  library graph (files + doc-links + freshness)
    concord.consistency(root)  -> {conflicts}     numeric radar; prose contradictions when verify=True

`graph()` writes `.concord/graph.json` so a UI (e.g. the cortex) can read it directly.
Deterministic except for the opt-in `consistency(verify=True)` LLM pass.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from . import activity as _activity
from . import links as _links
from .index import _DIR

_FRESH_DAYS, _AGING_DAYS = 120, 270


def _age_days(last: str, today) -> Optional[int]:
    if not last:
        return None
    try:
        import datetime as _dt
        return (today - _dt.date.fromisoformat(last)).days
    except Exception:
        return None


def _freshness(age) -> str:
    if age is None:
        return "unknown"
    return "fresh" if age <= _FRESH_DAYS else "aging" if age <= _AGING_DAYS else "stale"


def graph(root, write: bool = True) -> dict:
    """Build the library graph: file nodes (passages + git churn/freshness) and doc-link edges.

    Relative-staleness ('lagging') is the cheap version of "this doc fell behind what it connects
    to": a node lags when a graph neighbour was last edited >60 days more recently than it.
    """
    import datetime as _dt
    root = Path(root)

    # passages + visibility from a built index if present, else chunk on the fly
    try:
        from .index import Index
        passages = Index.load(root).passages
    except Exception:
        from .chunk import chunk_repo
        passages = list(chunk_repo(root, None, prose=True))
    by_file = Counter(p.file for p in passages)
    vis = {}
    for p in passages:
        vis.setdefault(p.file, getattr(p, "visibility", "unknown"))

    act = {r["file"]: r for r in _activity.file_activity(root)}
    linkmap = _links.scan(root)
    files = set(by_file) | set(linkmap) | {e["target"] for v in linkmap.values() for e in v}

    today = _dt.date.today()
    last_of = {f: act.get(f, {}).get("last", "") for f in files}

    # neighbour latest-commit, for the relative-staleness signal (undirected over the link graph)
    nbr_latest = {f: "" for f in files}
    for src, edges in linkmap.items():
        for e in edges:
            tgt = e["target"]
            nbr_latest[src] = max(nbr_latest.get(src, ""), last_of.get(tgt, ""))
            nbr_latest[tgt] = max(nbr_latest.get(tgt, ""), last_of.get(src, ""))

    nodes = []
    for f in sorted(files):
        a = act.get(f, {})
        last = last_of.get(f, "")
        age = _age_days(last, today)
        nb = nbr_latest.get(f, "")
        lag = bool(last and nb and (_age_days(last, today) or 0) - (_age_days(nb, today) or 0) > 60)
        nodes.append({
            "file": f,
            "passages": by_file.get(f, 0),
            "visibility": vis.get(f, "unknown"),
            "churn": a.get("churn", 0),
            "authors": a.get("authors", 0),
            "last": last,
            "freshness": _freshness(age),
            "lagging": lag,
        })

    edges = [{"source": src, "target": e["target"], "text": e.get("text", ""), "kind": "link"}
             for src, es in linkmap.items() for e in es]

    g = {
        "nodes": nodes,
        "edges": edges,
        "stats": {"files": len(nodes), "links": len(edges),
                  "stale": sum(1 for n in nodes if n["freshness"] == "stale"),
                  "lagging": sum(1 for n in nodes if n["lagging"])},
    }
    if write:
        d = root / _DIR
        d.mkdir(exist_ok=True)
        (d / "graph.json").write_text(json.dumps(g, ensure_ascii=False), encoding="utf-8")
    return g


def consistency(root, verify: bool = False, max_conflicts: int = 200) -> dict:
    """Contradictions across the corpus: deterministic typed-value clashes always; non-numeric
    prose contradictions (LLM-judged, opt-in, cheap on DeepSeek) when verify=True."""
    from . import radar as _radar
    from .index import Index
    idx = Index.load(root)
    out = {"conflicts": [], "verified": bool(verify)}
    conflicts = _radar.find_conflicts(idx.passages, idx.matrix, max_conflicts=max_conflicts)
    out["conflicts"] = list(conflicts)
    if verify and hasattr(_radar, "find_prose_conflicts"):
        try:
            out["conflicts"].extend(_radar.find_prose_conflicts(idx.passages, idx.matrix))
        except Exception as e:
            out["prose_error"] = str(e)
    return out
