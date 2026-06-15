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


def freshness_map(g: dict) -> dict:
    """{file: {freshness, lagging, last}} from a graph dict -- the input radar.annotate_canonical wants."""
    return {n["file"]: {"freshness": n["freshness"], "lagging": n["lagging"], "last": n["last"]}
            for n in g["nodes"]}


_PROSE_EXTS = (".md", ".mdx", ".markdown", ".rst", ".txt")


def coverage(root, g: Optional[dict] = None) -> dict:
    """Doc-coverage signal off the library graph: code files with NO inbound doc-link
    (undocumented surface, riskiest where churn is high) and docs that lag the code
    they reference. Deterministic; reads the graph, no model."""
    g = g or graph(root, write=False)
    indeg = Counter()
    for e in g["edges"]:
        indeg[e["target"]] += 1
    undocumented, lagging = [], []
    for n in g["nodes"]:
        f = n["file"]
        ext = ("." + f.rsplit(".", 1)[-1]).lower() if "." in f else ""
        is_doc = ext in _PROSE_EXTS
        if not is_doc and n["passages"] and indeg.get(f, 0) == 0:
            undocumented.append(n)
        if n["lagging"]:
            lagging.append(n)
    undocumented.sort(key=lambda n: -n["churn"])    # high-churn + undocumented = riskiest
    lagging.sort(key=lambda n: n["last"])
    return {"undocumented": undocumented, "lagging": lagging,
            "stats": {"files": g["stats"]["files"],
                      "undocumented": len(undocumented), "lagging": len(lagging)}}


def _short(f: str) -> str:
    return "/".join(f.split("/")[-2:])  # last 2 path segments: disambiguates same-named files


_MERMAID_FILL = {"fresh": "fill:#1f7a3d,color:#fff", "aging": "fill:#b8860b,color:#fff",
                 "stale": "fill:#a32222,color:#fff", "unknown": "fill:#555,color:#fff"}


def to_mermaid(g: dict, max_edges: int = 200) -> str:
    """The doc-link graph as a Mermaid 'graph LR' to paste into a README or PR.
    Nodes are coloured by freshness; lagging nodes get a thick white border."""
    by_file = {n["file"]: n for n in g["nodes"]}
    nid, lines, seen = {}, ["graph LR"], set()

    def _id(f: str) -> str:
        if f not in nid:
            nid[f] = "n%d" % len(nid)
        return nid[f]

    for e in g["edges"][:max_edges]:
        lines.append(f'  {_id(e["source"])}["{_short(e["source"])}"] --> '
                     f'{_id(e["target"])}["{_short(e["target"])}"]')
        seen.add(e["source"]); seen.add(e["target"])
    for f in sorted(seen):
        n = by_file.get(f, {})
        fill = _MERMAID_FILL.get(n.get("freshness", "unknown"), _MERMAID_FILL["unknown"])
        ring = ",stroke:#fff,stroke-width:4px" if n.get("lagging") else ""
        lines.append(f"  style {_id(f)} {fill}{ring}")
    return "\n".join(lines)


_DOT_COLOR = {"fresh": "#1f7a3d", "aging": "#b8860b", "stale": "#a32222", "unknown": "#555555"}


def to_dot(g: dict, max_edges: int = 400) -> str:
    """The doc-link graph as Graphviz DOT (dot -Tsvg)."""
    by_file = {n["file"]: n for n in g["nodes"]}
    lines = ["digraph concord {", '  rankdir=LR; node [shape=box,style=filled,fontname="monospace",fontcolor="white"];']
    seen = set()
    for e in g["edges"][:max_edges]:
        lines.append(f'  "{e["source"]}" -> "{e["target"]}";')
        seen.add(e["source"]); seen.add(e["target"])
    for f in sorted(seen):
        n = by_file.get(f, {})
        c = _DOT_COLOR.get(n.get("freshness", "unknown"), "#555555")
        pen = ",penwidth=3,color=white" if n.get("lagging") else ""
        lines.append(f'  "{f}" [fillcolor="{c}"{pen}];')
    lines.append("}")
    return "\n".join(lines)


def consistency(root, verify: bool = False, max_conflicts: int = 200, canonical: bool = True) -> dict:
    """Contradictions across the corpus: deterministic typed-value clashes always; non-numeric
    prose contradictions (LLM-judged, opt-in, cheap on DeepSeek) when verify=True. When
    canonical=True, each numeric conflict is tagged with a deterministic freshness-based
    suggestion of which side is the source of truth (see radar.pick_canonical)."""
    from . import radar as _radar
    from .index import Index
    idx = Index.load(root)
    out = {"conflicts": [], "verified": bool(verify)}
    conflicts = list(_radar.find_conflicts(idx.passages, idx.matrix, max_conflicts=max_conflicts)["conflicts"])
    if canonical and conflicts:
        try:
            _radar.annotate_canonical(conflicts, freshness_map(graph(root, write=False)))
        except Exception as e:
            out["canonical_error"] = str(e)
    out["conflicts"] = conflicts
    if verify and hasattr(_radar, "find_prose_conflicts"):
        try:
            out["conflicts"].extend(_radar.find_prose_conflicts(idx.passages, idx.matrix))
        except Exception as e:
            out["prose_error"] = str(e)
    return out
