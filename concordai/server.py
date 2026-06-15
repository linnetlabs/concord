"""concordai.server -- the live explorer.

A tiny stdlib HTTP server (no framework, no Streamlit) that loads a built index +
the sentiment.ai embedder once, then serves real semantic search / topic / passage
queries as JSON to a bespoke browser UI. Launched by `concord ui`; local only.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_STATE: dict = {}
_EMB_LOCK = threading.Lock()  # torch model is loaded once; serialise inference across handler threads


def _embed(texts, kind="query"):
    with _EMB_LOCK:
        return _STATE["emb"].embed(texts, kind=kind)


def _get_radar(idx):
    """Compute the radar once, persist it (self-ignored .concord), reuse thereafter."""
    if _STATE.get("radar") is None:
        from . import radar
        _STATE["radar"] = radar.find_conflicts(idx.passages, idx.matrix)
        try:
            (Path(_STATE["root"]) / ".concord" / "radar.json").write_text(
                json.dumps({"commit": idx.meta.get("commit"), "radar": _STATE["radar"]}), encoding="utf-8")
        except Exception:
            pass
    return _STATE["radar"]


def _git_base(root: str) -> str:
    """https://github.com/owner/repo/blob/<branch> for line-deep links, or ''."""
    def g(*a):
        try:
            return subprocess.run(["git", "-C", root, *a], capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    url = g("config", "--get", "remote.origin.url")
    m = re.search(r"github\.com[:/]+([^/]+)/([^/.]+)", url or "")
    if not m:
        return ""
    return f"https://github.com/{m.group(1)}/{m.group(2)}/blob/{g('rev-parse', '--abbrev-ref', 'HEAD') or 'main'}"


def drift(root: str, term: str, n: int = 25) -> list:
    """Commits where `term` was added/removed (git pickaxe) -- how a fact evolved."""
    if not term.strip():
        return []
    try:
        out = subprocess.run(
            ["git", "-C", root, "log", "-S", term, "--format=%h|%ad|%s", "--date=short", "-n", str(n)],
            capture_output=True, text=True,
        ).stdout
    except OSError:
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            rows.append({"hash": parts[0], "date": parts[1], "subject": parts[2][:90]})
    return rows


def file_hist(root: str, file: str, months: int = 12) -> list:
    """Commits touching `file` bucketed into the last `months` calendar months -- a tiny
    activity sparkline for the graph inspector (how recently/often a file has changed)."""
    if not file:
        return []
    try:
        out = subprocess.run(
            ["git", "-C", root, "log", "--format=%ad", "--date=format:%Y-%m", "--", file],
            capture_output=True, text=True).stdout
    except OSError:
        return []
    counts = Counter(l.strip() for l in out.splitlines() if l.strip())
    import datetime as _dt
    y, m, keys = _dt.date.today().year, _dt.date.today().month, []
    for _ in range(months):
        keys.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    keys.reverse()
    return [{"month": k, "commits": counts.get(k, 0)} for k in keys]


def _load(root: str) -> None:
    from .embed import get_embedder
    from .index import Index
    idx = Index.load(root)
    if idx.matrix is None:
        raise RuntimeError(f"No semantic index at {root}/.concord -- run `concord index {root}` first.")
    # ~tokens to read the WHOLE corpus into context -- the naive baseline Concord avoids.
    # chars/4 is the standard rough token proxy; honest, model-agnostic, computed once.
    corpus_tokens = sum(len(p.text) for p in idx.passages) // 4
    _STATE.update(root=root, idx=idx, emb=get_embedder(idx.meta.get("model")), cl=None,
                  ghbase=_git_base(root), label_cache={}, corpus_tokens=corpus_tokens)
    _STATE["emb"].embed(["warmup"], kind="query")  # materialise the torch model in the MAIN thread
    # load persisted radar + topic labels (keyed by commit, so they survive restarts)
    commit, cdir = idx.meta.get("commit"), Path(root) / ".concord"
    try:
        d = json.loads((cdir / "radar.json").read_text("utf-8"))
        if d.get("commit") == commit:
            _STATE["radar"] = d["radar"]
    except Exception:
        pass
    try:
        d = json.loads((cdir / "labels.json").read_text("utf-8"))
        if d.get("commit") == commit:
            _STATE["label_cache"] = d.get("byk", {})
    except Exception:
        pass


def _clustering(k=28, s=6):
    if _STATE.get("cl") is None:
        from . import cluster as C
        idx = _STATE["idx"]
        _STATE["cl"] = C.cluster(idx.matrix, [p.text for p in idx.passages], k_leaves=k, n_super=s)
    return _STATE["cl"]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj).encode(), "application/json", code)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        idx = _STATE["idx"]

        if u.path in ("/", "/index.html"):
            html = (Path(__file__).parent / "explorer.html").read_text(encoding="utf-8")
            repo = os.path.basename(_STATE["root"].rstrip("/")) or _STATE["root"]
            from collections import Counter
            exts = Counter(os.path.splitext(p.file)[1].lower() or "?" for p in idx.passages)
            types = ", ".join(f"{n}x{e}" for e, n in exts.most_common(12))
            html = (html.replace("__REPO__", repo).replace("__N__", f"{len(idx.passages):,}")
                        .replace("__TYPES__", types or "--")
                        .replace("__GHBASE__", _STATE.get("ghbase", "")))
            return self._send(html.encode(), "text/html; charset=utf-8")

        if u.path == "/api/search":
            import numpy as np
            queries = [x for x in ([q.get("q", [""])[0]] + q.get("also", [])) if x.strip()]
            if not queries:
                return self._json({"hits": [], "total": len(idx.passages)})
            try:
                qv = np.asarray(_embed(queries, "query"), dtype="float32")
                qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
                sims = (idx.matrix @ qv.T).max(axis=1)  # best phrasing per passage
                rows = [(int(i), float(sims[int(i)])) for i in np.argsort(-sims)[:int(q.get("k", ["25"])[0])]]
            except Exception as e:  # surface, never silently return empty
                return self._json({"hits": [], "error": repr(e)})
            facet = None
            if q.get("facets", ["0"])[0] == "1" and len(rows) >= 4:
                try:
                    from . import cluster as C
                    texts = [idx.passages[i].text for i, _ in rows]
                    facet = C.facet_labels(texts, _embed(texts, "passage"))
                except Exception:
                    facet = None
            hits = []
            read_tokens = 0
            for n, (i, sc) in enumerate(rows):
                p = idx.passages[i]
                read_tokens += len(p.text) // 4  # what the model actually reads to answer
                hits.append({"file": p.file, "line": p.start_line, "score": round(sc, 3),
                             "type": "semantic", "facet": (facet[n] if facet else None),
                             "text": " ".join(p.text.split())[:400]})
            return self._json({"hits": hits, "total": len(idx.passages),
                               "read_tokens": read_tokens, "corpus_tokens": _STATE.get("corpus_tokens", 0)})

        if u.path == "/api/topics":
            import numpy as np
            cl = _clustering(int(q.get("k", ["28"])[0]), int(q.get("s", ["6"])[0]))
            byleaf = defaultdict(list)
            for i, lf in enumerate(cl.leaf_of):
                byleaf[int(lf)].append(i)

            # topic labels: clean LLM names (best-effort, cached) else tf-idf keyword bags
            leaf_labels = list(cl.leaf_labels)
            super_labels = list(cl.super_labels)
            # LLM labels are OPT-IN (?llm=1) -- they call the user's PAID API. A cached
            # result (already paid) is reused for free regardless.
            want_llm = q.get("llm", ["0"])[0] == "1"
            cache, ck = _STATE.setdefault("label_cache", {}), str(cl.k)
            names = cache.get(ck)
            if names is None and want_llm:
                from . import llmlabel
                samples = []
                for L in range(cl.k):
                    c = cl.leaf_centroids[L]
                    c = c / (np.linalg.norm(c) + 1e-9)
                    near = sorted(byleaf[L], key=lambda i: -float(idx.matrix[i] @ c))[:3]
                    samples.append([idx.passages[i].text for i in near])
                names = llmlabel.label_clusters(samples)
                if names:
                    cache[ck] = names
                    try:
                        (Path(_STATE["root"]) / ".concord" / "labels.json").write_text(
                            json.dumps({"commit": idx.meta.get("commit"), "byk": cache}), encoding="utf-8")
                    except Exception:
                        pass
            if names:
                leaf_labels = (list(names) + list(cl.leaf_labels[len(names):]))[:cl.k]
                big = {}  # name each theme from its largest leaf's label
                for L in range(cl.k):
                    s = int(cl.super_of_leaf[L])
                    if s not in big or len(byleaf[L]) > big[s][1]:
                        big[s] = (L, len(byleaf[L]))
                for s, (L, _) in big.items():
                    super_labels[s] = leaf_labels[L]

            nodes = []
            for sidx in range(len(super_labels)):
                nodes.append({"id": f"S{sidx}", "label": super_labels[sidx], "parent": "", "value": 0})
            for L in range(cl.k):
                nodes.append({"id": f"L{L}", "label": leaf_labels[L], "parent": f"S{int(cl.super_of_leaf[L])}", "value": 0})
                for fp, cnt in Counter(idx.passages[i].file for i in byleaf[L]).most_common():
                    nodes.append({"id": f"F{L}|{fp}", "label": fp.split("/")[-1], "parent": f"L{L}",
                                  "value": cnt, "file": fp, "leaf": L})
            return self._json({"nodes": nodes, "llm": names is not None})

        if u.path == "/api/passages":
            cl = _clustering()
            leaf = int(q.get("leaf", ["-1"])[0])
            fp = q.get("file", [""])[0]
            out = []
            for i, lf in enumerate(cl.leaf_of):
                p = idx.passages[i]
                if int(lf) == leaf and p.file == fp:
                    out.append({"file": p.file, "line": p.start_line, "text": " ".join(p.text.split())[:500]})
            return self._json({"passages": out[:50]})

        if u.path == "/api/radar":
            return self._json(_get_radar(idx))

        if u.path == "/api/llm":
            from . import llmlabel
            return self._json(llmlabel.status())

        if u.path == "/api/llm/set":
            from . import llmlabel
            llmlabel.set_provider(q.get("provider", ["auto"])[0])
            _STATE["label_cache"] = {}  # provider changed -- re-name on demand with the new one
            return self._json(llmlabel.status())

        if u.path == "/api/verify":
            from . import verify as V
            return self._json({"verdicts": V.verify(_get_radar(idx)["conflicts"][:10])})

        if u.path == "/api/activity":
            from . import activity
            since = q.get("since", ["3 months ago"])[0]
            files = activity.file_activity(_STATE["root"], since=since)
            return self._json({"tree": activity.tree_nodes(files), "files": files[:50],
                               "collisions": sum(1 for f in files if f["authors"] >= 2), "since": since})

        if u.path == "/api/drift":
            return self._json({"commits": drift(_STATE["root"], q.get("term", [""])[0])})

        if u.path == "/api/graph":
            from . import graph as G
            if _STATE.get("graph") is None:
                _STATE["graph"] = G.graph(_STATE["root"], write=True)
            return self._json(_STATE["graph"])

        if u.path == "/api/coverage":
            from . import graph as G
            if _STATE.get("graph") is None:
                _STATE["graph"] = G.graph(_STATE["root"], write=True)
            return self._json(G.coverage(_STATE["root"], _STATE["graph"]))

        if u.path == "/api/filehist":
            return self._json({"file": q.get("file", [""])[0],
                               "months": file_hist(_STATE["root"], q.get("file", [""])[0])})

        self.send_response(404)
        self.end_headers()


def make_server(root: str, port: int = 8765) -> ThreadingHTTPServer:
    """Load the index + embedder (slow: model load), then return a bound server."""
    _load(root)
    return ThreadingHTTPServer(("127.0.0.1", port), _Handler)
