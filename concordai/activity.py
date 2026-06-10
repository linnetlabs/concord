"""Activity radar — where dev effort goes, and where edits collide, from git.

Per-file churn and how many authors touched each file in the window (a concurrent-edit
/ merge-conflict signal), mapped onto the repo's directory tree so the explorer can
render it as a heat-map treemap (size = effort, colour = how many authors).
"""
from __future__ import annotations

import subprocess
from collections import defaultdict


_SKIP_DIRS = ("graphify-out/", ".concord/", "node_modules/", "dist/", "build/", "vendor/")


def _skip(path: str) -> bool:
    if any(path.startswith(d) or ("/" + d) in path for d in _SKIP_DIRS):
        return True
    return path.endswith((".lock", ".min.js", ".map", "-lock.json", "package-lock.json"))


def _git(root, *a) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout
    except OSError:
        return ""


def file_activity(root, since: str = "3 months ago", max_files: int = 500):
    """Per-file {file, commits, authors, author_names, churn, last} over the window."""
    out = _git(root, "log", f"--since={since}", "--no-merges", "--numstat",
               "--format=__C__|%h|%an|%ad", "--date=short")
    files = defaultdict(lambda: {"commits": set(), "authors": set(), "churn": 0, "last": ""})
    h = a = d = None
    for line in out.splitlines():
        if line.startswith("__C__|"):
            _, h, a, d = line.split("|", 3)
        elif "\t" in line and h:
            parts = line.split("\t")
            if len(parts) >= 3:
                add, rem, path = parts[0], parts[1], parts[2]
                if _skip(path):
                    continue
                f = files[path]
                f["commits"].add(h)
                f["authors"].add(a)
                f["churn"] += (int(add) if add.isdigit() else 0) + (int(rem) if rem.isdigit() else 0)
                if not f["last"]:
                    f["last"] = d
    rows = [{"file": p, "commits": len(v["commits"]), "authors": len(v["authors"]),
             "author_names": sorted(v["authors"]), "churn": v["churn"], "last": v["last"]}
            for p, v in files.items()]
    rows.sort(key=lambda x: -x["churn"])
    return rows[:max_files]


def tree_nodes(files):
    """Directory-tree nodes for a treemap: leaf value=churn (size), authors=heat
    (colour), with directories aggregating churn and taking the max collision of
    their children. branchvalues='total' compatible (parents = sum of leaf churn)."""
    agg = defaultdict(lambda: {"churn": 0, "authors": 0, "commits": 0, "leaf": False})
    for f in files:
        parts = f["file"].split("/")
        node = agg[f["file"]]
        node.update(churn=f["churn"], authors=f["authors"], commits=f["commits"], leaf=True)
        for i in range(1, len(parts)):  # ancestors
            anc = agg["/".join(parts[:i])]
            anc["churn"] += f["churn"]
            anc["authors"] = max(anc["authors"], f["authors"])
            anc["commits"] += f["commits"]
    nodes = []
    for nid, v in agg.items():
        parts = nid.split("/")
        nodes.append({"id": nid, "label": parts[-1],
                      "parent": "/".join(parts[:-1]) if len(parts) > 1 else "",
                      "churn": v["churn"], "authors": v["authors"], "commits": v["commits"], "leaf": v["leaf"]})
    return nodes
