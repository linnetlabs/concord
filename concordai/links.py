"""Cross-reference extraction -- the link layer that makes the corpus a GRAPH.

Pulls intra-repo references out of each file (markdown inline/reference links, bare
"see X.md" mentions, and HTML href/src) so the corpus is a document graph, not just a
bag of passages. External URLs, anchors, and mailto are dropped; targets are resolved
to repo-relative paths and kept only when they point at a file that actually exists.

This is deterministic and dependency-free (regex + stdlib HTMLParser), in keeping with
Concord's "the deterministic core stands alone" design.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

_LINK_EXTS = (".md", ".mdx", ".markdown", ".rst", ".txt", ".html", ".htm")
_PROSE_EXTS = (".md", ".mdx", ".markdown", ".rst", ".txt")
# Transient / derived copies that pollute the graph (git worktrees, vendored, build output).
_SKIP_DIRS = (".claude/", ".worktrees/", ".git/", ".concord/", "node_modules/", "dist/", "build/", "vendor/")


def _skip(path: str) -> bool:
    return any(path.startswith(d) or ("/" + d) in path for d in _SKIP_DIRS)

_MD_INLINE = re.compile(r"\[([^\]]*)\]\(\s*<?([^)>\s]+)>?(?:\s+[\"'][^\"']*[\"'])?\s*\)")  # [text](path "title")
_MD_REFDEF = re.compile(r"(?m)^[ \t]*\[([^\]]+)\]:[ \t]*<?(\S+?)>?[ \t]*$")               # [ref]: path
_MD_REFUSE = re.compile(r"\[([^\]]*)\]\[([^\]]+)\]")                                       # [text][ref]
_SEE = re.compile(
    r"\b(?:see|ref|refer to|cf\.?|per|defined in)\s+(?:also\s+)?[`'\"(]?"
    r"([\w./-]+\.(?:md|mdx|markdown|rst|txt))\b", re.I)


def _is_external(href: str) -> bool:
    h = (href or "").strip()
    return (not h) or h.startswith(
        ("http://", "https://", "//", "mailto:", "tel:", "#", "data:", "javascript:"))


def _resolve(src_file: str, href: str) -> Optional[str]:
    """Resolve a raw href found in src_file to a normalised repo-relative path, or None
    if it is external/an anchor or cannot be resolved."""
    h = (href or "").split("#", 1)[0].split("?", 1)[0].strip()
    if _is_external(h):
        return None
    base = PurePosixPath(src_file).parent
    target = PurePosixPath(h.lstrip("/")) if h.startswith("/") else (base / h)
    parts: List[str] = []
    for seg in target.parts:
        if seg == "..":
            if parts:
                parts.pop()
        elif seg not in (".", ""):
            parts.append(seg)
    return "/".join(parts) or None


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: List[str] = []

    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            if not v:
                continue
            if (tag == "a" and k == "href") or (tag in ("img", "script", "link", "iframe") and k in ("src", "href")):
                self.hrefs.append(v)


def links_in(src_file: str, text: str) -> List[dict]:
    """Outbound intra-repo links from one file's text: [{target, text}] (anchor text where known)."""
    ext = PurePosixPath(src_file).suffix.lower()
    raw: List[tuple] = []  # (href, anchor_text)
    if ext in _PROSE_EXTS:
        refdefs = {m.group(1).lower(): m.group(2) for m in _MD_REFDEF.finditer(text)}
        for m in _MD_INLINE.finditer(text):
            raw.append((m.group(2), m.group(1)))
        for m in _MD_REFUSE.finditer(text):
            tgt = refdefs.get(m.group(2).lower())
            if tgt:
                raw.append((tgt, m.group(1)))
        for m in _SEE.finditer(text):
            raw.append((m.group(1), "see"))
    elif ext in (".html", ".htm"):
        p = _HrefParser()
        try:
            p.feed(text)
        except Exception:
            pass
        raw = [(h, "") for h in p.hrefs]

    out, seen = [], set()
    for href, anchor in raw:
        tgt = _resolve(src_file, href)
        if tgt and tgt != src_file and tgt not in seen:
            seen.add(tgt)
            out.append({"target": tgt, "text": (anchor or "").strip()[:80]})
    return out


def scan(root) -> Dict[str, List[dict]]:
    """{source_file: [{target, text}]} for every link-bearing file in the repo. Targets are
    filtered to files that actually exist in the corpus (the manifest), so dead links drop out."""
    from . import manifest as _manifest
    files = {f for f in _manifest.scan(root) if not _skip(f)}
    out: Dict[str, List[dict]] = {}
    for rel in files:
        if PurePosixPath(rel).suffix.lower() not in _LINK_EXTS:
            continue
        try:
            text = (Path(root) / rel).read_text(errors="replace")
        except Exception:
            continue
        ls = [l for l in links_in(rel, text) if l["target"] in files]
        if ls:
            out[rel] = ls
    return out
