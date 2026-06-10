"""Passage chunking — split prose into paragraph-level passages with line spans.

Passages are the unit Concord embeds and retrieves. Keeping line spans means
every hit is citable as file:line, like a grep result.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .rules import Ruleset
from .visibility import classify

_TEXT_EXTS = {".md", ".markdown", ".txt", ".html", ".htm", ".rst", ".mdx"}
_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".concord"}


@dataclass
class Passage:
    file: str
    start_line: int
    end_line: int
    text: str
    visibility: str = "unknown"


def chunk_file(path: "str | Path", rel: Optional[str] = None, visibility: str = "unknown") -> List[Passage]:
    """Split one file into passages on blank-line boundaries."""
    path = Path(path)
    rel = rel or path.as_posix()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return []

    passages: List[Passage] = []
    buf: List[str] = []
    start = 1
    for i, line in enumerate(lines, start=1):
        if line.strip() == "":
            if buf:
                passages.append(Passage(rel, start, i - 1, "\n".join(buf), visibility))
                buf = []
            start = i + 1
        else:
            if not buf:
                start = i
            buf.append(line)
    if buf:
        passages.append(Passage(rel, start, len(lines), "\n".join(buf), visibility))
    return passages


def chunk_repo(root: "str | Path", ruleset: Optional[Ruleset] = None) -> Iterable[Passage]:
    """Yield passages for every text file under `root`, tagged with visibility."""
    root = Path(root)
    vis = ruleset.visibility if ruleset else {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if Path(fn).suffix.lower() not in _TEXT_EXTS:
                continue
            p = Path(dirpath) / fn
            rel = p.relative_to(root).as_posix()
            category = classify(rel, vis)
            if category == "data":
                continue
            for passage in chunk_file(p, rel=rel, visibility=category):
                yield passage
