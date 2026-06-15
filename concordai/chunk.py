"""Passage chunking -- turn files into citable passages.

Two modes:
- raw (default): split on blank lines, oversized blocks capped. The leak-lint uses
  this -- it must see every byte, so a codename can't hide in markup or a string.
- prose (`prose=True`): structure-aware extraction via `extract.py` -- visible HTML
  text, code comments/strings/gating constants, config values -- the meaningful units
  the semantic index and contradiction radar should reason over. The index uses this.

Either way every passage keeps its file:line span, so a hit is citable like grep.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from . import extract as _extract
from .rules import Ruleset
from .visibility import classify

# Extensions Concord indexes, sourced from the extractor registry (adding a language
# = registering an extractor). The lint scans these too, but raw.
_TEXT_EXTS = frozenset(_extract.supported_extensions())
_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".concord"}

# Cap passage size so a dense block can't bloat retrieval or overrun the embedder's
# context window. Normal units fall well under this; oversized ones sub-split on line
# boundaries, line spans preserved.
_MAX_PASSAGE_CHARS = 1200


@dataclass
class Passage:
    file: str
    start_line: int
    end_line: int
    text: str
    visibility: str = "unknown"


def _emit(passages: List[Passage], rel: str, start: int, lines: List[str], visibility: str) -> None:
    """Append `lines` as one passage, or several capped at _MAX_PASSAGE_CHARS, packing
    lines greedily and keeping accurate start/end line numbers."""
    if not lines:
        return
    if len("\n".join(lines)) <= _MAX_PASSAGE_CHARS:
        passages.append(Passage(rel, start, start + len(lines) - 1, "\n".join(lines), visibility))
        return
    buf: List[str] = []
    buf_start, size = start, 0
    for off, ln in enumerate(lines):
        if buf and size + len(ln) + 1 > _MAX_PASSAGE_CHARS:
            passages.append(Passage(rel, buf_start, start + off - 1, "\n".join(buf), visibility))
            buf, buf_start, size = [], start + off, 0
        buf.append(ln)
        size += len(ln) + 1
    if buf:
        passages.append(Passage(rel, buf_start, start + len(lines) - 1, "\n".join(buf), visibility))


def _blocks_to_passages(blocks, rel: str, visibility: str) -> List[Passage]:
    """Wrap extractor blocks (text, start, end) as Passages, capping oversized ones."""
    out: List[Passage] = []
    for txt, s, e in blocks:
        if len(txt) <= _MAX_PASSAGE_CHARS:
            out.append(Passage(rel, s, e, txt, visibility))
        else:
            _emit(out, rel, s, txt.split("\n"), visibility)
    return out


def chunk_file(path: "str | Path", rel: Optional[str] = None, visibility: str = "unknown",
               prose: bool = False) -> List[Passage]:
    """Passages for one file. `prose=True` uses structure-aware extraction; otherwise
    raw blank-line chunking (what the leak-lint needs)."""
    path = Path(path)
    rel = rel or path.as_posix()
    try:
        raw = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    if prose:
        return _blocks_to_passages(_extract.extract(raw, path.suffix), rel, visibility)

    passages: List[Passage] = []
    buf: List[str] = []
    start = 1
    for i, line in enumerate(raw.splitlines(), start=1):
        if line.strip() == "":
            _emit(passages, rel, start, buf, visibility)
            buf = []
            start = i + 1
        else:
            if not buf:
                start = i
            buf.append(line)
    _emit(passages, rel, start, buf, visibility)
    return passages


def chunk_repo(root: "str | Path", ruleset: Optional[Ruleset] = None,
               prose: bool = False) -> Iterable[Passage]:
    """Yield passages for every indexed file under `root`, tagged with visibility."""
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
            for passage in chunk_file(p, rel=rel, visibility=category, prose=prose):
                yield passage
