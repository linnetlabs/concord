"""Content-hash manifest — change detection without git.

`concord update` prefers git (it already knows the diff for free). But in a non-git
folder there is no diff source, so the index also records a manifest of file content
hashes in `.concord/manifest.json`. On update we re-scan (cheap: stat + read the
~text files, no embedding) and diff hashes; only changed files get re-embedded.

A content hash beats mtime: a bare `touch` or a checkout that rewrites mtimes won't
trigger a needless re-embed, and a real edit always will.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

from .chunk import _SKIP_DIRS, _TEXT_EXTS


def hash_file(path: Path) -> str:
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


def scan(root: "str | Path") -> Dict[str, str]:
    """Map every indexable text file (repo-relative) to its content hash."""
    root = Path(root)
    out: Dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if Path(fn).suffix.lower() in _TEXT_EXTS:
                p = Path(dirpath) / fn
                out[p.relative_to(root).as_posix()] = hash_file(p)
    return out


def diff(old: Dict[str, str], new: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """(changed_or_added, deleted) by comparing two manifests."""
    changed = [f for f, h in new.items() if old.get(f) != h]
    deleted = [f for f in old if f not in new]
    return changed, deleted
