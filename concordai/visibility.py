"""File-visibility classification: public | internal | data | unknown.

The leak lint only scans `public` files. `internal` and `data` are exempt.
Globs support `**` (matches across directory separators) and `*` (within a
path segment), matched against repo-relative POSIX paths.
"""
from __future__ import annotations

import re
from typing import Dict, List

_CACHE: Dict[str, "re.Pattern[str]"] = {}


def _glob_to_regex(glob: str) -> "re.Pattern[str]":
    if glob in _CACHE:
        return _CACHE[glob]
    out: List[str] = []
    i, n = 0, len(glob)
    while i < n:
        c = glob[i]
        if c == "*":
            if i + 1 < n and glob[i + 1] == "*":
                # ** -- span across separators (and an optional trailing slash)
                out.append(".*")
                i += 2
                if i < n and glob[i] == "/":
                    i += 1
                continue
            out.append("[^/]*")  # * -- within a single segment
        elif c == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(c))
        i += 1
    pat = re.compile("^" + "".join(out) + "$")
    _CACHE[glob] = pat
    return pat


def classify(path: str, visibility: Dict[str, List[str]]) -> str:
    """Return the visibility category for a repo-relative POSIX path.

    First match wins in the order public, internal, data. Unmatched paths are
    `unknown` and, like internal, are NOT linted (fail-safe: opt files INTO
    public scope explicitly rather than risk false confidence).
    """
    rel = path.replace("\\", "/").lstrip("./")
    for category in ("public", "internal", "data"):
        for glob in visibility.get(category, []) or []:
            if _glob_to_regex(glob).match(rel):
                return category
    return "unknown"
