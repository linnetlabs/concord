"""Concord: keep a sprawling repo telling one story.

Deterministic codename-leak lint plus semantic retrieval over a repo's prose.
The core is deterministic: lint is regex, ranking is geometry, and a language
model only ever sees the passages Concord already selected.
"""
from __future__ import annotations

__version__ = "0.0.1"

from .rules import Ruleset, Term, load_ruleset
from .visibility import classify
from .lint import Finding, lint_repo
from .chunk import Passage, chunk_file, chunk_repo
from .find import Hit, find, find_all
from .links import links_in, scan as scan_links
from .graph import graph, consistency, coverage, to_mermaid, to_dot, freshness_map

__all__ = [
    "Ruleset",
    "Term",
    "load_ruleset",
    "classify",
    "Finding",
    "lint_repo",
    "Passage",
    "chunk_file",
    "chunk_repo",
    "Hit",
    "find",
    "find_all",
    "links_in",
    "scan_links",
    "graph",
    "consistency",
    "coverage",
    "to_mermaid",
    "to_dot",
    "freshness_map",
    "__version__",
]
