"""Concord — keep a sprawling repo telling one story.

Deterministic codename-leak lint + semantic retrieval over a repo's prose.
The engine is computed, not generated: lint is regex, ranking is geometry,
and a language model only ever sees the passages Concord already selected.
"""
from __future__ import annotations

__version__ = "0.0.1"

from .rules import Ruleset, Term, load_ruleset
from .visibility import classify
from .lint import Finding, lint_repo
from .chunk import Passage, chunk_file, chunk_repo
from .find import Hit, find

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
    "__version__",
]
