"""The leak lint — Concord's deterministic, recall-complete layer (M0).

Scans every PUBLIC file line-by-line for banned wording (exact + regex terms).
No embeddings, no language model, zero token cost. This is the layer that runs
in CI / a pre-commit hook and the one whose recall is provably 1 on the known
list. Semantic (paraphrase) terms are handled separately by the find engine.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .rules import Ruleset
from .visibility import classify

_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".concord"}


@dataclass
class Finding:
    term_id: str
    severity: str
    reason: str
    file: str
    line: int
    col: int
    snippet: str
    match_type: str = "exact"

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}: [{self.severity}] {self.term_id} — {self.reason}"


def _iter_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            yield Path(dirpath) / fn


def _read_text(path: Path, max_bytes: int = 2_000_000) -> Optional[str]:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None  # binary or unreadable — skip


def lint_repo(
    root: "str | Path",
    ruleset: Ruleset,
    scope: Iterable[str] = ("public",),
) -> List[Finding]:
    """Scan `root` and return findings where a banned term reaches a scoped file.

    `scope` is the set of visibility categories to lint (default: public only).
    """
    root = Path(root)
    scope = set(scope)
    terms = ruleset.exact_terms()
    compiled = [(t, t.compiled()) for t in terms]
    findings: List[Finding] = []

    for path in _iter_files(root):
        rel = path.relative_to(root).as_posix()
        if classify(rel, ruleset.visibility) not in scope:
            continue
        text = _read_text(path)
        if text is None:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for term, rx in compiled:
                if rx is None:
                    continue
                m = rx.search(line)
                if m:
                    findings.append(
                        Finding(
                            term_id=term.id,
                            severity=term.severity,
                            reason=term.reason,
                            file=rel,
                            line=lineno,
                            col=m.start() + 1,
                            snippet=line.strip()[:160],
                            match_type=term.match,
                        )
                    )
    findings.sort(key=lambda f: (f.severity != "error", f.file, f.line))
    return findings
