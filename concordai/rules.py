"""Ruleset model and loader.

A ruleset declares (a) which files are publicly shippable and (b) the wording
that must not appear in them. The real ruleset is private (gitignored); only the
generic example ships.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

Severity = str  # "error" | "warn"
MatchKind = str  # "exact" | "regex" | "semantic"


@dataclass
class Term:
    id: str
    match: MatchKind
    reason: str
    severity: Severity = "error"
    pattern: Optional[str] = None          # for exact / regex
    anchors: List[str] = field(default_factory=list)  # for semantic
    ignore_case: bool = False

    def compiled(self) -> Optional["re.Pattern[str]"]:
        """A regex that matches this term, or None for semantic-only terms."""
        flags = re.IGNORECASE if self.ignore_case else 0
        if self.match == "regex":
            return re.compile(self.pattern or "", flags)
        if self.match == "exact":
            return re.compile(re.escape(self.pattern or ""), flags)
        return None  # semantic — handled by the embedding channel


@dataclass
class Ruleset:
    visibility: Dict[str, List[str]] = field(default_factory=dict)
    terms: List[Term] = field(default_factory=list)

    def exact_terms(self) -> List[Term]:
        return [t for t in self.terms if t.match in ("exact", "regex")]

    def semantic_terms(self) -> List[Term]:
        return [t for t in self.terms if t.match == "semantic"]


def load_ruleset(path: "str | Path") -> Ruleset:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    terms = [
        Term(
            id=t["id"],
            match=t.get("match", "exact"),
            reason=t.get("reason", ""),
            severity=t.get("severity", "error"),
            pattern=t.get("pattern"),
            anchors=list(t.get("anchors", []) or []),
            ignore_case=bool(t.get("ignore_case", False)),
        )
        for t in (data.get("terms") or [])
    ]
    return Ruleset(visibility=data.get("visibility", {}) or {}, terms=terms)
