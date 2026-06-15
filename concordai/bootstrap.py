"""Repo bootstrapping -- make the private ruleset and the built index uncommittable.

Concord's whole premise is that the ruleset (your real codenames) must never enter
version control. `concord init` enforces that mechanically rather than trusting the
user to remember: it scaffolds rules.yaml and ensures the root .gitignore covers it.
The built index protects itself separately (Index.save writes .concord/.gitignore).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

_GITIGNORE_LINES = ["rules.yaml", "rules.local.yaml", "*.local.yaml", ".concord/"]
_HEADER = "# Concord -- private ruleset + built index (never commit these)"


def ensure_gitignore(root: "str | Path") -> List[str]:
    """Append any missing Concord ignore lines to <root>/.gitignore. Returns the
    lines added (empty if already covered)."""
    gi = Path(root) / ".gitignore"
    text = gi.read_text(encoding="utf-8") if gi.exists() else ""
    present = {ln.strip() for ln in text.splitlines()}
    missing = [ln for ln in _GITIGNORE_LINES if ln not in present]
    if missing:
        sep = "" if text == "" or text.endswith("\n") else "\n"
        block = sep + "\n" + _HEADER + "\n" + "\n".join(missing) + "\n"
        gi.write_text(text + block, encoding="utf-8")
    return missing


def scaffold_rules(root: "str | Path") -> Tuple[Path, bool]:
    """Create rules.yaml from the shipped example if absent. Returns (path, created)."""
    dst = Path(root) / "rules.yaml"
    if dst.exists():
        return dst, False
    src = Path(__file__).parent / "rules.example.yaml"
    shutil.copyfile(src, dst)
    return dst, True
