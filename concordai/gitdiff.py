"""Git-diff driver for incremental index updates.

The index records the commit it was built at (`.concord/meta.json`). `concord update`
asks git what changed since then -- or, in a post-commit hook, just what the last
commit touched -- so re-embedding cost scales with the diff, not the whole corpus.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


def _git(root, *args) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, check=False,
        ).stdout
    except OSError:
        return ""


def head(root) -> Optional[str]:
    sha = _git(root, "rev-parse", "HEAD").strip()
    return sha or None


def _parse_name_status(out: str, changed: List[str], deleted: List[str]) -> None:
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status, path = parts[0], parts[-1]  # last field handles renames (R100 old new)
        (deleted if status.startswith("D") else changed).append(path)


def changed_files(root, since: Optional[str] = None, last_commit: bool = False) -> Tuple[List[str], List[str]]:
    """Return (changed_or_added, deleted) repo-relative paths.

    - last_commit=True: just what HEAD~1..HEAD touched (the post-commit-hook case).
    - since=<sha>:       committed changes since the indexed commit, plus the dirty
                         working tree, plus untracked files.
    - neither:           the working tree vs HEAD.
    """
    changed: List[str] = []
    deleted: List[str] = []
    if last_commit:
        _parse_name_status(_git(root, "diff", "--name-status", "HEAD~1", "HEAD"), changed, deleted)
    else:
        if since:
            _parse_name_status(_git(root, "diff", "--name-status", since, "HEAD"), changed, deleted)
        _parse_name_status(_git(root, "diff", "--name-status", "HEAD"), changed, deleted)
        for f in _git(root, "ls-files", "--others", "--exclude-standard").splitlines():
            if f.strip():
                changed.append(f.strip())

    # dedupe, preserve order; a path that was both changed and deleted counts deleted
    deleted = list(dict.fromkeys(deleted))
    dset = set(deleted)
    changed = [c for c in dict.fromkeys(changed) if c not in dset]
    return changed, deleted
