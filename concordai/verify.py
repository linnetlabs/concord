"""LLM verification + resolution of contradiction candidates.

The radar finds same-topic/different-value CANDIDATES deterministically. This asks an
LLM (grounded on the cited passages) to judge which are GENUINE contradictions, name
the canonical value, and say which passage to change -- then `resolve` can auto-apply
the fix. Best-effort: with no API key, verify() returns None and candidates stay raw.

The loop is deterministic find -> grounded judgment -> a human (or --apply) commits
the fix. The model never invents; it judges only the cited text.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import llmlabel


def verify(conflicts):
    """Return verdicts aligned to `conflicts`, or None if no LLM available.

    verdict = {"real": bool, "canonical": str|None, "change": "a"|"b"|None, "why": str}
    """
    if not conflicts:
        return []
    if not llmlabel.available():
        return None
    items = []
    for i, c in enumerate(conflicts):
        items.append(
            f"{i}.\n"
            f"  A [{c['a']['file']}:{c['a']['line']}]: {c['a']['text'][:220]}\n"
            f"  B [{c['b']['file']}:{c['b']['line']}]: {c['b']['text'][:220]}\n"
            f"  values in conflict: {', '.join(c['clash'])}"
        )
    prompt = (
        "For each numbered pair, decide if it is a GENUINE contradiction: A and B describe "
        "the SAME specific fact about the SAME subject, but with conflicting numbers. "
        "Different products, pricing tiers, stages, or distinct policies legitimately have "
        "different numbers and are NOT contradictions. Judge each on its own merits. For a "
        "genuine one, give the canonical value (copied exactly from the conflict values) and "
        "which passage must change ('a' or 'b'). Reply ONLY as a JSON array, one object per "
        "item in order:\n"
        '{"real": true/false, "canonical": "<value or null>", "change": "a"/"b"/null, "why": "<short reason>"}\n\n'
        + "\n\n".join(items)
    )
    raw = llmlabel._llm(prompt, max_tokens=2500)
    if not raw:
        return None
    a, b = raw.find("["), raw.rfind("]")
    if a < 0 or b <= a:
        return None
    try:
        arr = json.loads(raw[a:b + 1])
        if not isinstance(arr, list):
            return None
        # pad/truncate to exactly len(conflicts); anything missing -> treated as not real
        return (list(arr) + [{"real": False}] * len(conflicts))[:len(conflicts)]
    except Exception:
        return None


def verify_prose(candidates):
    """Judge prose contradiction candidates (no numeric clash) via LLM.

    Returns aligned list of {real, why} dicts, or None if no LLM available.
    Only pairs where real=True are genuine prose contradictions.
    """
    if not candidates:
        return []
    if not llmlabel.available():
        return None
    items = []
    for i, c in enumerate(candidates):
        items.append(
            f"{i}.\n"
            f"  A [{c['a']['file']}:{c['a']['line']}]: {c['a']['text'][:220]}\n"
            f"  B [{c['b']['file']}:{c['b']['line']}]: {c['b']['text'][:220]}"
        )
    prompt = (
        "For each numbered pair, decide if it is a GENUINE prose contradiction: A and B make "
        "conflicting factual claims about the SAME specific subject with no numeric mismatch "
        "(e.g. 'SOC2 certified' vs 'SOC2 in progress', 'open source' vs 'proprietary', "
        "'deprecated' vs 'recommended'). Different scopes, products, time periods, or "
        "conditions that legitimately differ are NOT contradictions. Judge each on its own "
        "merits. Reply ONLY as a JSON array, one object per item in order:\n"
        '{"real": true/false, "why": "<one short sentence reason>"}\n\n'
        + "\n\n".join(items)
    )
    raw = llmlabel._llm(prompt, max_tokens=1500)
    if not raw:
        return None
    a, b = raw.find("["), raw.rfind("]")
    if a < 0 or b <= a:
        return None
    try:
        arr = json.loads(raw[a:b + 1])
        if not isinstance(arr, list):
            return None
        return (list(arr) + [{"real": False}] * len(candidates))[:len(candidates)]
    except Exception:
        return None


def _flex(v: str) -> str:
    """Whitespace/operator-tolerant regex for a normalised value, with boundaries so
    '$49' doesn't match inside '$490' and 'n>=4' matches 'n >= 4'."""
    parts, i = [], 0
    while i < len(v):
        if v[i] in ">=>=":
            while i < len(v) and v[i] in ">=>=":
                i += 1
            parts.append(r"[>=>]=?")
        else:
            parts.append(re.escape(v[i]))
            i += 1
    return r"(?<![\w.])" + r"\s*".join(parts) + r"(?![\d])"


def apply_fix(root, side, canonical, conflict, dry_run=False):
    """Replace the conflicting value on the to-change side with `canonical`.

    Returns (file, lineno, before, after) or None if nothing matched. Searches the
    side's passage line and the few lines after it for a conflicting value to swap.
    """
    s = conflict["a"] if side == "a" else conflict["b"]
    other = conflict["b"] if side == "a" else conflict["a"]
    # the value to replace = a conflicting value present on this side but not canonical
    candidates = [v for v in s["values"] if v in conflict["clash"] and v.lower() != canonical.lower()]
    if not candidates:
        candidates = [v for v in conflict["clash"] if v.lower() != canonical.lower()]
    path = Path(root) / s["file"]
    try:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        return None
    start = max(0, s["line"] - 1)
    for lineno in range(start, min(len(lines), start + 20)):
        for old in candidates:
            m = re.search(_flex(old), lines[lineno], re.I)
            if m:
                before = lines[lineno].rstrip("\n")
                new_line = lines[lineno][:m.start()] + canonical + lines[lineno][m.end():]
                if not dry_run:
                    lines[lineno] = new_line
                    path.write_text("".join(lines), encoding="utf-8")
                return (s["file"], lineno + 1, before.strip(), new_line.rstrip("\n").strip())
    return None
