"""Contradiction radar -- same-topic passages that state DIFFERENT hard values.

Catches the highest-value drift class deterministically: prices, thresholds,
percentages and durations that disagree between semantically-near passages
(e.g. "$49/seat" vs "$39/seat", "n >= 4" vs "n >= 8"). It is a CANDIDATE list --
same-topic + different-number -- for human review, not a proof of contradiction.
The embedding decides "same topic"; a regex extracts the value; the conflict is
where those two disagree.
"""
from __future__ import annotations

import re

import numpy as np

_TYPES = [
    ("price", re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?[kKmM]?")),          # $49, $8,000, $15k (suffix must attach)
    ("threshold", re.compile(r"n\s?[>=><]=?\s?\d+", re.I)),            # n >= 8, n>=4
    ("percent", re.compile(r"\b\d+(?:\.\d+)?\s?%")),                   # 30%
    ("duration", re.compile(r"\b\d+\s?(?:days?|months?|years?|weeks?)\b", re.I)),  # 30 days
    # a named numeric constant in code/config -- MIN_N = 8, "min_n": 5, maxSeats=100.
    # The identifier rides along in the value so a conflict reads NAME=v vs NAME=v'.
    ("config", re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}[\"']?\s*[:=]\s*[\"']?\$?\d[\d,]*(?:\.\d+)?")),
]


_STOP = set(
    "the a an of to and or for with in on at by is are be this that from your you our "
    "we as it its not will can may any all per each only also more most than then them".split()
)


def _norm(s: str) -> str:
    return re.sub(r"\.0+(?=\D|$)", "", re.sub(r"\s+", "", s).lower())  # $29.00 -> $29


def _typed_values(text: str) -> dict:
    out = {}
    for name, rx in _TYPES:
        vs = frozenset(_norm(m.group(0)) for m in rx.finditer(text))
        if vs:
            out[name] = vs
    return out


def _content(text: str) -> "frozenset[str]":
    return frozenset(t for t in re.findall(r"[a-z]{4,}", text.lower()) if t not in _STOP)


def _conflicting(va: dict, vb: dict):
    """Values of the SAME type that differ with no overlap (e.g. price<->price)."""
    out = []
    for t in set(va) & set(vb):
        if va[t] != vb[t] and not (va[t] & vb[t]):
            out += sorted(va[t] | vb[t])
    return out


def find_conflicts(passages, matrix, sim_threshold: float = 0.88, neighbors: int = 6, max_conflicts: int = 60):
    """Return {value_passages, conflicts:[{sim, a, b}]} -- value-conflict CANDIDATES.

    A pair is a candidate when the two passages are semantically near (cosine >=
    threshold), **share a subject word**, are **not near-identical copies**, yet carry
    disjoint hard values. This is a review queue (same-topic + different-number), not a
    verdict -- a human or an LLM driver confirms which are real contradictions.
    """
    def _flat(v):
        return sorted(x for s in v.values() for x in s)

    rows, vals, cont = [], [], []
    for i, p in enumerate(passages):
        # skip tool/hidden dirs (.claude, .planning). Passages here are already
        # extraction-cleaned, so code/config conflicts (MIN_N=8 vs "min_n":5) are
        # in scope alongside prose -- the extractor dropped the syntax noise.
        if any(part.startswith(".") for part in p.file.split("/")):
            continue
        v = _typed_values(p.text)
        if v:
            rows.append(i)
            vals.append(v)
            cont.append(_content(p.text))
    if len(rows) < 2:
        return {"value_passages": len(rows), "conflicts": []}

    M = np.asarray(matrix[rows], dtype="float32")
    S = M @ M.T
    conflicts, seen = [], set()
    for a in range(len(rows)):
        cnt = 0
        for b in np.argsort(-S[a]):
            if b == a:
                continue
            if S[a][b] < sim_threshold or cnt >= neighbors:
                break
            cnt += 1
            clash = _conflicting(vals[a], vals[b])  # same-TYPE values that differ
            if not clash:
                continue
            shared = cont[a] & cont[b]
            union = cont[a] | cont[b]
            if not shared:
                continue  # different subjects that merely both carry a number
            if union and len(shared) / len(union) > 0.9:
                continue  # near-identical copies -- not a contradiction
            key = (min(rows[a], rows[b]), max(rows[a], rows[b]))
            if key in seen:
                continue
            seen.add(key)
            pa, pb = passages[rows[a]], passages[rows[b]]
            conflicts.append({
                "sim": round(float(S[a][b]), 3),
                "clash": clash,
                "subject": sorted(shared)[:4],
                "a": {"file": pa.file, "line": pa.start_line, "values": _flat(vals[a]),
                      "text": " ".join(pa.text.split())[:240]},
                "b": {"file": pb.file, "line": pb.start_line, "values": _flat(vals[b]),
                      "text": " ".join(pb.text.split())[:240]},
            })
    conflicts.sort(key=lambda c: -c["sim"])
    return {"value_passages": len(rows), "conflicts": conflicts[:max_conflicts]}


def find_prose_conflicts(passages, matrix, sim_threshold: float = 0.88, neighbors: int = 6, max_candidates: int = 30):
    """Return same-topic pairs with NO numeric clash -- candidates for LLM prose-contradiction judging.

    Reuses the same sim-matrix, shared-subject, and near-identical-copy gates as find_conflicts,
    but scans ALL non-hidden passages (not just those carrying typed values) and skips pairs that
    already have a numeric clash (those belong to the deterministic radar). The returned dicts share
    the same shape as find_conflicts output, with clash=[], kind="prose" added. Pass the result to
    verify.verify_prose() to get LLM verdicts.
    """
    rows, cont = [], []
    for i, p in enumerate(passages):
        if any(part.startswith(".") for part in p.file.split("/")):
            continue
        rows.append(i)
        cont.append(_content(p.text))
    if len(rows) < 2:
        return []

    M = np.asarray(matrix[rows], dtype="float32")
    S = M @ M.T
    candidates, seen = [], set()
    for a in range(len(rows)):
        cnt = 0
        for b in np.argsort(-S[a]):
            if b == a:
                continue
            if S[a][b] < sim_threshold or cnt >= neighbors:
                break
            cnt += 1
            key = (min(rows[a], rows[b]), max(rows[a], rows[b]))
            if key in seen:
                continue
            # pairs with a numeric clash are already covered by find_conflicts
            va = _typed_values(passages[rows[a]].text)
            vb = _typed_values(passages[rows[b]].text)
            if _conflicting(va, vb):
                continue
            shared = cont[a] & cont[b]
            union = cont[a] | cont[b]
            if not shared:
                continue  # different subjects
            if union and len(shared) / len(union) > 0.9:
                continue  # near-identical copies -- not a contradiction
            seen.add(key)
            pa, pb = passages[rows[a]], passages[rows[b]]
            candidates.append({
                "sim": round(float(S[a][b]), 3),
                "clash": [],
                "kind": "prose",
                "subject": sorted(shared)[:4],
                "a": {"file": pa.file, "line": pa.start_line, "values": [],
                      "text": " ".join(pa.text.split())[:240]},
                "b": {"file": pb.file, "line": pb.start_line, "values": [],
                      "text": " ".join(pb.text.split())[:240]},
            })
    candidates.sort(key=lambda c: -c["sim"])
    return candidates[:max_candidates]
