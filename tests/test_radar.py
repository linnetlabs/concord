"""Contradiction radar: same-topic + same-value-type + different number, prose only."""
from __future__ import annotations

import numpy as np

from concordai import radar
from concordai.chunk import Passage


def _p(file, text):
    return Passage(file, 1, 1, text, "unknown")


def _sim2():  # two near-identical-direction rows (cosine > 0.99)
    M = np.array([[1.0, 0.0], [0.97, 0.05]], dtype="float32")
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def test_typed_values_normalises():
    v = radar._typed_values("priced at $49.00, n >= 8, within 30 days, up 30%")
    assert "$49" in v["price"]          # .00 stripped
    assert "30days" in v["duration"]
    assert "30%" in v["percent"]
    assert any("8" in x for x in v["threshold"])


def test_conflicting_requires_same_type():
    assert radar._conflicting({"price": frozenset(["$49"])}, {"price": frozenset(["$39"])})
    assert not radar._conflicting({"price": frozenset(["$49"])}, {"duration": frozenset(["30days"])})
    assert not radar._conflicting({"price": frozenset(["$49"])}, {"price": frozenset(["$49"])})


def test_flags_same_subject_conflicting_price():
    a = _p("a.md", "Bluebird Cloud is priced at $49 per seat, billed annually to customers")
    b = _p("b.md", "Each Bluebird Cloud seat is $39 monthly on the standard tier plan")
    r = radar.find_conflicts([a, b], _sim2(), sim_threshold=0.8)
    assert len(r["conflicts"]) == 1
    c = r["conflicts"][0]
    assert "$49" in c["clash"] and "$39" in c["clash"]
    assert {c["a"]["file"], c["b"]["file"]} == {"a.md", "b.md"}


def test_near_identical_copies_not_flagged():
    a = _p("a.md", "Bluebird Cloud costs $49 per seat per month always")
    b = _p("b.md", "Bluebird Cloud costs $39 per seat per month always")  # only the number differs
    assert radar.find_conflicts([a, b], _sim2(), sim_threshold=0.8)["conflicts"] == []


def test_skips_hidden_tool_dirs():
    a = _p(".claude/x.md", "Bluebird Cloud is priced at $49 per seat billed annually now")
    b = _p("b.md", "Each Bluebird Cloud seat is $39 monthly on the standard tier plan")
    assert radar.find_conflicts([a, b], _sim2(), sim_threshold=0.8)["conflicts"] == []


def test_typed_values_catches_code_config_forms():
    v = radar._typed_values("MIN_RESPONDENTS = 8  # anonymity floor")
    assert any("8" in x for x in v["config"])
    v2 = radar._typed_values('"min_respondents": 5')
    assert any("5" in x for x in v2["config"])


def test_flags_gate_constant_conflict_across_code_and_config():
    a = _p("privacy.py", "MIN_RESPONDENTS = 8  # anonymity floor before aggregation")
    b = _p("settings.json", '"min_respondents": 5')
    r = radar.find_conflicts([a, b], _sim2(), sim_threshold=0.8)
    assert len(r["conflicts"]) == 1
    assert {c for c in r["conflicts"][0]["clash"]}  # the disagreeing values surfaced
    assert {r["conflicts"][0]["a"]["file"], r["conflicts"][0]["b"]["file"]} == {"privacy.py", "settings.json"}
