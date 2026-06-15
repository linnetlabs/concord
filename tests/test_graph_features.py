"""Library-graph extras: canonical suggestion, coverage, and mermaid/dot export.

All pure-function tests -- no built index needed; we feed synthetic graph/conflict dicts.
"""
from __future__ import annotations

from concordai import cli, radar
# NB: `concordai.graph` the NAME is the re-exported function; reach module members directly.
from concordai.graph import coverage, to_mermaid, to_dot


# --- a synthetic library graph (the shape graph.graph() returns) -----------------
def _g():
    return {
        "nodes": [
            {"file": "src/a.py", "passages": 3, "churn": 50, "authors": 1,
             "last": "2026-01-01", "freshness": "fresh", "lagging": False, "visibility": "internal"},
            {"file": "README.md", "passages": 2, "churn": 10, "authors": 1,
             "last": "2026-01-01", "freshness": "fresh", "lagging": False, "visibility": "public"},
            {"file": "docs/old.md", "passages": 1, "churn": 1, "authors": 1,
             "last": "2024-01-01", "freshness": "stale", "lagging": True, "visibility": "public"},
        ],
        "edges": [{"source": "README.md", "target": "docs/old.md", "text": "see", "kind": "link"}],
        "stats": {"files": 3, "links": 1, "stale": 1, "lagging": 1},
    }


# --- canonical suggestion --------------------------------------------------------
def test_pick_canonical_lagging_loses():
    fm = {"old.md": {"freshness": "stale", "lagging": True, "last": "2025-01-01"},
          "new.md": {"freshness": "fresh", "lagging": False, "last": "2026-06-01"}}
    side, why = radar.pick_canonical({"a": {"file": "old.md"}, "b": {"file": "new.md"}}, fm)
    assert side == "b" and "lags" in why


def test_pick_canonical_fresher_wins_then_date_then_none():
    base = {"a": {"file": "a.md"}, "b": {"file": "b.md"}}
    fresh = {"a.md": {"freshness": "fresh", "lagging": False, "last": "2026-01-01"},
             "b.md": {"freshness": "stale", "lagging": False, "last": "2026-01-01"}}
    assert radar.pick_canonical(base, fresh)[0] == "a"
    bydate = {"a.md": {"freshness": "fresh", "lagging": False, "last": "2026-06-01"},
              "b.md": {"freshness": "fresh", "lagging": False, "last": "2026-01-01"}}
    assert radar.pick_canonical(base, bydate)[0] == "a"
    tie = {"a.md": {"freshness": "fresh", "lagging": False, "last": "2026-06-01"},
           "b.md": {"freshness": "fresh", "lagging": False, "last": "2026-06-01"}}
    assert radar.pick_canonical(base, tie)[0] is None


def test_annotate_canonical_tags_conflicts():
    fm = {"docs/old.md": {"freshness": "stale", "lagging": True, "last": "2024-01-01"},
          "README.md": {"freshness": "fresh", "lagging": False, "last": "2026-06-01"}}
    conflicts = [{"a": {"file": "docs/old.md", "values": ["$49"]},
                  "b": {"file": "README.md", "values": ["$39"]}}]
    radar.annotate_canonical(conflicts, fm)
    c = conflicts[0]
    assert c["canonical"] == "b"
    assert c["canonical_file"] == "README.md"
    assert c["canonical_value"] == "$39"


# --- coverage --------------------------------------------------------------------
def test_coverage_flags_undocumented_code_and_lagging_docs():
    cov = coverage(".", _g())
    undoc = {n["file"] for n in cov["undocumented"]}
    assert "src/a.py" in undoc                      # code file, no inbound doc-link
    assert "README.md" not in undoc                 # a doc, never "undocumented code"
    assert "docs/old.md" not in undoc               # it is a link target (indeg 1)
    assert {n["file"] for n in cov["lagging"]} == {"docs/old.md"}
    assert cov["stats"]["undocumented"] == 1 and cov["stats"]["lagging"] == 1


# --- exports ---------------------------------------------------------------------
def test_to_mermaid_styles_and_lagging_ring():
    m = to_mermaid(_g())
    assert m.startswith("graph LR")
    assert "-->" in m
    assert "stroke-width:4px" in m                  # the lagging node gets a ring
    assert "README.md" in m and "docs/old.md" in m  # 2-segment labels disambiguate


def test_to_dot_is_valid_ish():
    d = to_dot(_g())
    assert "digraph concord {" in d and d.rstrip().endswith("}")
    assert '"README.md" -> "docs/old.md";' in d
    assert "penwidth=3" in d                         # lagging node


# --- CI exit codes ---------------------------------------------------------------
def test_radar_exit_codes():
    mk = lambda mode: type("A", (), {"fail_on": mode})()
    assert cli._radar_exit(mk("none"), [1], []) == 0
    assert cli._radar_exit(mk("conflict"), [1], []) == 2
    assert cli._radar_exit(mk("conflict"), [], []) == 0
    assert cli._radar_exit(mk("verified"), [1], []) == 0
    assert cli._radar_exit(mk("verified"), [1], [1]) == 2
