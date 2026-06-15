"""MCP stdio server: protocol handshake + deterministic tool calls (no embedder)."""
from __future__ import annotations

from concordai import mcp_server

_REPO = "examples/demo"


def _req(method, params=None, mid=1):
    return {"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}


def test_initialize_and_tools_list():
    r = mcp_server.handle(".", _req("initialize"))
    assert r["result"]["serverInfo"]["name"] == "concord"
    assert r["result"]["protocolVersion"]
    names = {t["name"] for t in mcp_server.handle(".", _req("tools/list", mid=2))["result"]["tools"]}
    assert {"concord_find", "concord_read", "concord_radar",
            "concord_graph", "concord_coverage", "concord_lint"} <= names


def test_notifications_get_no_response():
    assert mcp_server.handle(".", {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_unknown_method_is_jsonrpc_error():
    r = mcp_server.handle(".", _req("bogus", mid=9))
    assert r["error"]["code"] == -32601


def test_graph_tool_on_demo():
    r = mcp_server.handle(_REPO, _req("tools/call", {"name": "concord_graph", "arguments": {}}, mid=3))
    assert r["result"]["isError"] is False
    assert "doc-links" in r["result"]["content"][0]["text"]


def test_coverage_tool_on_demo():
    r = mcp_server.handle(_REPO, _req("tools/call", {"name": "concord_coverage", "arguments": {}}, mid=4))
    txt = r["result"]["content"][0]["text"]
    assert r["result"]["isError"] is False and "undocumented" in txt


def test_unknown_tool_is_flagged_error():
    r = mcp_server.handle(_REPO, _req("tools/call", {"name": "concord_nope", "arguments": {}}, mid=5))
    assert r["result"]["isError"] is True
