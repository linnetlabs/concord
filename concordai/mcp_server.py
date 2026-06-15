"""Minimal MCP (Model Context Protocol) stdio server -- concord as agent tools.

`concord serve --mcp [path]` speaks newline-delimited JSON-RPC 2.0 on stdin/stdout, so an
MCP client (Claude Desktop, an agent harness) can call find / read / radar / graph /
coverage / lint over a repo. Dependency-free: the protocol is a thin JSON-RPC loop, and
each tool delegates to the same library functions the CLI uses. This ships the "Agent ->
MCP server" driver the README advertises.

The graph / coverage / lint tools are deterministic and need no model. find / read / radar
load the local embedder on first call (the semantic index), same as `concord ui`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import __version__

PROTOCOL_VERSION = "2024-11-05"


def _tools() -> list:
    obj = lambda props, req=(): {"type": "object", "properties": props, "required": list(req)}
    return [
        {"name": "concord_find", "description": "Exact + semantic search over the repo, cited to file:line. Set all=true for a recall-complete sweep (find ALL X).",
         "inputSchema": obj({"query": {"type": "string"}, "all": {"type": "boolean"}, "max": {"type": "integer"}}, ["query"])},
        {"name": "concord_read", "description": "Retrieve the ranked passages that answer a question (token-efficient; cited to file:line).",
         "inputSchema": obj({"query": {"type": "string"}, "max": {"type": "integer"}}, ["query"])},
        {"name": "concord_radar", "description": "Value contradictions (same topic, conflicting value) with a deterministic canonical suggestion.",
         "inputSchema": obj({"max": {"type": "integer"}})},
        {"name": "concord_graph", "description": "The document link graph: files, doc-links, git freshness, lagging docs.",
         "inputSchema": obj({})},
        {"name": "concord_coverage", "description": "Undocumented code (no inbound doc-link) and docs lagging the code they reference.",
         "inputSchema": obj({})},
        {"name": "concord_lint", "description": "Flag banned/internal terms that reach public files (deterministic leak guard).",
         "inputSchema": obj({"scope": {"type": "string"}})},
    ]


def _rules(root: Path):
    from .rules import load_ruleset
    for c in (root / "rules.yaml", root / "rules.local.yaml"):
        if c.exists():
            return load_ruleset(c)
    return load_ruleset(Path(__file__).parent / "rules.example.yaml")


def call_tool(root: str, name: str, args: dict):
    """Dispatch one tool call -> (text, is_error). Pure-ish; used directly by tests."""
    root_p = Path(root)
    args = args or {}
    try:
        if name == "concord_lint":
            from .lint import lint_repo
            scope = tuple(s for s in (args.get("scope") or "public").split(",") if s)
            findings = lint_repo(root_p, _rules(root_p), scope=scope)
            return ("\n".join(str(f) for f in findings) or "no leaks found", False)

        if name == "concord_graph":
            from .graph import graph as build
            g = build(root, write=False)
            s = g["stats"]
            indeg = {}
            for e in g["edges"]:
                indeg[e["target"]] = indeg.get(e["target"], 0) + 1
            top = sorted(indeg.items(), key=lambda x: -x[1])[:10]
            lines = [f"{s['files']} files, {s['links']} doc-links, {s['stale']} stale, {s['lagging']} lagging",
                     "most-referenced:"] + [f"  {n} <- {f}" for f, n in top]
            return ("\n".join(lines), False)

        if name == "concord_coverage":
            from .graph import coverage
            cov = coverage(root)
            s = cov["stats"]
            lines = [f"{s['files']} files, {s['undocumented']} undocumented code, {s['lagging']} lagging docs",
                     "undocumented (high churn first):"] + [f"  {n['churn']} churn  {n['file']}" for n in cov["undocumented"][:15]]
            return ("\n".join(lines), False)

        if name == "concord_find":
            from .find import find, find_all
            rules = _rules(root_p)
            q = args["query"]
            hits = find_all(q, root, rules) if args.get("all") else find(q, root, rules, top=int(args.get("max", 20)))
            out = [f"[{h.score:.3f}] {h.file}:{h.line}  {h.text}" + (f"  (facet: {h.facet})" if h.facet else "") for h in hits]
            return ("\n".join(out) or "(no hits)", False)

        if name == "concord_read":
            from .find import find
            rules = _rules(root_p)
            hits = find(args["query"], root, rules, channels=("semantic", "exact"), top=int(args.get("max", 12)))
            out = [f"## [{h.score:.3f}] {h.file}:{h.line}\n{h.text}" for h in hits]
            return ("\n\n".join(out) or "(no passages)", False)

        if name == "concord_radar":
            from .graph import consistency
            conflicts = consistency(root)["conflicts"][: int(args.get("max", 40))]
            if not conflicts:
                return ("no contradictions found", False)
            out = []
            for c in conflicts:
                line = f"~ {' vs '.join(c.get('clash') or [])}  ({c['a']['file']}:{c['a']['line']} | {c['b']['file']}:{c['b']['line']})"
                if c.get("canonical"):
                    line += f"\n  -> canonical: {c['canonical_file']} ({c['canonical_reason']})"
                out.append(line)
            return ("\n".join(out), False)

        return (f"unknown tool: {name}", True)
    except Exception as e:  # noqa: BLE001 -- tool errors are reported, never crash the server
        return (f"error: {e}", True)


def handle(root: str, msg: dict):
    """One JSON-RPC message -> a response dict, or None for notifications."""
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if method == "initialize":
        return _ok(mid, {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}},
                         "serverInfo": {"name": "concord", "version": __version__}})
    if method in ("notifications/initialized", "notifications/cancelled"):
        return None
    if method == "ping":
        return _ok(mid, {})
    if method == "tools/list":
        return _ok(mid, {"tools": _tools()})
    if method == "tools/call":
        text, is_err = call_tool(root, params.get("name"), params.get("arguments"))
        return _ok(mid, {"content": [{"type": "text", "text": text}], "isError": is_err})
    if mid is not None:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    return None


def _ok(mid, result):
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def serve(root: str = ".") -> int:
    """Run the stdio loop until EOF. Newline-delimited JSON-RPC 2.0."""
    root = str(Path(root).resolve())
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(root, msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    return 0
