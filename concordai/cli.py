"""Concord CLI — the substrate every driver (human, skill, MCP) sits on.

    concord lint  [--rules rules.yaml] [--scope public] [path]
    concord find  "<query>" [--scope public] [path]
    concord index [path]
    concord read  "<question>" [path]      # retrieve; synthesis is the driver's job
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .find import find
from .lint import lint_repo
from .rules import load_ruleset


def _default_rules(path: Path) -> Path:
    for cand in (path / "rules.yaml", path / "rules.local.yaml"):
        if cand.exists():
            return cand
    return Path(__file__).parent / "rules.example.yaml"


def cmd_init(args) -> int:
    from .bootstrap import ensure_gitignore, scaffold_rules
    root = Path(args.path)
    rules_path, created = scaffold_rules(root)
    added = ensure_gitignore(root)
    print(f"rules.yaml: {'created from example' if created else 'already present'}  ({rules_path})")
    if added:
        print("gitignore: added " + ", ".join(added))
    else:
        print("gitignore: already covers Concord's private files")
    print("\nrules.yaml holds your real terms and is now gitignored — edit it, then run `concord lint`.")
    return 0


def cmd_lint(args) -> int:
    root = Path(args.path)
    rules = load_ruleset(args.rules or _default_rules(root))
    findings = lint_repo(root, rules, scope=tuple(s for s in args.scope.split(",") if s))
    if getattr(args, "since", None):  # PR-diff: only what changed since <ref>
        from . import gitdiff
        changed = set(gitdiff.changed_files(root, since=args.since)[0])
        findings = [f for f in findings if f.file in changed]
    errors = [f for f in findings if f.severity == "error"]
    for f in findings:
        print(f)
    n_sem = len(rules.semantic_terms())
    print(
        f"\n{len(findings)} finding(s): {len(errors)} error, {len(findings) - len(errors)} warn"
        + (f"  ({n_sem} semantic term(s) need `concord find` — not covered by lint)" if n_sem else ""),
        file=sys.stderr,
    )
    return 1 if errors else 0


def cmd_find(args) -> int:
    root = Path(args.path)
    rules = load_ruleset(_default_rules(root))
    channels = ("exact",) if args.exact_only else ("exact", "semantic")
    scope = tuple(s for s in args.scope.split(",") if s) if args.scope else None
    hits = find(args.query, root, rules, channels=channels, scope=scope)
    for h in hits:
        tag = "=" if h.match_type == "exact" else f"{h.score:.2f}"
        print(f"[{tag:>4}] {h.file}:{h.line}  {h.text}")
    print(f"\n{len(hits)} hit(s)", file=sys.stderr)
    return 0


def cmd_index(args) -> int:
    from .index import Index
    from .embed import get_embedder
    from . import gitdiff
    root = Path(args.path)
    rules = load_ruleset(_default_rules(root))
    idx = Index.build(root, rules, embedder=get_embedder(args.model))
    idx.save(root, meta={"model": args.model, "commit": gitdiff.head(root)})
    from collections import Counter
    by_ext = Counter(Path(p.file).suffix.lower() or "(none)" for p in idx.passages)
    breakdown = ", ".join(f"{n}×{e}" for e, n in by_ext.most_common())
    print(f"Indexed {len(idx.passages)} passage(s) [{breakdown}] -> {root}/.concord/", file=sys.stderr)
    return 0


def cmd_types(args) -> int:
    from . import extract
    exts = extract.supported_extensions()
    print("Concord semantically indexes these file types (extraction-cleaned content):")
    print("  " + "  ".join(exts))
    print("\nEach type has an extractor that keeps the meaningful units (prose, comments,")
    print("strings, config values, gating constants) and drops the syntax. The leak-lint")
    print("scans all of these too, but raw. Add a language by registering an extractor")
    print("in concordai/extract.py.")
    return 0


def cmd_update(args) -> int:
    from .index import Index
    from . import gitdiff
    root = Path(args.path)
    idx = Index.load(root)
    model = idx.meta.get("model")

    use_git = (not args.no_git) and gitdiff.head(root) is not None and bool(idx.meta.get("commit"))
    if use_git:
        since = None if args.last_commit else idx.meta.get("commit")
        changed, deleted = gitdiff.changed_files(root, since=since, last_commit=args.last_commit)
        new_commit = gitdiff.head(root)
        source = "git"
    else:
        from . import manifest
        changed, deleted = manifest.diff(idx.manifest, manifest.scan(root))
        new_commit = idx.meta.get("commit")
        source = "manifest"

    from .embed import get_embedder
    idx.update(root, changed, deleted, get_embedder(model))
    idx.save(root, meta={"commit": new_commit})
    print(
        f"updated [{source}]: {len(changed)} changed, {len(deleted)} deleted "
        f"-> {len(idx.passages)} passages",
        file=sys.stderr,
    )
    return 0


def cmd_topics(args) -> int:
    from collections import defaultdict
    from .index import Index
    from . import cluster as _cluster
    root = Path(args.path)
    idx = Index.load(root)
    if idx.matrix is None:
        print("No semantic index — run `concord index` first.", file=sys.stderr)
        return 1
    cl = _cluster.cluster(idx.matrix, [p.text for p in idx.passages], k_leaves=args.k, n_super=args.super)

    if args.route:
        from .embed import get_embedder
        qv = get_embedder(idx.meta.get("model")).embed([args.route])[0]
        leaf = _cluster.route(qv, cl)[0]
        sup = int(cl.super_of_leaf[leaf])
        members = [i for i in range(len(idx.passages)) if cl.leaf_of[i] == leaf]
        print(f"'{args.route}'\n  -> topic: {cl.leaf_labels[leaf]}   (theme: {cl.super_labels[sup]})")
        print(f"  the whole topic neighbourhood = {len(members)} passages (coverage, not a top-k cutoff):")
        for i in members[: args.show]:
            p = idx.passages[i]
            print(f"    {p.file}:{p.start_line}  {' '.join(p.text.split())[:110]}")
        return 0

    if args.samples:
        # Emit representative passages per topic so an LLM/human driver can NAME them
        # (the "spend a few tokens to describe the splits" path — naming stays with the driver).
        import numpy as np
        cent = cl.leaf_centroids
        for leaf in sorted(range(cl.k), key=lambda l: -cl.sizes[l]):
            members = [i for i in range(len(idx.passages)) if cl.leaf_of[i] == leaf]
            c = cent[leaf] / (np.linalg.norm(cent[leaf]) + 1e-9)
            members.sort(key=lambda i: -float(idx.matrix[i] @ c))
            print(f"\n## topic {leaf}  ({cl.sizes[leaf]} passages)  auto-label: {cl.leaf_labels[leaf]}")
            print("   (name this topic from its representative passages:)")
            for i in members[:3]:
                p = idx.passages[i]
                print(f"   - {p.file}:{p.start_line}  {' '.join(p.text.split())[:130]}")
        return 0

    by_super = defaultdict(list)
    for leaf in range(cl.k):
        by_super[int(cl.super_of_leaf[leaf])].append(leaf)
    for sup in sorted(by_super, key=lambda s: -sum(cl.sizes[l] for l in by_super[s])):
        tot = sum(cl.sizes[l] for l in by_super[sup])
        print(f"\n# {cl.super_labels[sup]}  ({tot} passages)")
        for leaf in sorted(by_super[sup], key=lambda l: -cl.sizes[l]):
            print(f"    - {cl.leaf_labels[leaf]}  ({cl.sizes[leaf]})")
    return 0


def cmd_radar(args) -> int:
    from . import radar
    from .index import Index
    idx = Index.load(args.path)
    if idx.matrix is None:
        print("No semantic index — run `concord index` first.", file=sys.stderr)
        return 1
    conflicts = radar.find_conflicts(idx.passages, idx.matrix)["conflicts"][: args.max]
    if getattr(args, "since", None):  # PR-diff: only conflicts touching changed files
        from . import gitdiff
        changed = set(gitdiff.changed_files(Path(args.path), since=args.since)[0])
        conflicts = [c for c in conflicts if c["a"]["file"] in changed or c["b"]["file"] in changed]
    if args.verify:
        from . import llmlabel, verify as V
        st = llmlabel.status()
        if st["available"]:
            print(f"# verifying with YOUR {st['provider']} API key ({st['model']}) — you pay for usage", file=sys.stderr)
        verdicts = V.verify(conflicts)
        if verdicts is None:
            print("# LLM verify unavailable (set ANTHROPIC_API_KEY/OPENAI_API_KEY) — showing raw candidates\n", file=sys.stderr)
        else:
            real = [(c, d) for c, d in zip(conflicts, verdicts) if d.get("real")]
            print(f"# {len(real)} CONFIRMED contradiction(s) of {len(verdicts)} candidate(s) — LLM-verified\n")
            for c, d in real:
                print(f"~ {' vs '.join(c['clash'])}  ->  canonical: {d.get('canonical')}   ({d.get('why', '')})")
                print(f"    {c['a']['file']}:{c['a']['line']}")
                print(f"    {c['b']['file']}:{c['b']['line']}")
            return 0
    print(f"# {len(conflicts)} value-conflict candidate(s) — same topic + same kind of number, different values")
    print("# confirm each (add --verify to let an LLM judge + name the canonical value).\n")
    for c in conflicts:
        print(f"~ {' vs '.join(c['clash'])}   (sim {c['sim']}; subject: {', '.join(c['subject'][:3])})")
        print(f"    {c['a']['file']}:{c['a']['line']}")
        print(f"    {c['b']['file']}:{c['b']['line']}")
    return 0


def cmd_resolve(args) -> int:
    from . import radar, verify as V
    from .index import Index
    root = str(Path(args.path).resolve())
    idx = Index.load(root)
    if idx.matrix is None:
        print("No semantic index — run `concord index` first.", file=sys.stderr)
        return 1
    conflicts = radar.find_conflicts(idx.passages, idx.matrix)["conflicts"][: args.max]
    from . import llmlabel
    st = llmlabel.status()
    if st["available"]:
        print(f"Verifying with YOUR {st['provider']} API key ({st['model']}) — you pay for usage.\n", file=sys.stderr)
    verdicts = V.verify(conflicts)
    if verdicts is None:
        print("resolve needs an LLM (set ANTHROPIC_API_KEY or OPENAI_API_KEY).", file=sys.stderr)
        return 1
    real = [(c, d) for c, d in zip(conflicts, verdicts)
            if d.get("real") and d.get("canonical") and d.get("change") in ("a", "b")]
    if not real:
        print("No confirmed, resolvable contradictions. ✅")
        return 0
    print(f"{len(real)} confirmed contradiction(s). {'(auto-apply)' if args.apply else '(review each)'}\n")
    applied = 0
    for c, d in real:
        side, canon = d["change"], d["canonical"]
        print(f"CONTRADICTION  {' vs '.join(c['clash'])}   — {d.get('why', '')}")
        print(f"  A  {c['a']['file']}:{c['a']['line']}   {c['a']['values']}")
        print(f"  B  {c['b']['file']}:{c['b']['line']}   {c['b']['values']}")
        prev = V.apply_fix(root, side, canon, c, dry_run=True)
        if not prev:
            print("  (could not locate the value to replace — skipping)\n")
            continue
        f, ln, before, after = prev
        print(f"  canonical = {canon}; change side {side.upper()}:")
        print(f"    - {f}:{ln}  {before}")
        print(f"    + {f}:{ln}  {after}")
        choice = "y" if args.apply else input("  apply? [y/N/q] ").strip().lower()
        if choice == "q":
            break
        if choice == "y":
            V.apply_fix(root, side, canon, c, dry_run=False)
            applied += 1
            print("  ✓ applied\n")
        else:
            print("  skipped\n")
    print(f"Resolved {applied} contradiction(s).")
    return 0


def cmd_drift(args) -> int:
    from . import server
    commits = server.drift(str(Path(args.path).resolve()), args.term)
    if not commits:
        print(f"No commits touch '{args.term}'.", file=sys.stderr)
        return 0
    print(f"# how '{args.term}' evolved ({len(commits)} commit(s)):")
    for c in commits:
        print(f"  {c['date']}  {c['hash']}  {c['subject']}")
    return 0


def cmd_badge(args) -> int:
    root = Path(args.path)
    findings = lint_repo(root, load_ruleset(_default_rules(root)), scope=("public",))
    errs = sum(1 for f in findings if f.severity == "error")
    label = "0%20leaks" if errs == 0 else f"{errs}%20leak{'s' if errs != 1 else ''}"
    color = "brightgreen" if errs == 0 else "critical"
    print(f"![Concord](https://img.shields.io/badge/concord-{label}-{color})")
    return 1 if errs else 0


def cmd_report(args) -> int:
    import datetime
    from . import radar, report
    from .index import Index
    root = Path(args.path)
    findings = lint_repo(root, load_ruleset(_default_rules(root)), scope=("public",))
    conflicts, verdicts = [], None
    try:
        idx = Index.load(root)
        if idx.matrix is not None:
            conflicts = radar.find_conflicts(idx.passages, idx.matrix)["conflicts"][: args.max]
            if args.verify:
                from . import verify as V
                verdicts = V.verify(conflicts)
    except Exception:  # noqa: BLE001 — report still useful with just the lint
        pass
    repo = root.resolve().name or str(root)
    html = report.build(repo, datetime.date.today().isoformat(), findings, conflicts, verdicts)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"Wrote {args.out}  ({len(findings)} leak(s), {len(conflicts)} contradiction candidate(s))", file=sys.stderr)
    return 0


def cmd_activity(args) -> int:
    from . import activity
    files = activity.file_activity(args.path, since=args.since)
    if not files:
        print("No git activity in window (is this a git repo?).", file=sys.stderr)
        return 0
    print(f"# dev activity since '{args.since}' — {len(files)} files touched\n")
    print("## hotspots (most churn = where effort goes):")
    for f in files[: args.max]:
        print(f"  {f['churn']:6d} lines · {f['commits']}c · {f['authors']}a   {f['file']}")
    coll = sorted([f for f in files if f["authors"] >= 2], key=lambda x: (-x["authors"], -x["churn"]))
    print(f"\n## collision risk — {len(coll)} file(s) touched by 2+ authors:")
    if not coll:
        print("  none (single author, or no concurrent edits in window).")
    for f in coll[: args.max]:
        print(f"  {f['authors']} authors ({', '.join(f['author_names'][:4])})   {f['file']}")
    return 0


def cmd_ui(args) -> int:
    import webbrowser
    from . import server
    root = str(Path(args.path).resolve())
    try:
        srv = server.make_server(root, args.port)  # loads index + embedder (slow first call)
    except Exception as e:  # noqa: BLE001
        print(f"Could not start explorer: {e}", file=sys.stderr)
        return 1
    from . import llmlabel
    st = llmlabel.status()
    ai = f"AI: {st['provider']} {st['model']} (opt-in; your key, you pay)" if st["available"] else "AI: off (no key — search/topics/lint/radar are free)"
    url = f"http://127.0.0.1:{args.port}"
    print(f"Concord explorer → {url}   [{ai}]   (Ctrl+C to stop)", file=sys.stderr)
    webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()
    return 0


def cmd_read(args) -> int:
    root = Path(args.path)
    rules = load_ruleset(_default_rules(root))
    queries = [args.query] + list(args.also or [])
    hits = find(queries, root, rules, channels=("semantic", "exact"), top=args.max)

    facet = None
    if args.facets and len(hits) >= 4:
        try:
            from . import cluster as _cluster
            from .embed import get_embedder
            from .index import Index
            emb = get_embedder(Index.load(root).meta.get("model"))
            V = emb.embed([h.text for h in hits], kind="passage")
            facet = _cluster.facet_labels([h.text for h in hits], V)  # k auto-selected
        except Exception:  # noqa: BLE001 — facets are a nicety; never block read
            facet = None

    n_facets = len(set(facet)) if facet else 0
    print(f"# {len(hits)} passages ranked across {len(queries)} phrasing(s).")
    if facet:
        print(f"# {n_facets} facet(s) detected. Walk top-down; keep going while NEW facets still appear,")
        print("# stop when both relevance AND new facets dry up. (1-facet query stops fast; multi-facet reads broader.)\n")
    else:
        print("# Read top-down and synthesise; STOP after ~4-5 consecutive irrelevant. No fixed cutoff.\n")
    for i, h in enumerate(hits):
        tag = f"  [facet: {facet[i]}]" if facet else ""
        print(f"## [{h.score:.3f}] {h.file}:{h.line} ({h.match_type}){tag}\n{h.text}\n")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="concord", description="Keep a sprawling repo telling one story.")
    p.add_argument("--version", action="version", version=f"concord {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="scaffold rules.yaml and gitignore the private ruleset + index")
    sp.add_argument("path", nargs="?", default=".")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("lint", help="fail if a banned term reaches a public file")
    sp.add_argument("path", nargs="?", default=".")
    sp.add_argument("--rules", default=None)
    sp.add_argument("--scope", default="public", help="comma-separated visibility categories")
    sp.add_argument("--since", default=None, help="PR-diff: only flag files changed since this git ref (e.g. origin/main)")
    sp.set_defaults(func=cmd_lint)

    sp = sub.add_parser("find", help="exact + semantic hits, ranked")
    sp.add_argument("query")
    sp.add_argument("-C", "--path", default=".", help="repo root (default: cwd)")
    sp.add_argument("--scope", default=None, help="comma-separated visibility categories")
    sp.add_argument("--exact-only", action="store_true")
    sp.set_defaults(func=cmd_find)

    sp = sub.add_parser("index", help="build the semantic index")
    sp.add_argument("path", nargs="?", default=".")
    sp.add_argument("--model", default=None, help="sentiment.ai model (e.g. e5-small); default = sentiment.ai's default")
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("types", help="list the file types Concord semantically indexes")
    sp.set_defaults(func=cmd_types)

    sp = sub.add_parser("update", help="re-embed only what git says changed since the last index")
    sp.add_argument("path", nargs="?", default=".")
    sp.add_argument("--last-commit", action="store_true", help="only diff HEAD~1..HEAD (post-commit hook)")
    sp.add_argument("--no-git", action="store_true", help="ignore git; detect changes via content-hash manifest")
    sp.set_defaults(func=cmd_update)

    sp = sub.add_parser("topics", help="annotated topic map over the index (+ --route a query to its cluster)")
    sp.add_argument("path", nargs="?", default=".")
    sp.add_argument("-k", type=int, default=40, help="leaf clusters")
    sp.add_argument("--super", type=int, default=8, help="super-clusters (themes)")
    sp.add_argument("--route", default=None, help="(experimental — underperforms flat retrieval) which cluster a query lands in")
    sp.add_argument("--show", type=int, default=12, help="passages to show when routing")
    sp.add_argument("--samples", action="store_true", help="emit representative passages per topic for a driver to name them")
    sp.set_defaults(func=cmd_topics)

    sp = sub.add_parser("ui", help="launch the live explorer (search + topic map + radar) in your browser")
    sp.add_argument("path", nargs="?", default=".")
    sp.add_argument("--port", type=int, default=8765)
    sp.set_defaults(func=cmd_ui)

    sp = sub.add_parser("radar", help="contradiction radar: same-topic passages stating different values")
    sp.add_argument("path", nargs="?", default=".")
    sp.add_argument("--max", type=int, default=40)
    sp.add_argument("--verify", action="store_true", help="let an LLM confirm real contradictions + name the canonical value")
    sp.add_argument("--since", default=None, help="PR-diff: only contradictions touching files changed since this git ref")
    sp.set_defaults(func=cmd_radar)

    sp = sub.add_parser("activity", help="where dev effort goes + collision risk (files 2+ authors touch), from git")
    sp.add_argument("path", nargs="?", default=".")
    sp.add_argument("--since", default="3 months ago", help="git window (e.g. '30 days ago', '2026-01-01')")
    sp.add_argument("--max", type=int, default=12)
    sp.set_defaults(func=cmd_activity)

    sp = sub.add_parser("report", help="write a shareable consistency report (HTML): lint + contradiction radar")
    sp.add_argument("path", nargs="?", default=".")
    sp.add_argument("--out", default="concord-report.html")
    sp.add_argument("--max", type=int, default=40)
    sp.add_argument("--verify", action="store_true", help="LLM-verify the contradictions in the report")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("resolve", help="LLM-verify contradictions then pick the canonical value and auto-apply the fix")
    sp.add_argument("path", nargs="?", default=".")
    sp.add_argument("--max", type=int, default=40)
    sp.add_argument("--apply", action="store_true", help="auto-apply the LLM's canonical fix without prompting (writes files!)")
    sp.set_defaults(func=cmd_resolve)

    sp = sub.add_parser("drift", help="how a value/term evolved across git history (pickaxe)")
    sp.add_argument("term")
    sp.add_argument("-C", "--path", default=".", help="repo root (default: cwd)")
    sp.set_defaults(func=cmd_drift)

    sp = sub.add_parser("badge", help="print a shields.io leak badge (markdown) for your README")
    sp.add_argument("path", nargs="?", default=".")
    sp.set_defaults(func=cmd_badge)

    sp = sub.add_parser("read", help="retrieve a ranked window for a question (drive a patience-walk over it)")
    sp.add_argument("query")
    sp.add_argument("-C", "--path", default=".", help="repo root (default: cwd)")
    sp.add_argument("--also", action="append", help="extra phrasing(s) of the question (multi-query; repeatable)")
    sp.add_argument("--max", type=int, default=40, help="max passages in the ranked window (a ceiling, not a target)")
    sp.add_argument("--facets", action="store_true", help="tag each passage with its result-facet for a facet-aware (dynamic-patience) walk")
    sp.set_defaults(func=cmd_read)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
