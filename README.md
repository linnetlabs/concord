# Concord

![Concord leak guard](https://img.shields.io/badge/concord-0%20leaks-brightgreen)
![tests](https://img.shields.io/badge/tests-49%20passing-brightgreen)
![PyPI](https://img.shields.io/pypi/v/concord-ai?color=blue)
![license](https://img.shields.io/badge/license-MIT-blue)

> ## Find where your repo contradicts itself, across code *and* docs
>
> `$49` on the pricing page, `$39` in an FAQ. `MIN_RESPONDENTS = 8` in a Python module,
> `"min_n": 5` in a config. Concord catches the contradictions `grep` and concept-graphs
> don't surface (the same fact stated two different ways, wherever it lives) and answers
> questions about your repo from the **exact passages, cited to `file:line`**, instead
> of reading the whole thing into a model.

**Status:** `init` and `lint` work today with no ML. The semantic features (`index`,
`find`, `read`, `radar`, `ui`) need the `[embeddings]` extra and are usable now; treat
them as beta. Embeddings run locally via [sentiment.ai](https://github.com/BenWiseman/sentiment.ai).

**Keep a sprawling repo telling one story.**

Concord indexes the meaningful content of a repository (docs, specs, READMEs, and the
strings, comments and named constants in your code and config) and lets you:

- **Radar:** *"where does the same fact disagree with itself?"* Same topic, same kind
  of value (a price, a threshold, a gating constant), different number, across prose
  and source (`$49` vs `$39`, `MIN_N = 8` vs `"min_n": 5`). With `--prose` it also
  surfaces **non-numeric** contradictions (`SOC 2 certified` vs `SOC 2 in progress`),
  LLM-judged over the same retrieval, opt-in. Each conflict gets a deterministic
  **canonical suggestion** -- the side whose file is fresher (or not lagging) in git is
  the likely source of truth, no model needed.
- **Lint:** *"does any internal codename or retired term reach a file that ships
  publicly?"* Deterministic exact-match over a known term list, scanning raw text.
  Runs in CI or a pre-commit hook.
- **Find / Read:** *"where else do we say something like this?"* and *"answer X from
  the repo."* Exact and semantic matches, cited to `file:line`, pulling only the
  relevant passages into context instead of whole files.
- **Graph:** *"how do the docs link together, and which have fallen behind?"* A
  library graph of intra-repo references (markdown links, `see X.md`, HTML hrefs) joined
  with per-file git freshness, written to `.concord/graph.json` and rendered live in the
  **Graph tab** of `concord ui` (nodes coloured by freshness, lagging nodes ringed).
  Flags docs last edited well before the neighbours they link to (`lagging`); export with
  `concord graph --mermaid` (or `--dot`) to paste a freshness-coloured map into a README/PR.
- **Coverage:** *"what's undocumented, and which docs trail the code?"* `concord coverage`
  reads the same graph: code files with **no inbound doc-link** (highest-churn first) and
  docs that lag the code they reference.

The core is **deterministic**: the lint is regex, the ranking is geometry, and
extraction and the contradiction radar run without a model. A language model enters
only as an *optional* pass to adjudicate flagged contradictions or synthesise
retrieved passages, handed only what Concord selected.

## Why it exists

Two failure modes plague any repo where strategy, internal notes, and public-facing
copy live side by side:

1. **Leaks:** an internal codename or a retired product name slips into a published
   page.
2. **Drift:** the same fact (a price, a policy, a product name) is stated three
   different ways across three files, and nobody notices.

A plain `grep` catches neither paraphrases nor contradictions. A vector search alone
is fuzzy and misses exact strings. Concord runs both signals together.

## Token efficiency

Concord hands a model only the relevant passages, not the whole repo. On a large
production repository (a few million tokens), answering *"find contradictory pricing
information"* (chars/4 token estimate; reproduce with the commands in [`eval/`](eval/)):

| Approach | Tokens into context | Gives you the conflicting values? |
|----------|--------------------:|-----------------------------------|
| Read the whole repo | **millions** | Yes, but it won't fit most context windows, and you pay for all of it on every query. |
| [Graphify](https://github.com/safishamsi/graphify) (knowledge graph) | **~1,565** | **No.** 46 concept nodes plus `file:line` pointers. Tells you *what relates to pricing*, not *where the numbers disagree*; you still open the files. |
| **Concord** (passage retrieval) | **a few hundred** | **Yes.** The verbatim passages, cited to `file:line`. |

A knowledge graph like [Graphify](https://github.com/safishamsi/graphify) maps how
*concepts* connect, useful for orientation. Concord retrieves the *verbatim prose* where
a claim lives, and its radar names the specific values that disagree. They're
complementary: the graph for structure, Concord for the exact conflicting lines.

These counts are the read *budget* (what goes into context), not a measure of answer
quality; retrieval accuracy is benchmarked separately in [`eval/`](eval/).

> **Completeness queries.** The numbers above are for *targeted* questions. For "find
> **all** X" sweeps (e.g. "every GDPR commitment"), a small top-k with an aggressive
> cutoff *under-retrieves*: it returns four near-identical clauses and misses the
> scattered rest. `concord read --all` handles this -- it clusters a generous candidate
> pool into facets and keeps walking the ranking while each hit is on-topic **or** adds a
> **new facet**, stopping only when neither holds. Recall-complete at the cost of more
> tokens (the recall-vs-tokens trade, made explicit). Concord prints what it retrieved
> either way, so the gap is never hidden.

## Updating: only what changed

The index records the commit it was built at (`.concord/meta.json`) **and** a
content-hash manifest (`.concord/manifest.json`). `concord update` re-embeds only the
diff:

- **In a git repo:** asks git what changed since the indexed commit (or just
  `HEAD~1..HEAD` with `--last-commit`, for a post-commit hook).
- **Outside git (`--no-git`, or a non-git folder):** diffs the content-hash manifest,
  so a real edit re-embeds and a bare `touch` does not.

Either way, cost scales with the diff, not the corpus.

## In CI: the leak guard and badge

Fail the build if a codename reaches a public file, and stamp a badge on your README:

```yaml
# .github/workflows/leak-guard.yml
on: [push, pull_request]
jobs:
  leak-guard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install concord-ai && concord lint . --scope public
```

```bash
concord badge .    # -> ![Concord](https://img.shields.io/badge/concord-0%20leaks-brightgreen)
```

The radar and graph carry the same CI story via exit codes (nonzero fails the build):

```bash
concord radar . --fail-on conflict     # fail on any value contradiction (or --fail-on verified, LLM-confirmed only)
concord graph . --fail-on-lagging      # fail if a doc lags the code it links to
concord coverage . --fail-on undocumented   # fail on undocumented code surface
```

Or wire them as **pre-commit hooks** (this repo ships `.pre-commit-hooks.yaml`):

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/linnetlabs/concord
    rev: v0.0.1
    hooks:
      - id: concord-lint            # block a codename leak
      - id: concord-radar           # block a value contradiction
```

For findings annotated inline on the PR, emit **SARIF** and upload it to GitHub
code-scanning (leaks and contradictions then appear in the "Code scanning" tab, on the
offending lines):

```yaml
# .github/workflows/concord.yml
      - run: pip install concord-ai && concord lint . --sarif > concord.sarif
      - uses: github/codeql-action/upload-sarif@v3
        with: { sarif_file: concord.sarif }
```

## Find drift across history

```bash
concord radar .                 # value-conflict candidates (same topic, different number)
concord drift "$49"             # which commits changed this value (git pickaxe)
```

## The driver model

Concord's core is a set of deterministic primitives. *Who drives the loop* is
pluggable:

| Driver | Surface | Relevance judge |
|--------|---------|-----------------|
| Human  | `concordai` (Python CLI), live explorer (`concord ui`) | geometry, or your eyes |
| Agent  | Claude skill, or MCP server (`concord serve --mcp`) | the model |

Same engine underneath. A human sits in the seat an agent would otherwise occupy.
`concord serve --mcp` speaks MCP (Model Context Protocol) on stdin/stdout, exposing
`find` / `read` / `radar` / `graph` / `coverage` / `lint` as tools any MCP client can call.

## Install

```bash
pip install concord-ai              # lint + exact find (no ML dependencies)
pip install "concord-ai[embeddings]"  # + sentiment.ai embedder for semantic find / read
```

Embeddings come from [sentiment.ai](https://github.com/BenWiseman/sentiment.ai), which
downloads a small e5 model (about 90 MB) and runs it locally on first use. No API key,
and no data leaves your machine. sentiment.ai is the only embedding backend; Concord
never silently swaps in a different model, since that would make a result look the same
while being incomparable.

## Quickstart

```bash
concord init   .                           # scaffold rules.yaml + gitignore it and .concord/
concord lint   .                           # fail CI if a banned term reaches a public file
concord index  .                           # build the semantic index (self-ignored)
concord find   "annual subscription pricing"  # exact + semantic hits, cited to file:line
concord read   "what have we said about pricing?"   # retrieve the relevant passages
concord read   "every GDPR commitment" --all   # recall-complete sweep (walk while new facets appear)
concord radar  . --verify                  # find contradictions; --verify lets an LLM confirm + name the canonical value
concord radar  . --prose                   # also catch non-numeric contradictions (SOC2 certified vs in progress); LLM-judged
concord graph  .                           # library graph (doc-links + git freshness) -> .concord/graph.json
concord graph  . --mermaid                 # export a freshness-coloured Mermaid graph to paste into a README/PR
concord coverage .                         # undocumented code (no inbound doc-link) + docs lagging their code
concord resolve .                          # walk confirmed contradictions and apply the fix (interactive; --apply = auto)
concord report . --out report.html         # shareable consistency report (lint + radar)
concord drift  "$49"                       # which commits changed a value (git pickaxe)
concord topics .                           # annotated topic map (browse; --samples to name them)
concord ui     .                           # premium live explorer: search, topics, radar, + the doc-link Graph tab
concord serve  . --mcp                     # run as an MCP server so agents can call find/read/radar/graph/coverage/lint
```

## AI is optional, and it's *your* key

Everything core is **free and deterministic**: lint, find, index, topics, radar candidates, report.
The optional LLM steps (`radar --verify`, `resolve`, and naming topics in the explorer) call **your own
API key** (you pay for usage), and the tool is explicit about it everywhere (a status pill, cost tooltips,
CLI notes).

- Set any of `DEEPSEEK_API_KEY` (the cheap default -- verify/label/resolve is constrained JSON judging,
  so a frontier model is overkill), `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`,
  `MISTRAL_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`. Auto-selection prefers DeepSeek when its key
  is present, then falls through that order; the explorer's settings picker uses whichever keys you actually have.
- `CONCORD_NO_LLM=1` turns AI off entirely; `CONCORD_LLM=<provider>` forces one.
- No key? Everything except verify / resolve / AI-naming still works.

> **Your real ruleset stays private, enforced not trusted.** `concord init` copies
> `rules.example.yaml` to `rules.yaml` and adds `rules.yaml`, `*.local.yaml`, and
> `.concord/` to your repo's `.gitignore`. The built index writes its own
> `.concord/.gitignore` too. A tool that prevents codename leaks must not leak the
> codenames, so it makes them uncommittable for you.

## Status and links

`init` and `lint` are stable with no ML dependency. The semantic features work and are
tested, but are young, so treat them as beta. See [`eval/README.md`](eval/README.md)
for the benchmarks and [the paper](paper/paper.md) for the design.

[GitHub](https://github.com/linnetlabs/concord), MIT licensed, a [Linnet Labs](https://linnetlabs.org) project.
