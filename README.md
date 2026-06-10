# Concord

![Concord leak guard](https://img.shields.io/badge/concord-0%20leaks-brightgreen)
![tests](https://img.shields.io/badge/tests-39%20passing-brightgreen)
![PyPI](https://img.shields.io/pypi/v/concord-ai?color=blue)
![license](https://img.shields.io/badge/license-MIT-blue)

> ## ⚡ Up to ~11,600× less context to audit a large repo
>
> Concord answers a question about your repo from the **exact passages that matter** —
> cited to `file:line` — reading a **near-constant ~200 tokens per query whatever the
> repo's size.** On a 3.1M-token production codebase that's a **measured ~11,600×
> reduction** versus reading the whole thing into a model, and the win grows with the
> repo: ~90% less context on a small repo → **99.99%** at 3M tokens.
> [_Measured across five real repositories →_](paper/)

**Keep a sprawling repo telling one story.**

Concord indexes the prose in a repository — docs, marketing copy, specs, READMEs —
and lets you ask it three kinds of question:

- **Lint** — *"does any internal codename / retired term / banned phrase appear in a
  file that ships publicly?"* Deterministic, exact-match, recall‑complete on a known
  list. Runs in CI or a pre-commit hook.
- **Find** — *"where else do we say something like this?"* Exact **and** semantic
  matches in one ranked result, so it catches paraphrases a `grep` would miss.
- **Read** — *"summarise everything we've said about X, and flag where it contradicts
  itself."* Retrieval-first, so only the relevant passages are pulled into context
  instead of whole files.

Concord is **computed, not generated**. The lint is regex. The ranking is geometry
(cosine + an elbow cutoff). The only place a language model enters is the optional
final *synthesis* of retrieved passages — and even that step is handed only the
passages Concord selected, which is where the token savings come from.

## Why it exists

Two failure modes plague any repo where strategy, internal notes, and public-facing
copy live side by side:

1. **Leaks** — an internal codename or a retired product name slips into a published
   page.
2. **Drift** — the same fact (a price, a policy, a product name) is stated three
   different ways across three files, and nobody notices.

A plain `grep` catches neither paraphrases nor contradictions. A vector search alone
is fuzzy and misses exact strings. Concord runs both signals together.

## Token efficiency

Concord hands a model only the passages that matter, not the whole repo. Measured
on a **24,416-passage** repository, answering *"find contradictory pricing
information"* (chars/4 token estimate; reproduce with the commands in [`eval/`](eval/)):

| Approach | Tokens into context | Gives you the conflicting values? |
|----------|--------------------:|-----------------------------------|
| Read the whole repo | **~3,100,000** | Yes — but it won't fit most context windows, and you pay for all of it on every query. |
| [Graphify](https://github.com/safishamsi/graphify) (knowledge graph) | **~1,565** | **No** — 46 concept nodes + `file:line` pointers. Tells you *what relates to pricing*, not *where the numbers disagree*; you still open the files. |
| **Concord** (passage retrieval) | **~290** | **Yes** — the verbatim passages, cited to `file:line`. |

A knowledge graph like [Graphify](https://github.com/safishamsi/graphify) maps how
*concepts* connect — useful orientation. Concord retrieves the *verbatim prose* where
a claim lives, and its radar names the specific values that disagree. They're
complementary: the graph for structure, Concord for the exact conflicting lines.

> **Honest caveat — completeness queries.** These numbers are for *targeted* questions.
> For "find **all** X" sweeps (e.g. "every GDPR commitment"), a small top-k with an
> aggressive cutoff *under-retrieves*: it can return four near-identical clauses and
> miss the scattered rest. That's a recall-vs-tokens trade, and it's exactly where a
> topic/cluster index helps (see Roadmap). Concord prints what it retrieved so the gap
> is visible, never hidden.

## Updating: only what changed

The index records the commit it was built at (`.concord/meta.json`) **and** a
content-hash manifest (`.concord/manifest.json`). `concord update` re-embeds only the
diff:

- **In a git repo:** asks git what changed since the indexed commit (or just
  `HEAD~1..HEAD` with `--last-commit`, for a post-commit hook).
- **Outside git (`--no-git`, or a non-git folder):** diffs the content-hash manifest,
  so a real edit re-embeds and a bare `touch` does not.

Either way, cost scales with the diff, not the corpus.

## In CI — the leak guard + badge

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
      - uses: linnetlabs/concord@v1     # the reusable action
        with: { scope: public }
```

```bash
concord badge .    # -> ![Concord](https://img.shields.io/badge/concord-0%20leaks-brightgreen)
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
| Agent  | Claude skill / MCP server | the model |

Same engine underneath. A human sits in the seat an agent would otherwise occupy.

## Install

```bash
pip install concord-ai              # lint + exact find (no ML dependencies)
pip install "concord-ai[embeddings]"  # + sentiment.ai embedder for semantic find / read
```

Embeddings come from [sentiment.ai](https://github.com/BenWiseman/sentiment.ai) — its
sibling package — so Concord inherits a local, auditable, provenance-tracked
embedder (e5 on-device by default) rather than calling a hosted API. sentiment.ai is
the **only** embedding backend: Concord never silently swaps in a different model,
because that would make a result look the same while being incomparable.

## Quickstart

```bash
concord init   .                           # scaffold rules.yaml + gitignore it and .concord/
concord lint   .                           # fail CI if a banned term reaches a public file
concord index  .                           # build the semantic index (self-ignored)
concord find   "founding-free pricing"     # exact + semantic hits, cited to file:line
concord read   "what have we said about pricing?"   # retrieve the relevant passages
concord radar  . --verify                  # find contradictions; --verify lets an LLM confirm + name the canonical value
concord resolve .                          # walk confirmed contradictions and apply the fix (interactive; --apply = auto)
concord report . --out report.html         # shareable consistency report (lint + radar)
concord drift  "$49"                       # which commits changed a value (git pickaxe)
concord topics .                           # annotated topic map (browse; --samples to name them)
concord ui     .                           # premium live explorer in your browser (search · topics · radar)
```

## AI is optional — and it's *your* key

Everything core is **free and deterministic**: lint, find, index, topics, radar candidates, report.
The optional LLM steps — `radar --verify`, `resolve`, and naming topics in the explorer — call **your own
API key** (you pay for usage), and the tool is explicit about it everywhere (a status pill, cost tooltips,
CLI notes).

- Set any of `ANTHROPIC_API_KEY` (preferred — the better judge), `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`,
  `GROQ_API_KEY`, `MISTRAL_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`. The explorer's ⚙ picks among
  the keys you actually have.
- `CONCORD_NO_LLM=1` turns AI off entirely; `CONCORD_LLM=<provider>` forces one.
- No key? Everything except verify / resolve / AI-naming still works.

> **Your real ruleset stays private — enforced, not trusted.** `concord init` copies
> `rules.example.yaml` to `rules.yaml` and adds `rules.yaml`, `*.local.yaml`, and
> `.concord/` to your repo's `.gitignore`. The built index writes its own
> `.concord/.gitignore` too. A tool that prevents codename leaks must not leak the
> codenames — so it makes them uncommittable for you.

## Status

Early scaffold. `lint` works today (no ML required). Semantic `find` / `read` and the
benchmark harness are in progress. See [`eval/README.md`](eval/README.md) for the
benchmark design (seed-efficiency, stopping-strategy, token-efficiency).

MIT licensed. A Linnet Labs project.
