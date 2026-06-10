---
name: concord
description: "Check a repo for codename/leak/consistency issues and answer questions about its prose. Use when the user asks to lint for banned terms before publishing, find where wording is repeated or contradicted across files, or summarise what the repo says about a topic. If a rules.yaml exists, treat 'check for leaks' as `concord lint`."
---

# /concord

Concord turns a repo's prose into something you can lint and query. You (the agent)
are one possible *driver* — you call the same primitives a human would from the
Python package or the live explorer. The engine is deterministic; your job is to
drive the loop and synthesise.

## Commands

```
/concord lint                 # fail if a banned term reaches a public file
/concord find "<query>"       # exact + semantic hits across the repo, ranked
/concord read "<question>"    # retrieve relevant passages, then YOU synthesise
```

## Setup (once)

```bash
pip install "concord-ai[embeddings]"   # or just concord-ai for lint-only
concord init .                          # scaffold rules.yaml + gitignore it and .concord/
```

`concord init` makes the private ruleset uncommittable before the user ever writes a
real codename into it. The ruleset lives in `rules.yaml` (gitignored — the user's real
codenames) or falls back to the packaged `rules.example.yaml`. Never print the contents of `rules.yaml`
back into a public artefact; it is the private list of exactly the terms that must
not leak.

## lint

```bash
concord lint --scope public .
```

Exit code 1 means a banned term reached a scoped file. Report each finding as
`file:line — reason`. This is deterministic and recall-complete: do not second-guess
it or re-scan with grep. If `lint` reports semantic terms it does not cover, run
`find` for those.

## find

```bash
concord find "founding-free pricing"
```

Returns exact hits (score `=`) and semantic hits (cosine) merged and ranked. Use it
for "where else do we say this?" and consistency-on-edit ("show near-duplicate
wording before I change this line").

## read (retrieve → synthesise)

```bash
concord read "what have we said about pricing?" \
  --also "Compass and Climate price per user" --also "subscription cost tiers"
```

`read` returns a **ranked window**, not a fixed top-k. As the driver you do two things
the tool deliberately leaves to you — they recover recall a single tight query misses:

1. **Multi-query.** Pass 2–3 phrasings via `--also`. A single phrasing misses passages
   that use different words; scores are max-merged so a passage matching ANY phrasing
   surfaces. (Measured: this recovered pricing recall a single query missed entirely.)
2. **Patience-walk, no hard limit.** Read the ranked list top-down and stop after ~4–5
   *consecutive* irrelevant passages — not the first, so one stray neighbour doesn't
   halt you. A narrow question stops after 2–3; "find ALL the GDPR commitments" keeps
   going. You're reading them to synthesise anyway, so judging relevance is free.
   For a broad, multi-faceted question ("find ALL the X"), add `--facets`: each
   passage is tagged with its result-facet, so you keep walking while *new facets*
   still appear and stop only when both relevance and new facets dry up. This stops a
   wall of near-duplicate clauses about one facet from crowding the others out of view.

Then synthesise: summarise, and explicitly flag contradictions (the same price stated
two ways). Cite every claim as `file:line`; never add facts not in the passages. Honest
limit: the walk only reaches passages the ranking placed reasonably high — if recall
still looks thin, add another phrasing rather than reading deeper.

## topics — browse + name

```bash
concord topics .            # annotated topic map (browse / orientation only — NOT a query router)
concord topics . --samples  # representative passages per topic, for YOU to name them
```

`--samples` is the "spend a few tokens to describe the splits" path: the tool emits a
few central passages per cluster and **you** write a clean human name for each (the
auto-labels are heading-or-keyword guesses). Naming stays with the driver — the tool
never calls a generation model. `concord ui [path]` opens the same topic map + search
in the browser as a live explorer.

## When to use which

- "Is anything leaking before we ship?" → `lint`
- "Where else does this phrase / idea appear?" → `find`
- "Summarise / find contradictions about X" → `read`, then synthesise
- "What's even in this repo / show me a map" → `concord topics` (or `concord ui`)
