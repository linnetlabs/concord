---
title: 'Concord: Token-efficient documentation consistency checking via semantic retrieval'
tags:
  - Python
  - documentation
  - information retrieval
  - large language models
  - software maintenance
  - technical writing
authors:
  - name: Ben Wiseman
    orcid: 0009-0002-1023-9026
    affiliation: 1
affiliations:
  - name: Linnet Labs
    index: 1
date: 10 June 2026
bibliography: paper.bib
---

# Summary

Large software and product repositories accumulate prose across many files —
pricing pages, onboarding guides, security policies, API references, and
marketing copy. As these documents evolve independently, hard facts drift apart:
a price quoted one way in a pricing page and another in an FAQ, an anonymity
threshold of `n >= 8` in a policy and `n >= 5` in a help article. Such
contradictions are hard to find by hand and easy to ship.

`Concord` is an open-source Python tool that keeps a documentation corpus
internally consistent. It provides three composable capabilities over a
repository's text files: (1) a **deterministic leak guard** that fails
continuous-integration if a banned term (an internal codename, a retired product
name) reaches a file marked public; (2) **semantic retrieval** that answers
plain-language questions with the exact passages that matter, each cited to
`file:line`, so a language model reasons over the relevant lines rather than the
whole repository; and (3) a **typed-value contradiction detector** that surfaces
semantically similar passages carrying disjoint hard values of the same type
(prices, thresholds, percentages, durations).

Concord runs fully on-device. Semantic features are powered by local sentence
embeddings via `sentiment.ai` [@sentimentai], requiring no hosted endpoint or
API key. An optional verification stage can route flagged candidates to a
language model for adjudication, but the core pipeline is deterministic and free
to run. The tool indexes incrementally on git-diff boundaries, exposes a 14-command
command-line interface, an interactive local-server HTML explorer, and a GitHub
Action.

# Statement of need

Two workflows motivate Concord. First, **auditing a large corpus with a language
model is token-expensive.** Feeding an entire repository into a model's context
to ask a single question scales poorly: a corpus of tens of thousands of passages
must be re-attended for every query. Concord instead retrieves a small, ranked,
citable set of passages.

We measure token efficiency across five repositories spanning two orders of
magnitude in size — four public (`roperators`, `squawkbox`, `concord`, and
`sentiment.ai`; 100 to 3,565 passages) and one large proprietary production
repository (~24,400 passages, ~3.1M tokens) included as a scale anchor. Reading
the top ten retrieved passages places a near-constant 130–310 tokens in context
regardless of corpus size, whereas the naive full-corpus baseline grows linearly
(\autoref{fig:scaling}). Context reduction therefore rises with scale, from ~90%
on the smallest repository to ~99.99% on the largest — an approximately
11,600-fold reduction at 3.1M tokens — closely tracking the structural relation
$1 - k/N$ for read depth $k$ and passage count $N$. Query embedding takes
approximately 313 ms on a consumer laptop CPU; the exact nearest-neighbour scan
adds 0.2–3.3 ms across the full size range. A purely geometric stopping rule
(elbow cutoff) was found to degrade at scale — on the largest repository it
retains most of the ranked list — so a fixed read depth is the production default.

![Token-efficiency scaling across five repositories. Reading the top ten passages
keeps per-query context cost nearly flat (left) while the naive full-corpus
baseline grows linearly; context reduction consequently approaches 100% with
corpus size (right), tracking the structural $1 - k/N$ bound. The `concord`
repository is an outlier because dense HTML files segment into oversized
passages.\label{fig:scaling}](fig_scaling.png){ width=100% }

Second, **keyword search and manual review miss paraphrased inconsistencies.**
A `grep` for `$49` will not find a conflicting `$39` elsewhere, and no reviewer
reliably remembers every place a value is stated. Concord's contradiction radar
addresses this directly: it extracts typed values by regular expression, gates
candidate pairs by embedding similarity (same topic) and shared subject words,
and requires the values to be of the same type yet disjoint. On a synthetic
five-document benchmark with twelve labelled conflict pairs, the radar recovers
all twelve (recall 1.0) as a high-recall pre-filter; an optional language-model
verification pass raises precision by examining full passage context. This
asymmetric design is deliberate: a missed inconsistency is expensive, whereas a
false candidate costs one dismissal in a review queue.

Existing tools address neighbouring but distinct problems. Spell- and
style-checkers such as `Vale` [@vale] enforce per-sentence rules but do not reason
across files about whether two passages agree. Retrieval-augmented generation
frameworks such as `LlamaIndex` [@llamaindex] provide general document retrieval
but are not oriented toward consistency auditing, citation to `file:line`, or
deterministic leak prevention in CI. Concord targets the specific maintenance
task of keeping a sprawling corpus telling one story, with auditing as a
first-class, low-token operation.

Concord is aimed at technical writers, documentation engineers, and developers
maintaining product and policy documentation, as well as at LLM-agent workflows
that need grounded, citable context from a repository without ingesting it whole.

# Functionality and design

A repository's text files are segmented into paragraph-level passages with line
spans, so every result is citable like a `grep` hit. Passages are embedded into a
NumPy matrix stored under a gitignored `.concord/` directory; similarity search is
exact cosine nearest-neighbour. Multiple query phrasings are max-merged to recover
recall a single phrasing misses, and a maximal-marginal-relevance re-ranker
[@carbonell1998] suppresses near-duplicate passages. The contradiction detector
scopes its pairwise comparison to value-bearing passages, giving complexity
quadratic in that subset rather than in the full corpus.

The leak guard is intentionally separate and dependency-light: it requires no
machine-learning packages, runs on every commit, and is recall-complete over a
user-supplied term list. Protected terms live in a gitignored ruleset so that the
sensitive list never enters version control; only a generic example ships with the
package.

The software includes a test suite, continuous integration, documentation, a
reproducible benchmark harness with a public synthetic corpus, and an MIT licence.

# Acknowledgements

Concord is developed by Linnet Labs. Its semantic features build on the
`sentiment.ai` embedding package.

# References
