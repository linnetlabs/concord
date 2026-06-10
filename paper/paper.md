---
title: 'Concord: Cross-file contradiction detection and leak guarding for code and documentation'
tags:
  - Python
  - documentation
  - consistency
  - contradiction detection
  - information retrieval
  - large language models
  - software maintenance
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

Large repositories mix prose and code: pricing pages, onboarding guides and
security policies alongside the configuration files, string literals and constants
that actually drive the product. As these evolve independently, hard facts drift
apart — a price quoted one way on a pricing page and another in an FAQ; an anonymity
floor set to `MIN_RESPONDENTS = 8` in a Python module but `"min_n": 5` in a config
file. Such contradictions are easy to ship and hard to find by hand.

`Concord` (installed as `concord-ai`) is an open-source Python tool that keeps a
repository telling one story. It offers three composable capabilities: (1) a
**typed-value contradiction radar** that surfaces semantically similar passages
carrying disjoint hard values of the same kind — prices, thresholds, durations, and
named code/config constants — across both prose and source; (2) a **deterministic
leak guard** that fails continuous integration if a banned term (an internal
codename, a retired product name) reaches a file marked public; and (3) **semantic
retrieval** that answers a plain-language question with the exact passages that
matter, each cited to `file:line`.

To work across file types, Concord extracts the meaningful units from each through a
modular per-extension registry — visible text from HTML (skipping `<script>` and
`<style>`), comments, string literals and named constants from code and
configuration, paragraphs from documentation — so supporting a new language is a
single function. The index is built incrementally: `concord update` re-embeds only
the files a `git diff` reports as changed (with a content-hash fallback when there is
no git), and `concord init` scaffolds the index and a private ruleset and adds both
to `.gitignore`, so neither is ever committed. Everything runs on-device using local
sentence embeddings via `sentiment.ai` [@sentimentai]; an optional language-model
pass can adjudicate flagged contradictions, but the core pipeline is deterministic
and free to run.

# Statement of need

**Contradictions hide across files, and across file types.** A `grep` for `$49`
will not find a conflicting `$39` elsewhere, and no reviewer remembers every place a
value is stated — still less that an anonymity floor in a Python module disagrees
with a JSON config. Concord's radar addresses this directly: it extracts typed
values (including code forms such as `MIN_N = 8` and `"min_n": 5`), pairs passages by
embedding similarity and shared subject words, and flags those carrying same-type but
disjoint values. On a synthetic five-document benchmark with twelve labelled conflict
pairs it recovers all twelve (recall 1.0) among 69 candidates (precision 0.17); the
low precision is deliberate for a high-recall pre-filter, and an optional
language-model pass adjudicates the false candidates by reading full context. The
asymmetry is intentional: a missed inconsistency is expensive, a false candidate
costs one dismissal.

**Auditing a repository with a language model is token-expensive.** Feeding a whole
repository into context to answer one question scales poorly; Concord instead
retrieves a small, ranked, citable set of passages at a near-constant cost. Across
five repositories spanning two orders of magnitude (16K to 2.08M corpus tokens),
answering a query reads only 150–307 tokens, while the naive baseline of reading the
whole corpus grows with it (\autoref{fig:scaling}); context reduction therefore rises
from ~98% to ~99.99% with scale. Neighbouring tools solve different problems: prose linters such as `Vale` [@vale] enforce per-sentence rules
but do not reason across files about whether two passages agree; retrieval frameworks
such as `LlamaIndex` [@llamaindex] provide general document search but are not
oriented toward consistency auditing, `file:line` citation, or deterministic leak
prevention in CI; and knowledge-graph tools such as Graphify [@graphify] map how
concepts relate, returning concept nodes with file pointers — useful orientation, but
not the conflicting text. Asked *"find contradictory pricing information"* over a
production repository, Graphify returns 46 concept nodes (~1,565 tokens) of pointers,
whereas Concord returns the verbatim passages cited to `file:line` and its radar
names the values that disagree.

![Per-query read cost is near-constant (~150–307 tokens) across five repositories
spanning 16K–2.08M corpus tokens (left, log–log), while reading the whole corpus
grows with size, so context reduction climbs toward 100% (right). The largest is a
proprietary production repository (~103K passages); token counts use a chars/4
proxy.\label{fig:scaling}](fig_scaling.png){ width=100% }

Concord is aimed at technical writers and developers maintaining product, policy and
configuration that must stay consistent, and at LLM-agent workflows that need
grounded, citable context from a repository without ingesting it whole.

# Functionality and design

Files become citable, line-numbered passages in two ways. For the semantic index,
structure-aware extraction keeps the contradiction-worthy content and discards
syntax; for the leak guard, the raw text is scanned, because a codename can hide in
an attribute or a minified string that extraction would strip. Passages are embedded
into a NumPy matrix under a gitignored `.concord/` directory; search is exact cosine
nearest-neighbour, with multiple query phrasings max-merged and a
maximal-marginal-relevance re-ranker [@carbonell1998] suppressing near-duplicates.
The contradiction detector compares only value-bearing passages, so its cost is
quadratic in that subset, not the whole corpus.

The leak guard is separate and dependency-light: no machine-learning packages,
recall-complete over a user-supplied term list, run on every commit. Protected terms
live in a gitignored ruleset, so the sensitive list never enters version control;
only a generic example ships with the package. `concord types` reports exactly which
file types are indexed. The software includes a test suite, continuous integration,
documentation, a reproducible benchmark with a public synthetic corpus, and an MIT
licence.

# Acknowledgements

Concord is developed by Linnet Labs. Its semantic features build on the
`sentiment.ai` embedding package.

# References
