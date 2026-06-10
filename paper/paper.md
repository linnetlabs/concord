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
that drive the product. As these evolve independently, hard facts drift apart. A
price is quoted one way on a pricing page and another in an FAQ; an anonymity floor
reads `MIN_RESPONDENTS = 8` in a Python module but `"min_n": 5` in a config file.
Such contradictions are easy to ship and hard to find by hand.

`Concord` (installed as `concord-ai`) is an open-source Python tool that keeps a
repository telling one story. It offers three composable capabilities: (1) a
**typed-value contradiction radar** that surfaces semantically similar passages
carrying disjoint hard values of the same kind (prices, thresholds, durations, and
named code or config constants) across both prose and source; (2) a **deterministic
leak guard** that fails continuous integration if a banned term (an internal
codename, a retired product name) reaches a file marked public; and (3) **semantic
retrieval** that answers a plain-language question with the relevant passages, each
cited to `file:line`.

To work across file types, Concord extracts the meaningful units from each through a
modular per-extension registry (visible text from HTML, skipping `<script>` and
`<style>`; comments, string literals and named constants from code and configuration;
paragraphs from documentation), so supporting a new language is a single function. The index is built incrementally: `concord update` re-embeds only
the files a `git diff` reports as changed (with a content-hash fallback when there is
no git), and `concord init` scaffolds the index and a private ruleset and adds both
to `.gitignore`, so neither is ever committed. Everything runs on-device using local
sentence embeddings via `sentiment.ai` [@sentimentai]; an optional language-model
pass can adjudicate flagged contradictions, but the core pipeline is deterministic
and free to run.

# Statement of need

**Contradictions hide across files, and across file types.** A `grep` for `$49`
will not find a conflicting `$39` elsewhere, and no reviewer remembers every place a
value is stated, still less that a threshold in a Python module disagrees with a JSON
config. Concord's radar pairs passages by embedding similarity and shared subject
words, then flags those carrying same-type but disjoint typed values (prices,
thresholds, durations, percentages, and numeric config constants). It detects
same-type values in similar contexts rather than verifying that two occurrences name
the identical constant, so it is a candidate generator, not a verdict. On a small
synthetic corpus (five documents, twelve labelled conflict pairs), at the production
similarity threshold it recovers 8 of 12 conflicts (recall 0.67) at precision 0.50;
relaxing the threshold, appropriate for so small a corpus, recovers all twelve at
precision 0.17. The design favours recall: a missed inconsistency is expensive, a
false candidate costs one dismissal, and an optional language-model pass adjudicates
the candidates by reading full context.

**Auditing a repository with a language model is token-expensive.** Feeding a whole
repository into context to answer one question scales poorly; Concord instead
retrieves a fixed number of ranked, citable passages. By construction this bounds the
read cost: reading the top ten passages costs a few hundred tokens regardless of
corpus size (150 to 307 tokens across five repositories from 16K to 2.08M corpus
tokens, Figure 1), while reading the whole corpus would grow with it. The figure
characterises the read budget, not retrieval quality; quality is measured separately
on the labelled corpus above. Neighbouring tools solve different problems: prose
linters such as `Vale` [@vale] enforce per-sentence rules but do not reason across
files about whether two passages agree; retrieval frameworks such as `LlamaIndex`
[@llamaindex] provide general document search but are not oriented toward consistency
auditing, `file:line` citation, or deterministic leak prevention in CI; and
knowledge-graph tools such as Graphify [@graphify] map how concepts relate, returning
concept nodes and file pointers rather than the conflicting text. Measured on a large
private repository, the question *"find contradictory pricing information"* returned
from Graphify 46 concept nodes (about 1,565 tokens) of pointers, where Concord returns
the verbatim passages cited to `file:line` and names the values that disagree.

![Per-query read cost is bounded by the fixed read depth: reading the top ten
passages costs 150 to 307 tokens across five repositories spanning 16K to 2.08M
corpus tokens (left, log-log), while reading the whole corpus would grow with size
(right). This characterises the read budget, not retrieval quality. The largest
repository is a proprietary production codebase (about 103K passages); token counts
use a chars/4 proxy.\label{fig:scaling}](fig_scaling.png){ width=100% }

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
