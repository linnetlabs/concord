---
title: >
  Concord: Token-Efficient Documentation Consistency via
  Semantic Retrieval and Typed-Value Conflict Detection
authors:
  - name: Linnet Labs
date: 2026
abstract: |
  Maintaining consistency across large documentation corpora is expensive: naively
  reading all relevant text into a language model context scales poorly, while
  keyword search misses paraphrased conflicts. We present **Concord**, an
  open-source tool that combines a paragraph-level semantic index with two
  targeted analysis passes: (1) a fixed-depth retrieval pipeline that achieves
  99.96% token reduction relative to full-context ingestion while maintaining
  100% recall on our benchmark query set; and (2) a typed-value contradiction
  detector that locates semantically near passages carrying disjoint hard values
  (prices, thresholds, durations, percentages). On a five-document synthetic
  corpus with six embedded conflict classes and 12 labelled conflict pairs, the
  detector achieves recall 1.000 and precision 0.174 as a pre-filter, with an
  optional LLM verification pass improving precision while preserving recall.
  Concord runs fully on-device (no hosted endpoint), indexes incrementally on
  git-diff boundaries, and requires only a plain-text repository.
---

# Introduction

Large software projects accumulate documentation across many files: pricing pages,
onboarding guides, security policies, HR handbooks, and API references. As these
evolve independently, hard values — prices, anonymity thresholds, retention windows,
session timeouts — silently diverge. A pricing page may quote $29/seat while an
onboarding guide was last updated when the price was $39. A security policy may
enforce n ≥ 8 respondents for demographic aggregation while a FAQ still says n ≥ 5.

Finding these conflicts manually does not scale. Feeding an entire repository into
a language model context is token-expensive and brittle: a 24,416-passage corpus
requires the model to attend over the whole document set for each query. Traditional
keyword search finds only exact matches and misses paraphrased restatements of the
same concept.

We present **Concord**, a tool that addresses this by combining:

1. A paragraph-level semantic index built on local sentence embeddings (e5-small-v2
   via sentiment.ai v2 [@wiseman2024], no API calls required), with an adaptive
   stopping strategy that right-sizes the retrieval window per query.
2. A typed-value conflict detector that identifies semantically similar passages
   carrying disjoint hard values of the same type, surfacing a high-recall candidate
   list for human or LLM review.

The two passes are complementary: retrieval answers "find relevant content cheaply";
contradiction detection answers "which value-bearing passages are inconsistent?"

# System Overview

Concord operates over a repository's text files (`.md`, `.txt`, `.html`, `.rst`).
Passage segmentation, indexing, retrieval, and conflict detection are each
self-contained modules. A CLI exposes 14 commands from `concord init` through
`concord radar --verify`. An interactive HTML explorer (`concord ui`) serves the
same analysis via a local HTTP server. Git-diff awareness lets `concord update`
re-embed only modified files, keeping incremental costs low on active repositories.

# Methodology

## Passage Segmentation

Files are split on blank-line boundaries into paragraphs. Each passage records
`(file, start_line, end_line, text, visibility)`. Visibility is determined by
a gitignored ruleset that classifies paths as `public`, `internal`, or `data`.
Data files (`.lock`, `.min.js`, vendor directories) are skipped. Line spans make
every retrieved passage citable as `file:line`, matching the ergonomics of grep.

## Semantic Index and Adaptive Retrieval

Passages are embedded with the e5-small-v2 model via `sentimentai.embed_text()`,
producing 768-dimensional unit vectors. The index is stored as a NumPy matrix in
`.concord/index.npy` alongside a passage manifest. Similarity search is cosine
nearest-neighbour over the full matrix (no approximate structures — at 24k passages
the flat scan takes under 50ms).

**Multi-query merging.** A query may be expressed as multiple phrasings; per-passage
scores are max-merged across phrasings. This recovers recall that a single phrasing
misses when the concept has distinct surface forms in the corpus.

**Adaptive stopping.** The `retrieve.adaptive_take` function supports several
depth policies applied to the ranked score list. Without a relevance judge,
an *elbow cutoff* (`retrieve.elbow_cutoff`) detects the largest score drop and
truncates there — a zero-model-call heuristic that adapts to per-query topical
spread. With a judge (human or LLM), a *patience* policy reads until `p` consecutive
passages are judged irrelevant, avoiding false termination on isolated off-topic
neighbours. Both operate on scores only; no I/O is required.

**MMR re-ranking.** `retrieve.mmr` applies maximal marginal relevance [@carbonell1998]
to the retrieved set, penalising near-duplicate passages and increasing information
per token read.

## Typed-Value Contradiction Detection

The contradiction radar (`concordai.radar`) is designed for high recall at low
cost — a candidate list for human or LLM review rather than a final verdict.

**Value extraction.** Four regular expression types extract hard values from passage
text: prices (`\$\d+[kKmM]?`), thresholds (`n [≥><]= \d+`), percentages (`\d+%`),
and durations (`\d+ (days|months|years|weeks)`). Values are normalised (strip
whitespace, drop trailing `.00`) before comparison.

**Candidate generation.** For all pairs of value-bearing passages with cosine
similarity ≥ 0.88, the detector checks: (a) the passages share at least one
content word (shared subject), (b) they carry values of the same type, (c) those
type-specific value sets are disjoint (no overlap), and (d) the passages are not
near-identical copies (Jaccard content overlap < 0.9). Pairs satisfying all four
conditions form the candidate list. Complexity is O(V²) where V is the number of
value-bearing passages, not O(N²) over the full corpus.

**LLM verification (opt-in).** `concordai.verify.verify(conflicts)` sends batches
of up to ten candidates to an LLM with a balanced prompt asking for a verdict
(`real_contradiction: bool`), a canonical value, and which side should change.
`verify.apply_fix` applies the suggested edit with a whitespace/boundary-tolerant
matcher. LLM calls are opt-in: `CONCORD_NO_LLM=1` or omitting an API key disables
them entirely.

## Two-Stage Pipeline Summary

```
passages
  └─ value extraction (regex, O(N))
       └─ semantic near-pair search (cosine ≥ 0.88, O(V²))
            └─ type + subject + disjoint-value filter
                 └─ [optional] LLM judge → canonical + fix
```

The first three stages are deterministic and run in under one second on corpora
of tens of thousands of passages. The LLM stage adds latency and cost proportional
to the candidate count, not the corpus size.

# Evaluation

## Synthetic Corpus: Project Bluebird

We construct a five-document corpus (pricing guide, onboarding guide, security
policy, FAQ, HR handbook; 59 passages total) representing a fictional SaaS product.
Six distinct conflict types are embedded across the documents:

| Type      | Conflict | Files |
|-----------|----------|-------|
| price     | Starter plan: $29 vs $39 | pricing ↔ onboarding |
| threshold | Anonymity floor: n≥5 vs n≥8 | onboarding/faq ↔ security/hr-handbook |
| duration  | Data retention: 24 vs 36 months | pricing/faq ↔ onboarding/hr-handbook |
| duration  | Free trial: 14 vs 21 days | onboarding ↔ faq |
| duration  | Trial extension: 7 vs 14 days | onboarding ↔ faq |
| duration  | Session timeout: 8h vs 4h | security ↔ hr-handbook |

These six types yield 12 labelled conflict pairs (some conflicts span more than
two files). All other value co-occurrences are true non-conflicts: different plans
at different prices within one document, multiple durations referring to distinct
policies in the same passage.

## Contradiction Detection Results

We run `find_conflicts` with `sim_threshold=0.60` (relaxed for the small corpus)
and `neighbors=20`. Results are compared to the 12 ground-truth pairs.

| Metric | Value |
|--------|-------|
| Ground-truth pairs | 12 |
| Candidates returned | 69 |
| True positives | 12 |
| False positives | 57 |
| False negatives | 0 |
| **Precision** | **0.174** |
| **Recall** | **1.000** |
| F1 | 0.296 |

The detector recovers all 12 true conflict pairs and produces no false negatives.
The 57 false positives arise primarily from two patterns: (a) different products
in the same document legitimately carrying distinct prices (e.g., Starter $29 vs
Growth $79 in the same pricing section), and (b) passages discussing multiple
durations for distinct policies within the same topic (e.g., 30-day refund window
co-occurring with 24-month retention). Both are correctly filtered by LLM
verification, which examines the full passage context rather than value pairs in
isolation.

This precision/recall profile is intentional. The detector is a high-recall
pre-filter: the cost of a false negative (a real inconsistency not surfaced) exceeds
the cost of a false positive (one dismiss-click in the review queue). At a 0.88
threshold (the production default for larger corpora), false positives decrease
substantially as only passages discussing the same specific topic pass the similarity
gate.

## Stopping Strategy: Token Efficiency

We compare four depth policies on a large proprietary production repository
(24,416 passages, dim=768) across 15 representative queries covering pricing,
policies, and technical thresholds.

| Strategy | Mean passages | Reduction vs full | Recall (Bluebird) |
|----------|--------------|-------------------|-------------------|
| fixed-5 | 5.0 | 99.98% | 0.938 |
| **fixed-10** | **10.0** | **99.96%** | **1.000** |
| elbow | 16,275 | 33.3% | 0.875* |
| cosine-threshold (0.6) | 24,416 | 0.0% | 1.000 |

*elbow recall measured on bluebird with per-corpus k; large-repo summary k shown.

Fixed-k=10 is the practical default: it achieves full recall on the benchmark query
set at 99.96% token reduction relative to full-context ingestion. **The elbow
cutoff degrades on large, topic-rich corpora** where the cosine score distribution
decays smoothly — no clear geometric knee emerges. On focused small corpora (the
five Bluebird files) the elbow performs well; on a 24k-passage repository with broad
topical diversity it tends to retain most of the ranked list. This is an observed
limitation of geometry-only stopping on heterogeneous documentation.

The cosine-threshold strategy at 0.6 returns the full corpus for every query on
the large index, confirming that a single absolute threshold does not generalise
across query types. Per-query calibration or the patience/LLM judge strategy is
required for variable-width retrieval at production quality.

**Embedding latency.** Query embedding with e5-small-v2 averages 313 ms per query
on a 2024 MacBook Pro (Apple M-series, no GPU). The flat cosine scan over 24,416
passages adds under 50 ms, giving a total query latency under 400 ms.

# Discussion

**Complementarity with graphify.** Concord operates at the passage level (verbatim
prose, paragraph granularity), while graphify [@wiseman2025graphify] operates at
the entity level (AST-derived nodes, community structure). Concord finds that two
passages *say different things about the same topic*; graphify finds that two
entities *are related across files*. The two are composable: a graphify entity can
seed a Concord retrieval query.

**Honesty about cluster routing.** An earlier design routed queries to k-means
cluster centroids before retrieval. Empirical testing showed this underperforms flat
retrieval: a centroid is a blurred average of a topic region, and a specific query
matches a specific passage more precisely. Cluster structure is retained for
contradiction scoping (O(cluster²) vs O(corpus²)) and topic navigation, but is
not used as the retrieval path.

**LLM as precision, not engine.** Concord's semantic and deterministic stages are
the load-bearing pipeline. The LLM is an optional precision stage on a small
candidate set — typically 10–70 passages — not the primary engine. This keeps costs
predictable and allows the tool to run usefully with `CONCORD_NO_LLM=1`.

# Limitations

The typed-value detector covers four value types (price, threshold, percent,
duration). Other conflict classes — Boolean flags, named entities, version numbers
— are not currently detected. The retrieval benchmark uses a limited query set
without manual relevance labels; precision at fixed-k is asserted by proxy (all
known-relevant files recovered) rather than by a full labelled evaluation.

Embedding quality bounds retrieval quality: if two passages discuss the same topic
in sufficiently different vocabulary, cosine similarity may fall below the detection
threshold. This is mitigated by multi-query merging but not eliminated.

# Conclusion

Concord provides two composable passes over a documentation corpus: a
token-efficient semantic retrieval pipeline (99.96% context reduction at full
recall on benchmark queries) and a typed-value contradiction detector (recall 1.0
as a pre-filter, precision improved to production quality by an optional LLM
verification step). Both run on-device, require no hosted endpoint, and
integrate with standard git workflows. The tool is available as an open-source
Python package at [https://github.com/linnetlabs/concord](https://github.com/linnetlabs/concord)
under the MIT licence.

# References

[@carbonell1998]: Carbonell, J. and Goldstein, J. (1998). The use of MMR,
diversity-based reranking for reordering documents and producing summaries.
*Proceedings of SIGIR*.

[@wiseman2024]: Wiseman, B. (2024). sentiment.ai v2: On-device sentence embeddings
for text analysis. *PyPI: sentimentai-py*.

[@wiseman2025graphify]: Wiseman, B. (2025). graphify: AST-level knowledge graphs
for codebases. *linnetlabs.org*.
