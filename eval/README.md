# Concord benchmark design

Concord publishes its own calibration. Three experiments, each reported with
bootstrap confidence intervals over resamples of the eval corpus — never a single
point estimate.

The eval corpus is **generic and public** (a fake company, "Project Bluebird"). A
private mirror runs the same harness on the real ruleset; only the public corpus
is committed.

## 1. Seed efficiency — how much hand-labelling does a user owe per concept?

The exact lint catches literal terms for free, so the embedding only has to cover
the **paraphrase tail**. Measure paraphrase recall as a function of `k`, the number
of seed anchor phrases per protected concept.

- Sweep `k ∈ {1, 2, 4, 8}`.
- Report `recall@k` on a paraphrase-only positive set (literal matches excluded —
  the lint already owns those).
- Headline number = the minimum `k` reaching ~0.95 recall. That is the user's
  labelling burden; lower is a cheaper tool to adopt.

## 2. Stopping strategy — which read-depth policy wins on recall-per-token?

`adaptive_take` supports several cutoffs. Treat the choice as an experiment:

| strategy        | model calls | what it optimises |
|-----------------|-------------|-------------------|
| fixed-k         | 0           | baseline          |
| cosine-threshold| 0           | absolute cutoff   |
| elbow           | 0           | per-query cutoff  |
| llm-patience    | O(reads)    | boundary accuracy |

Metric: **recall-per-token** on the retrieval set. Expectation: elbow is the best
zero-cost default; llm-patience only earns its tokens near the relevance boundary.

## 3. Token efficiency — the headline

Cost of answering a corpus question via retrieval vs. the naive baseline of
reading every relevant file into context (the way an agent does today).

- x-axis: tokens placed in the synthesiser's context.
- y-axis: answer quality (held-out judgement).
- Report the **knee** and ship it as the default depth.

## Metrics are asymmetric by job

- **Leak** → optimise recall at a fixed false-alarm budget (a miss is expensive; a
  false alarm is one dismiss-click).
- **Contradiction / summarise** → optimise precision of the flagged conflict (a
  false "you contradict yourself" erodes trust).

## Embedding bake-off

Run every experiment across backends and publish the table: local e5-base (default)
vs. bge-m3 / gte vs. OpenAI small/large as the **ceiling, not a dependency**. If the
local model lands within CI of the ceiling, the local default is justified with
evidence rather than assertion.

---

# Running the benchmarks

Three scripts, two corpora. Accuracy is measured against ground truth on the
public synthetic corpus; token efficiency is measured at scale on real
repositories (where no relevance labels exist).

```bash
pip install "concord-ai[embeddings]" matplotlib

python eval/bench_contradictions.py   # labelled precision/recall  → results_contradictions.json
python eval/bench_retrieval.py        # stopping-strategy / recall  → results_retrieval.json
python eval/bench_scaling.py          # token efficiency vs size    → results_scaling.json
python eval/plot_scaling.py           # render the scaling figure    → paper/fig_scaling.png
```

## Corpora

- **Bluebird** (`corpus/bluebird/`, committed) — five documents, six embedded
  conflict types, twelve labelled conflict pairs. Used for contradiction
  precision/recall and retrieval recall, where ground truth is known.
- **Real repositories** (referenced by path, not committed) — `roperators`,
  `squawkbox`, `concord`, `sentiment.ai` (public) plus one large proprietary
  production repository as a scale anchor (reported only by size, anonymised).
  Used for the token-efficiency scaling study.

## Headline results (reproduced 2026-06)

**Contradiction radar (Bluebird, labelled):** recall 1.000, precision 0.174 as a
pre-filter (0 false negatives over 12 pairs); LLM verification raises precision by
reading full passage context.

**Token efficiency (five repos, 16K → 2.08M corpus tokens, structure-aware
extraction):** reading the top ten passages costs a near-constant **150–307 tokens
per query** regardless of corpus size, while the naive full-corpus baseline grows
with it. Reduction rises with scale, ~98% → ~99.99%. The exact nearest-neighbour
scan stays at 0.1–13 ms/query across the full range (the largest is a ~103K-passage
production repository). Structure-aware extraction (visible HTML text, code
comments/strings/constants, prose paragraphs — not markup or syntax) is what makes
read cost flat: earlier raw chunking let un-extracted HTML/JS dominate retrieval and
distort these numbers.
