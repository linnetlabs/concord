# Concord benchmarks

Two corpora, three scripts. Accuracy is measured against ground truth on a public
synthetic corpus; read cost is measured on real repositories (which have no
relevance labels, so only cost is reported there, not quality).

## Running

```bash
pip install "concord-ai[embeddings]" matplotlib

python eval/bench_contradictions.py   # contradiction precision/recall -> results_contradictions.json
python eval/bench_retrieval.py        # stopping-strategy read depth    -> results_retrieval.json
python eval/bench_scaling.py          # read budget vs corpus size      -> results_scaling.json
python eval/plot_scaling.py           # render the read-budget figure   -> paper/fig_scaling.png
```

## Corpora

- **Bluebird** (`corpus/bluebird/`, committed): five documents with six embedded
  conflict types and twelve labelled conflict pairs. The only corpus with ground
  truth, used for contradiction precision/recall. It is a controlled smoke test,
  not an independent benchmark: the conflicts were authored to exercise the radar.
- **Real repositories** (referenced by path, not committed): `roperators`,
  `concord`, `sentiment.ai` (public), plus one large proprietary production
  repository as a scale anchor reported by size only (set `CONCORD_BENCH_PRIVATE_REPO`).
  These have no relevance labels, so they measure read cost, not retrieval quality.

## Results (reproduced 2026-06)

**Contradiction radar (Bluebird).** At the production similarity threshold (0.88)
the radar recovers 8 of 12 labelled conflicts (recall 0.67) at precision 0.50.
Relaxing the threshold to 0.60, appropriate for so small a corpus, recovers all
twelve at precision 0.17. The radar is a high-recall candidate generator; an
optional LLM pass adjudicates what it flags.

**Read budget (four repos, 17K to 2.08M corpus tokens).** With a fixed read depth
(top-10), a query reads 150 to 307 tokens regardless of corpus size, while reading
the whole corpus would grow with it. This is a property of fixed-k retrieval: a
read-budget characterisation, not a measure of retrieval quality. The exact
nearest-neighbour scan stays under about 13 ms/query (largest repo about 103K
passages).

## A note on extraction

Structure-aware extraction (visible HTML text; code comments, strings and named
constants; prose paragraphs) is what keeps read cost meaningful. Earlier raw line
chunking let un-extracted HTML and JS dominate retrieval and distorted these numbers.
