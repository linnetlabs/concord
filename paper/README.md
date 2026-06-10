# Paper: Concord

`concord.md` — preprint draft, written for arXiv (CS.IR / CS.CL).

## Reproducing the benchmarks

```bash
pip install "concord-ai[embeddings]"

# Contradiction detection (Bluebird corpus)
python eval/bench_contradictions.py   # → eval/results_contradictions.json

# Retrieval stopping strategy + token efficiency
python eval/bench_retrieval.py        # → eval/results_retrieval.json
```

Both scripts use the synthetic corpus at `eval/corpus/bluebird/` — five markdown
files with six embedded conflict types and 12 labelled conflict pairs. No network
access or API keys required.

## Converting to PDF

```bash
pandoc paper/concord.md -o paper/concord.pdf \
  --pdf-engine=xelatex \
  --citeproc \
  -V geometry:margin=1in \
  -V fontsize=11pt
```

Or for a two-column ACL-style layout, compile with the `acl` LaTeX class (a
`.tex` port of the markdown is straightforward).

## Key numbers (from benchmark runs)

| Metric | Value |
|--------|-------|
| Corpus size (proprietary prod. repo) | 24,416 passages |
| Query embedding latency | 313 ms (e5-small-v2, M-series CPU) |
| Token reduction at fixed-k=10 | 99.96% vs full-context |
| Contradiction recall (Bluebird) | 1.000 |
| Contradiction precision pre-LLM | 0.174 |
| False negatives | 0 |
