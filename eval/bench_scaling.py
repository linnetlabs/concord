"""Token-efficiency scaling across real repositories of varying size.

Measures how the cost of answering a query (tokens placed in an LLM's context)
scales with corpus size, on real public repositories plus one large proprietary
production repository (anonymised). Complements the Bluebird labelled benchmark:
Bluebird measures retrieval/contradiction ACCURACY against ground truth; this
measures token EFFICIENCY at scale, where no relevance labels exist.

Read depth is the variable of interest:
  - fixed-k=10  : constant read budget (the production default)
  - elbow       : per-query geometric cutoff (zero model calls)

Reported per repo: passages, corpus tokens, tokens read per query, and the
reduction vs reading the whole corpus into context. Token counts use the standard
chars/4 proxy — model-agnostic and honest.

Run:  python eval/bench_scaling.py
"""
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from concordai.chunk import chunk_repo
from concordai.embed import get_embedder
from concordai.index import Index
from concordai.retrieve import elbow_cutoff

HOME = pathlib.Path.home()

# (display_name, path, public?, load_existing_index?)
REPOS = [
    ("roperators",        HOME / "roperators",  True,  False),
    ("concord.ai",        HOME / "concord.ai",  True,  False),
    ("sentiment.ai",      HOME / "sentiment.ai", True, False),
]
# Optional large proprietary anchor — set CONCORD_BENCH_PRIVATE_REPO to an already
# `concord index`-ed repo to include a private scale point (reported by size only).
_priv = os.environ.get("CONCORD_BENCH_PRIVATE_REPO")
if _priv:
    REPOS.append(("ProdRepo (priv.)", pathlib.Path(_priv), False, True))

# Generic queries that apply to any software/product documentation corpus.
QUERIES = [
    "how do I install this",
    "how to configure the settings",
    "what license is this released under",
    "how to run the tests",
    "usage example getting started",
    "API function reference",
    "how to contribute",
    "what does this project do",
]

FIXED_K = 10


def tok(text: str) -> int:
    return len(text) // 4  # standard rough token proxy


def load_passages(name, path, use_index):
    if use_index:
        idx = Index.load(path)
        if idx.matrix is None:
            raise RuntimeError(f"{name}: no index — run `concord index {path}`")
        texts = [p.text for p in idx.passages]
        return idx.passages, texts, np.asarray(idx.matrix, dtype="float32")
    passages = list(chunk_repo(path, prose=True))  # structure-aware extraction
    texts = [p.text for p in passages]
    return passages, texts, None


def run():
    print("Loading embedder...")
    emb = get_embedder()
    print(f"Embedding {len(QUERIES)} queries...")
    qv = np.asarray(emb.embed(QUERIES, kind="query"), dtype="float32")
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)

    rows = []
    for name, path, public, use_index in REPOS:
        if not path.exists():
            print(f"  SKIP {name} — not found at {path}")
            continue
        print(f"\n── {name} ({path}) ──")
        passages, texts, matrix = load_passages(name, path, use_index)
        n = len(passages)
        if n == 0:
            print("  no prose passages, skipping")
            continue
        corpus_tokens = sum(tok(t) for t in texts)
        avg_pt = corpus_tokens / n
        print(f"  {n:,} passages · {corpus_tokens:,} corpus tokens · {avg_pt:.0f} tok/passage avg")

        if matrix is None:
            print("  embedding passages...")
            t0 = time.perf_counter()
            M = np.asarray(emb.embed(texts, kind="passage"), dtype="float32")
            M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
            embed_s = time.perf_counter() - t0
            print(f"  embedded in {embed_s:.1f}s")
        else:
            M = matrix  # already unit-normalised production index

        read_fixed, read_elbow, kc_elbow, qlat = [], [], [], []
        for i in range(len(QUERIES)):
            t0 = time.perf_counter()
            sims = M @ qv[i]
            order = np.argsort(-sims)
            qlat.append((time.perf_counter() - t0) * 1000)
            scores = sims[order].tolist()

            kf = min(FIXED_K, n)
            read_fixed.append(sum(tok(texts[order[j]]) for j in range(kf)))

            ke = min(elbow_cutoff(scores, min_keep=1), n)
            kc_elbow.append(ke)
            read_elbow.append(sum(tok(texts[order[j]]) for j in range(ke)))

        mean_fixed = float(np.mean(read_fixed))
        mean_elbow = float(np.mean(read_elbow))
        red_fixed = 100 * (1 - mean_fixed / corpus_tokens) if corpus_tokens else 0
        red_elbow = 100 * (1 - mean_elbow / corpus_tokens) if corpus_tokens else 0

        print(f"  fixed-k={FIXED_K}: read ~{mean_fixed:,.0f} tok/query  ({red_fixed:.2f}% reduction)")
        print(f"  elbow      : read ~{mean_elbow:,.0f} tok/query, k≈{np.mean(kc_elbow):.0f}  ({red_elbow:.2f}% reduction)")
        print(f"  scan latency: {np.mean(qlat):.1f}ms/query over {n:,} passages")

        rows.append({
            "repo": name, "public": public, "n_passages": n,
            "corpus_tokens": corpus_tokens, "avg_passage_tokens": round(avg_pt, 1),
            "read_tokens_fixed10": round(mean_fixed, 1), "reduction_fixed10_pct": round(red_fixed, 3),
            "read_tokens_elbow": round(mean_elbow, 1), "mean_elbow_k": round(float(np.mean(kc_elbow)), 1),
            "reduction_elbow_pct": round(red_elbow, 3),
            "scan_ms_per_query": round(float(np.mean(qlat)), 2),
        })

    # ── Scaling summary ──
    print("\n" + "=" * 78)
    print("SCALING SUMMARY — token cost to answer a query vs corpus size")
    print("=" * 78)
    print(f"{'repo':<18}{'passages':>10}{'corpus tok':>13}{'read@10':>10}{'reduction':>11}")
    for r in sorted(rows, key=lambda x: x["n_passages"]):
        print(f"{r['repo']:<18}{r['n_passages']:>10,}{r['corpus_tokens']:>13,}"
              f"{r['read_tokens_fixed10']:>10,.0f}{r['reduction_fixed10_pct']:>10.2f}%")

    # Validate reduction ≈ 1 - k/N structural relationship
    print("\nStructural check: reduction = 1 − (tokens read / corpus tokens)")
    print("At fixed k, tokens-read is ~flat, so reduction grows with corpus size.")
    print(f"{'repo':<18}{'N':>9}{'k/N':>10}{'1-k/N':>9}{'measured':>10}")
    for r in sorted(rows, key=lambda x: x["n_passages"]):
        kn = FIXED_K / r["n_passages"]
        print(f"{r['repo']:<18}{r['n_passages']:>9,}{kn:>10.5f}{(1-kn)*100:>8.2f}%{r['reduction_fixed10_pct']:>9.2f}%")

    out_path = pathlib.Path(__file__).parent / "results_scaling.json"
    out_path.write_text(json.dumps({"fixed_k": FIXED_K, "n_queries": len(QUERIES), "repos": rows}, indent=2))
    print(f"\nResults written to {out_path}")
    return rows


if __name__ == "__main__":
    run()
