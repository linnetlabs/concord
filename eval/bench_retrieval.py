"""Stopping-strategy benchmark: token cost vs retrieval coverage.

Compares four zero-cost stopping policies on a large index (~24k passages).
No labelled relevance judgements — reports passages-per-query (token proxy)
and score distributions. Ground-truth recall is measured on the Bluebird corpus
where relevant passages are known by construction.

Run:  python eval/bench_retrieval.py
"""
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import os

import numpy as np
from concordai.embed import get_embedder
from concordai.index import Index
from concordai.retrieve import adaptive_take, elbow_cutoff, mmr

# A large already-`concord index`-ed repository, supplied by path (reported by size
# only). Defaults to this repo so the script runs anywhere; point it at a bigger
# corpus via CONCORD_BENCH_PRIVATE_REPO to reproduce the at-scale numbers.
BIG_REPO = pathlib.Path(os.environ.get("CONCORD_BENCH_PRIVATE_REPO", str(ROOT)))
CORPUS = pathlib.Path(__file__).parent / "corpus" / "bluebird"

QUERIES = [
    "pricing plans and costs",
    "anonymity floor minimum respondents",
    "data retention policy",
    "free trial duration",
    "session timeout security",
    "encryption key rotation",
    "enterprise contract terms",
    "refund policy",
    "onboarding new accounts",
    "penetration testing schedule",
    "multi-factor authentication",
    "GDPR incident response notification",
    "survey response anonymity",
    "volume discounts",
    "trial extension request",
]

def compare_strategies(matrix, passages, query_vecs, label):
    results = {}
    for q_label, qv in query_vecs:
        qn = qv / (np.linalg.norm(qv) + 1e-9)
        sims = (matrix @ qn).tolist()
        ranked_sims = sorted(enumerate(sims), key=lambda x: -x[1])
        scores = [s for _, s in ranked_sims]
        ranked = [passages[i] for i, _ in ranked_sims]

        # fixed-5
        fixed5 = 5
        # fixed-10
        fixed10 = 10
        # elbow
        elbow_k = elbow_cutoff(scores, min_keep=1)
        # cosine-threshold 0.6
        thresh_k = sum(1 for s in scores if s >= 0.6)

        results.setdefault("fixed5",  []).append(fixed5)
        results.setdefault("fixed10", []).append(fixed10)
        results.setdefault("elbow",   []).append(elbow_k)
        results.setdefault("thresh",  []).append(thresh_k)

    summary = {}
    for strategy, counts in results.items():
        arr = np.array(counts)
        summary[strategy] = {
            "mean": round(float(arr.mean()), 1),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "min": int(arr.min()),
            "max": int(arr.max()),
        }
    return summary


def run():
    print("Loading embedder...")
    emb = get_embedder()

    # ── Large repository corpus ──
    print(f"\nLoading index ({BIG_REPO}) ...")
    idx = Index.load(BIG_REPO)
    matrix = np.asarray(idx.matrix, dtype="float32")
    n_passages = matrix.shape[0]
    print(f"  {n_passages:,} passages, dim={matrix.shape[1]}")

    print("Embedding queries...")
    t0 = time.perf_counter()
    qvecs = np.asarray(emb.embed(QUERIES, kind="query"), dtype="float32")
    embed_ms = (time.perf_counter() - t0) * 1000
    print(f"  {len(QUERIES)} queries in {embed_ms:.0f}ms ({embed_ms/len(QUERIES):.1f}ms/query)")

    qpairs = list(zip(QUERIES, qvecs))
    summary = compare_strategies(matrix, idx.passages, qpairs, "bigrepo")

    print("\n── Stopping strategy — passages returned per query (large repo) ──")
    print(f"  {'strategy':<12} {'mean':>6} {'median':>8} {'p25–p75':>14}  {'min':>5}  {'max':>5}")
    for s, v in summary.items():
        rng = f"{v['p25']:.0f}–{v['p75']:.0f}"
        print(f"  {s:<12} {v['mean']:>6.1f} {v['median']:>8.0f} {rng:>14}  {v['min']:>5}  {v['max']:>5}")

    # Token reduction vs naive (all passages)
    print(f"\n── Token reduction vs naive full-context ({n_passages:,} passages) ──")
    for s, v in summary.items():
        pct = 100 * (1 - v['mean'] / n_passages)
        print(f"  {s:<12}  {v['mean']:>6.1f} passages  ({pct:.2f}% reduction)")

    # ── Bluebird corpus — recall with known relevant passages ──
    print(f"\n── Bluebird corpus — recall on known-relevant passages ──")
    from concordai.chunk import chunk_file

    bb_passages = []
    for md in sorted(CORPUS.glob("*.md")):
        for p in chunk_file(md, rel=md.name):
            bb_passages.append(p)

    bb_texts = [p.text for p in bb_passages]
    bb_vecs  = np.asarray(emb.embed(bb_texts, kind="passage"), dtype="float32")
    bb_norms = np.linalg.norm(bb_vecs, axis=1, keepdims=True)
    bb_matrix = bb_vecs / (bb_norms + 1e-9)

    # Queries with known relevant files
    bb_queries = [
        ("pricing plans and costs",          {"pricing.md", "onboarding.md", "faq.md"}),
        ("anonymity floor minimum respondents", {"onboarding.md", "security.md", "faq.md", "hr-handbook.md"}),
        ("data retention policy",            {"pricing.md", "onboarding.md", "faq.md", "hr-handbook.md"}),
        ("session timeout inactivity",       {"security.md", "hr-handbook.md"}),
    ]

    recall_by_strategy = {s: [] for s in ["fixed5", "fixed10", "elbow", "thresh"]}
    for q_text, rel_files in bb_queries:
        qv = np.asarray(emb.embed([q_text], kind="query"), dtype="float32")[0]
        qn = qv / (np.linalg.norm(qv) + 1e-9)
        sims = (bb_matrix @ qn).tolist()
        ranked_idx = sorted(range(len(sims)), key=lambda i: -sims[i])
        scores = [sims[i] for i in ranked_idx]

        def recall_at(k):
            top_files = {bb_passages[ranked_idx[i]].file for i in range(min(k, len(ranked_idx)))}
            return len(top_files & rel_files) / len(rel_files)

        elbow_k = elbow_cutoff(scores, min_keep=1)
        thresh_k = max(1, sum(1 for s in scores if s >= 0.6))

        recall_by_strategy["fixed5"].append(recall_at(5))
        recall_by_strategy["fixed10"].append(recall_at(10))
        recall_by_strategy["elbow"].append(recall_at(elbow_k))
        recall_by_strategy["thresh"].append(recall_at(thresh_k))

    print(f"  {'strategy':<12} {'mean recall':>12}  {'mean k':>8}")
    for s, recs in recall_by_strategy.items():
        mean_r = np.mean(recs)
        mean_k = summary[s]["mean"]
        print(f"  {s:<12} {mean_r:>12.3f}  {mean_k:>8.1f}")

    out = {
        "n_queries": len(QUERIES), "n_passages_bigrepo": int(n_passages),
        "embed_ms_per_query": round(embed_ms / len(QUERIES), 1),
        "stopping_strategies": summary,
        "bluebird_recall": {s: round(float(np.mean(v)), 4) for s, v in recall_by_strategy.items()},
    }
    out_path = pathlib.Path(__file__).parent / "results_retrieval.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_path}")
    return out


if __name__ == "__main__":
    run()
