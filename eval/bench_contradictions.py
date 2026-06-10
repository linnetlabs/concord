"""Contradiction radar precision/recall benchmark on the Bluebird synthetic corpus.

Ground truth: 6 distinct conflict types spanning the 5 documents.
Run:  python eval/bench_contradictions.py
"""
import json
import sys
import pathlib
import tempfile
import shutil

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from concordai.chunk import chunk_file
from concordai.radar import find_conflicts
from concordai.embed import get_embedder

CORPUS = pathlib.Path(__file__).parent / "corpus" / "bluebird"

# Ground truth: (type, canonical conflict description, files involved)
GROUND_TRUTH = [
    ("price",     "Starter plan: $29 vs $39",              {"pricing.md", "onboarding.md"}),
    ("threshold", "Anonymity floor: n≥5 vs n≥8",           {"onboarding.md", "security.md"}),
    ("threshold", "Anonymity floor: n≥5 vs n≥8",           {"faq.md",        "security.md"}),
    ("threshold", "Anonymity floor: n≥5 vs n≥8",           {"onboarding.md", "hr-handbook.md"}),
    ("threshold", "Anonymity floor: n≥5 vs n≥8",           {"faq.md",        "hr-handbook.md"}),
    ("duration",  "Data retention: 24 vs 36 months",       {"pricing.md",    "onboarding.md"}),
    ("duration",  "Data retention: 24 vs 36 months",       {"faq.md",        "onboarding.md"}),
    ("duration",  "Data retention: 24 vs 36 months",       {"pricing.md",    "hr-handbook.md"}),
    ("duration",  "Data retention: 24 vs 36 months",       {"faq.md",        "hr-handbook.md"}),
    ("duration",  "Free trial: 14 vs 21 days",             {"onboarding.md", "faq.md"}),
    ("duration",  "Trial extension: 7 vs 14 days",         {"onboarding.md", "faq.md"}),
    ("duration",  "Session timeout: 8 vs 4 hours",         {"security.md",   "hr-handbook.md"}),
]
N_TRUE_PAIRS = len(GROUND_TRUTH)
N_TRUE_UNIQUE = 6  # distinct conflict types

def run():
    print("Loading embedder...")
    emb = get_embedder()

    print(f"Chunking {CORPUS} ...")
    passages = []
    for md in sorted(CORPUS.glob("*.md")):
        for p in chunk_file(md, rel=md.name):
            passages.append(p)
    print(f"  {len(passages)} passages across {len(list(CORPUS.glob('*.md')))} files")

    texts = [p.text for p in passages]
    print("Embedding passages...")
    import numpy as np
    vecs = np.asarray(emb.embed(texts, kind="passage"), dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    matrix = vecs / (norms + 1e-9)

    def files_of(c):
        return {pathlib.Path(c["a"]["file"]).name, pathlib.Path(c["b"]["file"]).name}

    def evaluate(threshold):
        found = find_conflicts(passages, matrix, sim_threshold=threshold,
                               neighbors=20, max_conflicts=200)["conflicts"]
        hits = {i for c in found for i, (_, _, gt) in enumerate(GROUND_TRUTH) if files_of(c) == gt}
        tp, n = len(hits), len(found)
        precision = tp / n if n else 0.0
        recall = tp / N_TRUE_PAIRS
        return {"threshold": threshold, "n_candidates": n, "tp": tp, "fp": n - tp,
                "fn": N_TRUE_PAIRS - tp, "precision": round(precision, 4),
                "recall": round(recall, 4)}

    # 0.88 is the radar's production default; 0.60 is a relaxed setting appropriate
    # for so small a corpus. We report both rather than tune the test to pass.
    runs = {f"{th:.2f}": evaluate(th) for th in (0.88, 0.60)}
    print("\n── Contradiction radar (Bluebird, file-pair recall) ──")
    print(f"  ground-truth pairs: {N_TRUE_PAIRS}")
    for th, r in runs.items():
        print(f"  threshold {th}: {r['n_candidates']:3d} candidates  "
              f"recall {r['recall']:.2f}  precision {r['precision']:.2f}")

    out = {
        "n_passages": len(passages), "n_files": len(list(CORPUS.glob("*.md"))),
        "n_ground_truth_pairs": N_TRUE_PAIRS, "n_ground_truth_unique": N_TRUE_UNIQUE,
        "production_threshold": 0.88, "by_threshold": runs,
        "note": "File-pair recall on a controlled synthetic corpus authored to exercise "
                "the radar; a smoke test, not an independent benchmark.",
    }
    out_path = pathlib.Path(__file__).parent / "results_contradictions.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_path}")
    return out

if __name__ == "__main__":
    run()
