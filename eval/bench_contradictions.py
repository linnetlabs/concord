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

    print("Running contradiction radar (threshold=0.60 for small corpus)...")
    result = find_conflicts(passages, matrix, sim_threshold=0.60, neighbors=20, max_conflicts=200)
    found = result["conflicts"]
    print(f"  {result['value_passages']} value-bearing passages, {len(found)} candidates")

    # Match candidates to ground truth
    def files_of(c):
        return {pathlib.Path(c["a"]["file"]).name, pathlib.Path(c["b"]["file"]).name}

    hits = set()
    for c in found:
        cf = files_of(c)
        for i, (_, _, gt_files) in enumerate(GROUND_TRUTH):
            if cf == gt_files:
                hits.add(i)

    tp = len(hits)
    fp = len(found) - tp
    fn = N_TRUE_PAIRS - tp
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / N_TRUE_PAIRS

    print("\n── Contradiction radar results ──")
    print(f"  True pairs in ground truth : {N_TRUE_PAIRS:3d}")
    print(f"  Candidates returned        : {len(found):3d}")
    print(f"  True positives             : {tp:3d}")
    print(f"  False positives            : {fp:3d}")
    print(f"  False negatives            : {fn:3d}")
    print(f"  Precision                  : {precision:.3f}")
    print(f"  Recall                     : {recall:.3f}")
    print(f"  F1                         : {2*precision*recall/(precision+recall+1e-9):.3f}")

    print("\n── Missed pairs ──")
    for i, (t, desc, files) in enumerate(GROUND_TRUTH):
        if i not in hits:
            print(f"  MISS [{t}] {desc} ({sorted(files)})")

    print("\n── False positives ──")
    for c in found:
        cf = files_of(c)
        is_tp = any(cf == gt_files for _, _, gt_files in GROUND_TRUTH)
        if not is_tp:
            print(f"  FP sim={c['sim']:.3f} clash={c['clash']} files={sorted(cf)}")

    out = {
        "n_passages": len(passages), "n_files": len(list(CORPUS.glob("*.md"))),
        "value_passages": result["value_passages"],
        "n_ground_truth_pairs": N_TRUE_PAIRS, "n_ground_truth_unique": N_TRUE_UNIQUE,
        "n_candidates": len(found), "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(2*precision*recall/(precision+recall+1e-9), 4),
    }
    out_path = pathlib.Path(__file__).parent / "results_contradictions.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_path}")
    return out

if __name__ == "__main__":
    run()
