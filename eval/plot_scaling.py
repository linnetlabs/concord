"""Render the token-efficiency scaling figure from results_scaling.json.

Two panels:
  (left)  tokens read per query vs corpus size — read cost stays ~flat while
          the corpus (and the naive full-context baseline) grows linearly.
  (right) reduction % vs corpus size, with the structural 1 − k/N reference.

Run:  python eval/plot_scaling.py   →  paper/fig_scaling.png
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
DATA = json.loads((HERE / "results_scaling.json").read_text())
repos = sorted(DATA["repos"], key=lambda r: r["n_passages"])
K = DATA["fixed_k"]

names = [r["repo"] for r in repos]
ctok = [r["corpus_tokens"] for r in repos]
read = [r["read_tokens_fixed10"] for r in repos]
red = [r["reduction_fixed10_pct"] for r in repos]

INK, ACC, ACC2, MUT = "#0b1120", "#16a34a", "#2563eb", "#64748b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.8})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

# ── Panel 1: read cost vs corpus, both log ──
ax1.plot(ctok, ctok, "--", color=MUT, lw=1.3, label="naive: read whole corpus", zorder=1)
ax1.plot(ctok, read, "o-", color=ACC, lw=2, ms=7, label=f"Concord: read top-{K}", zorder=3)
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("corpus size (tokens, log)")
ax1.set_ylabel("tokens placed in LLM context / query (log)")
ax1.set_title("Read cost stays ~flat as the corpus grows", fontsize=11, color=INK, weight="bold")
ax1.grid(True, which="both", ls=":", lw=0.5, color="#e2e8f0")
ax1.legend(frameon=False, fontsize=9, loc="upper left")
for x, y, n in zip(ctok, read, names):
    ax1.annotate(n, (x, y), textcoords="offset points", xytext=(7, -11),
                 fontsize=7.5, color=INK)

# ── Panel 2: reduction vs corpus ──
ax2.plot(ctok, red, "o-", color=ACC2, lw=2, ms=7, zorder=3, label="measured reduction")
struct = [100 * (1 - K / r["n_passages"]) for r in repos]
ax2.plot(ctok, struct, "--", color=MUT, lw=1.2, zorder=2, label="structural 1 − k/N")
ax2.set_xscale("log")
ax2.set_ylim(60, 101)
ax2.set_xlabel("corpus size (tokens, log)")
ax2.set_ylabel("context reduction vs full corpus (%)")
ax2.set_title("Reduction approaches 100% with scale", fontsize=11, color=INK, weight="bold")
ax2.grid(True, which="both", ls=":", lw=0.5, color="#e2e8f0")
ax2.legend(frameon=False, fontsize=9, loc="lower right")
for x, y, n in zip(ctok, red, names):
    dy = 7 if n != "concord.ai" else -14
    ax2.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, dy),
                 fontsize=7.5, color=INK, ha="center")

fig.suptitle("Concord token-efficiency scaling across five repositories (100 → 24,416 passages)",
             fontsize=12, weight="bold", color=INK, y=1.02)
fig.tight_layout()
out = HERE.parent / "paper" / "fig_scaling.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
