"""Render the token-efficiency scaling figure from results_scaling.json.

To keep the figure legible, it plots three well-separated repositories spanning
three orders of magnitude (the full five-repo measurement lives in the JSON and
is cited in the paper). Two panels:
  (left)  tokens read per query vs corpus size — read cost stays in a low band
          while the naive full-corpus baseline grows linearly.
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
K = DATA["fixed_k"]
by_name = {r["repo"]: r for r in DATA["repos"]}

# Two well-separated anchors spanning three orders of magnitude (smallest and
# largest). All five measured repos are in results_scaling.json and the paper.
PLOT = ["roperators", "ProdRepo (priv.)"]
rows = [by_name[n] for n in PLOT if n in by_name]

ctok = [r["corpus_tokens"] for r in rows]
read = [r["read_tokens_fixed10"] for r in rows]
red = [r["reduction_fixed10_pct"] for r in rows]
names = [r["repo"] for r in rows]

INK, ACC, ACC2, MUT = "#0b1120", "#16a34a", "#2563eb", "#64748b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.8})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

# ── Panel 1: read cost vs corpus, log-log ──
xline = [min(ctok) * 0.6, max(ctok) * 1.6]
ax1.plot(xline, xline, "--", color=MUT, lw=1.3, label="naive: read whole corpus", zorder=1)
ax1.plot(ctok, read, "o-", color=ACC, lw=2.2, ms=9, label=f"Concord: read top-{K}", zorder=3)
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlim(xline)
ax1.set_xlabel("corpus size (tokens, log)")
ax1.set_ylabel("tokens into LLM context / query (log)")
ax1.set_title("Read cost stays in a low band as the corpus grows", fontsize=11, color=INK, weight="bold")
ax1.grid(True, which="both", ls=":", lw=0.5, color="#e2e8f0")
ax1.legend(frameon=False, fontsize=9, loc="upper left")
lab_off = {"roperators": (30, 4), "ProdRepo (priv.)": (-36, 4)}
for x, y, n in zip(ctok, read, names):
    dx, dy = lab_off.get(n, (6, 8))
    ax1.annotate(f"{n}\n{y:,.0f} tok", (x, y), textcoords="offset points", xytext=(dx, dy),
                 fontsize=8.5, color=INK, ha="center")

# ── Panel 2: reduction vs corpus ──
ax2.plot(ctok, red, "o-", color=ACC2, lw=2.2, ms=9, zorder=3, label="measured reduction")
struct = [100 * (1 - K / r["n_passages"]) for r in rows]
ax2.plot(ctok, struct, "--", color=MUT, lw=1.2, zorder=2, label="structural 1 − k/N")
ax2.set_xscale("log")
ax2.set_ylim(88, 101)
ax2.set_xlim(xline)
ax2.set_xlabel("corpus size (tokens, log)")
ax2.set_ylabel("context reduction vs full corpus (%)")
ax2.set_title("Reduction approaches 100% with scale", fontsize=11, color=INK, weight="bold")
ax2.grid(True, which="both", ls=":", lw=0.5, color="#e2e8f0")
ax2.legend(frameon=False, fontsize=9, loc="lower right")
red_off = {"roperators": (28, -4), "ProdRepo (priv.)": (-30, -4)}
for x, y, n in zip(ctok, red, names):
    dx, dy = red_off.get(n, (0, 8))
    ax2.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(dx, dy),
                 fontsize=8.5, color=INK, ha="center")

fig.suptitle("Concord reads a few hundred tokens per query whatever the repo size "
             f"({ctok[0]:,} → {ctok[-1]:,} corpus tokens)",
             fontsize=12, weight="bold", color=INK, y=1.02)
fig.tight_layout()
out = HERE.parent / "paper" / "fig_scaling.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}  (plotted: {', '.join(names)})")
