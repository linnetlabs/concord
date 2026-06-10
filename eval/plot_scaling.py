"""Render the read-budget figure from results_scaling.json.

Honest framing: with a fixed read depth (top-k), per-query read cost is bounded by
construction, it does not grow with the corpus. The figure shows that bound holding
in practice across repositories. It characterises the read BUDGET, not retrieval
quality (quality is measured on the labelled Bluebird corpus, where ground truth
exists). One panel, no derived "reduction %" curve (that is just 1 - read/corpus).

Run:  python eval/plot_scaling.py   ->  paper/fig_scaling.png
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
DATA = json.loads((HERE / "results_scaling.json").read_text())
K = DATA["fixed_k"]
rows = sorted(DATA["repos"], key=lambda r: r["corpus_tokens"])

ctok = [r["corpus_tokens"] for r in rows]
read = [r["read_tokens_fixed10"] for r in rows]
names = [r["repo"] for r in rows]

INK, ACC, MUT = "#0b1120", "#16a34a", "#64748b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.8})

fig, ax = plt.subplots(figsize=(8.2, 4.8))
xline = [min(ctok) * 0.55, max(ctok) * 1.8]
ax.plot(xline, xline, "--", color=MUT, lw=1.4, label="read the whole corpus (naive)", zorder=1)
ax.plot(ctok, read, "o-", color=ACC, lw=2.4, ms=9, label=f"Concord: read top-{K} passages", zorder=3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(xline)
ax.set_xlabel("corpus size (tokens, log scale)")
ax.set_ylabel("tokens placed in LLM context / query (log scale)")
ax.set_title(f"Read cost is bounded by the read depth (top-{K}), not the corpus size",
             fontsize=12, color=INK, weight="bold", pad=12)
ax.grid(True, which="both", ls=":", lw=0.5, color="#e2e8f0")
ax.legend(frameon=False, fontsize=10, loc="upper left")
for n, (x, y, nm) in enumerate(zip(ctok, read, names)):
    dy = 14 if n % 2 == 0 else -20
    dx = (n - 1) * 20 if x < 5e4 else 0
    ax.annotate(f"{nm}\n{y:,.0f} tok", (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=8, color=INK, ha="center")

lo, hi = int(round(min(read))), int(round(max(read)))
fig.text(0.5, -0.02, f"Across {len(rows)} repositories ({ctok[0]:,} to {ctok[-1]:,} corpus tokens) "
         f"a query reads {lo} to {hi} tokens. This is the read budget, not a measure of retrieval quality.",
         ha="center", fontsize=8.5, color=MUT)
fig.tight_layout()
out = HERE.parent / "paper" / "fig_scaling.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}  ({len(rows)} repos)")
