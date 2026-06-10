"""Render the token-efficiency scaling figure from results_scaling.json.

After structure-aware extraction, per-query read cost is genuinely flat — a query
pulls a small, near-constant set of passages regardless of corpus size — so the
figure plots every measured repository. The naive baseline (read the whole corpus)
rises with size; Concord's read stays put.

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
rows = sorted(DATA["repos"], key=lambda r: r["corpus_tokens"])

ctok = [r["corpus_tokens"] for r in rows]
read = [r["read_tokens_fixed10"] for r in rows]
red = [r["reduction_fixed10_pct"] for r in rows]
names = [r["repo"] for r in rows]

INK, ACC, ACC2, MUT = "#0b1120", "#16a34a", "#2563eb", "#64748b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.8})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

# ── Panel 1: read cost vs corpus, log-log — flat Concord line vs rising baseline ─
xline = [min(ctok) * 0.55, max(ctok) * 1.8]
ax1.plot(xline, xline, "--", color=MUT, lw=1.3, label="naive: read whole corpus", zorder=1)
ax1.plot(ctok, read, "o-", color=ACC, lw=2.2, ms=8, label=f"Concord: read top-{K}", zorder=3)
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlim(xline)
ax1.set_xlabel("corpus size (tokens, log)")
ax1.set_ylabel("tokens into LLM context / query (log)")
ax1.set_title("Per-query read cost is flat as the corpus grows", fontsize=11, color=INK, weight="bold")
ax1.grid(True, which="both", ls=":", lw=0.5, color="#e2e8f0")
ax1.legend(frameon=False, fontsize=9, loc="upper left")
for n, (x, y, nm) in enumerate(zip(ctok, read, names)):
    dy = 13 if n % 2 == 0 else -18
    dx = (n - 1) * 18 if x < 5e4 else 0  # fan out the clustered small repos
    ax1.annotate(f"{nm}\n{y:,.0f}", (x, y), textcoords="offset points", xytext=(dx, dy),
                 fontsize=7.5, color=INK, ha="center")

# ── Panel 2: reduction vs corpus ──
ax2.plot(ctok, red, "o-", color=ACC2, lw=2.2, ms=8, zorder=3, label="measured reduction")
ax2.set_xscale("log")
ax2.set_xlim(xline)
lo = min(red) - 1.2
ax2.set_ylim(lo, 100.3)
ax2.set_xlabel("corpus size (tokens, log)")
ax2.set_ylabel("context reduction vs full corpus (%)")
ax2.set_title("Reduction climbs toward 100% with scale", fontsize=11, color=INK, weight="bold")
ax2.grid(True, which="both", ls=":", lw=0.5, color="#e2e8f0")
for x, y, nm in zip(ctok, red, names):
    ax2.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                 fontsize=8, color=INK, ha="center")

lo_t, hi_t = int(round(min(read))), int(round(max(read)))
fig.suptitle(f"Concord reads ~{lo_t}–{hi_t} tokens per query whatever the repo size "
             f"({ctok[0]:,} → {ctok[-1]:,} corpus tokens)",
             fontsize=12, weight="bold", color=INK, y=1.02)
fig.tight_layout()
out = HERE.parent / "paper" / "fig_scaling.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}  ({len(rows)} repos: {', '.join(names)})")
