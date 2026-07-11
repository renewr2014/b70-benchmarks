#!/usr/bin/env python3
"""Headline chart: B70 vLLM XPU decode speed vs context, with published baselines."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"
SANS = ["DejaVu Sans"]

tokens = [488, 1874, 7493, 14939, 22460]
decode = [32.2, 31.6, 30.1, 28.5, 27.0]

LCPP = 22.5   # best-tuned llama.cpp SYCL, Qwen3.5-27B Q4 (TeksEdge)
VLLM_OLD = 13.4  # vLLM, mismatched quant, Qwen3.5-27B Q4 (TeksEdge)

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

# grid + spines
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
ax.tick_params(colors=MUTED, labelsize=9, length=0)

# reference lines (published numbers, different model/quant — annotations, not series)
ax.axhline(LCPP, color=MUTED, linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
ax.axhline(VLLM_OLD, color=MUTED, linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
ax.text(22460, LCPP + 0.7, "best published llama.cpp SYCL, 27B Q4 — 22.5",
        ha="right", va="bottom", fontsize=8.5, color=INK2, family=SANS)
ax.text(22460, VLLM_OLD + 0.7, "published vLLM, mismatched quant, 27B Q4 — 13.4",
        ha="right", va="bottom", fontsize=8.5, color=INK2, family=SANS)

# measured series
ax.plot(tokens, decode, color=SERIES, linewidth=2.2, zorder=3,
        marker="o", markersize=6.5, markerfacecolor=SERIES,
        markeredgecolor=SURFACE, markeredgewidth=1.4)

# endpoint direct labels only
ax.annotate("32.2", (tokens[0], decode[0]), textcoords="offset points",
            xytext=(2, 9), fontsize=10, fontweight="bold", color=INK, family=SANS)
ax.annotate("27.0", (tokens[-1], decode[-1]), textcoords="offset points",
            xytext=(4, 9), fontsize=10, fontweight="bold", color=INK, family=SANS,
            ha="right")
# series direct label (no legend box for a single series); ink text, colored mark
ax.plot([8300, 9000], [32.55, 32.55], color=SERIES, linewidth=2.2,
        solid_capstyle="round", zorder=3)
ax.text(9300, 31.6, "this config — Intel AutoRound INT4,\nIntel llm-scaler container",
        fontsize=9, color=INK2, family=SANS, fontweight="bold", va="bottom")

ax.set_xlim(0, 23400)
ax.set_ylim(0, 35)
ax.xaxis.set_major_locator(FixedLocator(tokens))
ax.set_xticklabels(["0.5k", "1.9k", "7.5k", "14.9k", "22.5k"], family=SANS)
ax.set_yticks([0, 10, 20, 30])
ax.set_xlabel("prompt tokens in context", fontsize=9.5, color=MUTED, family=SANS)
ax.set_ylabel("decode tok/s", fontsize=9.5, color=MUTED, family=SANS)

# title block
fig.text(0.055, 0.955, "One Arc Pro B70: 27B at 27–32 tok/s decode, nearly flat to 22.5k context",
         fontsize=13, fontweight="bold", color=INK, family=SANS, va="top")
fig.text(0.055, 0.885, "Qwen3.6-27B · vLLM XPU · single GPU, 32k context window · prefill ~1,550–1,780 tok/s throughout",
         fontsize=9.5, color=INK2, family=SANS, va="top")
fig.text(0.055, 0.052,
         "Baselines are published single figures on adjacent-generation models/quants, not same-machine head-to-heads.",
         fontsize=7.5, color=MUTED, family=SANS, va="bottom")
fig.text(0.055, 0.018,
         "Method + raw data: github.com/renewr2014/b70-benchmarks · @renewr20",
         fontsize=7.5, color=MUTED, family=SANS, va="bottom")

fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.19)
out = "/home/john/Projects/b70-benchmarks/assets/decode_vs_context.png"
import os; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, facecolor=SURFACE)
print("saved", out)
