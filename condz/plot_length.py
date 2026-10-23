"""Figure: condz AUROC against the length of the scored text, Qwen vs Llama, for detectors
trained on full texts and on the first 100 tokens only.

Reads the window-grid results written by condz/length.py (repeat 0):
    runs/qwen/window_grid_results_r0.json
    runs/llama/window_grid_results_r0.json
Writes results/fig_length_auroc.pdf (for LaTeX) and a .png preview.

Usage (from the repository root):
    python condz/plot_length.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = [  # legend name, short label, results folder, colour (validated categorical slots 1-2), marker
    ("Qwen2.5-1.5B", "Qwen", ROOT / "runs" / "qwen", "#2a78d6", "o"),
    ("Llama-3.1-8B", "Llama", ROOT / "runs" / "llama", "#eb6834", "s"),
]
TRAIN = [("full", "-", "trained on full texts"), ("100", "--", "trained on 100 tokens")]
SCORED = ["100", "250", "500", "full"]
X = {"100": 100, "250": 250, "500": 500, "full": 1000}  # "full" drawn at ~1,000 tokens (mean scored length)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for model, short, folder, colour, marker in MODELS:
        res = {(r["train_len"], r["test_len"]): r["auc"]
               for r in json.loads((folder / "window_grid_results_r0.json").read_text())}
        for tr, ls, tr_label in TRAIN:
            y = [res[(tr, s)] for s in SCORED]
            ax.plot([X[s] for s in SCORED], y, ls=ls, color=colour, lw=1.6, marker=marker, ms=4.5,
                    mec="white", mew=0.8, label=f"{model}, {tr_label}")
            ax.annotate(f"{short}, {'full' if tr == 'full' else '100'}  {y[-1]:.3f}",
                        (X["full"], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([X[s] for s in SCORED], ["100", "250", "500", "Full"])
    ax.minorticks_off()
    ax.set_xlim(85, 2400)
    ax.set_ylim(0.6, 0.95)
    ax.set_xlabel("Scored text (tokens)")
    ax.set_ylabel("AUROC")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False, fontsize=6.5,
              handlelength=2.6)
    fig.tight_layout()
    out = ROOT / "results" / "fig_length_auroc"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
