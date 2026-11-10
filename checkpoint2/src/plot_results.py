"""Draw aggregate plots for the Checkpoint 2 demonstration."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results"
POLICIES = {
    "random": ("Random", "#202833", "o", "--"),
    "uncertainty": ("Uncertainty", "#a43d22", "^", "-."),
    "diversity": ("Diversity", "#1c6593", "s", "-"),
}
MODELS = {
    "constant_0.5": "Constant 0.5",
    "metadata_logistic": "Metadata logistic",
    "fusion_temperature": "Calibrated fusion",
}
SOURCES = {
    "matharena": "MathArena",
    "multi_swebench": "Multi-SWE-bench",
    "real_webagents": "REAL web agents",
    "researchcodebench": "ResearchCodeBench",
    "swe_rebench": "SWE-rebench",
}


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    curves = pd.read_csv(OUT / "learning_curves.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.3), sharey=True, layout="constrained")
    for ax, (model, title) in zip(axes, MODELS.items()):
        for policy, (label, color, mark, line) in POLICIES.items():
            data = curves[(curves.model == model) & (curves.policy == policy)].sort_values("budget")
            x = np.arange(len(data))
            ax.plot(x, data.brier, marker=mark, ms=4, ls=line, lw=1.45, color=color, label=label)
        ax.set_xticks(x, data.budget)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Pair-macro Brier score")
    axes[2].legend(frameon=False, fontsize=8)
    fig.supxlabel("Number of revealed labels")
    fig.savefig(OUT / "learning_curves.png", dpi=180)
    plt.close(fig)

    scores = pd.read_csv(OUT / "source_alc.csv")
    scores = scores[scores.model == "fusion_temperature"].pivot(index="source", columns="policy", values="alc")
    scores = scores.reindex(SOURCES.keys())
    fig, ax = plt.subplots(figsize=(6.5, 3.35), layout="constrained")
    for offset, policy in [(-.12, "uncertainty"), (.12, "diversity")]:
        label, color, mark, _ = POLICIES[policy]
        ax.scatter((scores[policy] - scores.random) * 1000, np.arange(len(scores)) + offset,
                   color=color, marker=mark, label=label)
    ax.set_yticks(np.arange(len(scores)), list(SOURCES.values()))
    ax.invert_yaxis()
    ax.axvline(0, color="#202833", ls="--", lw=.9)
    ax.set_xlabel("Brier ALC difference vs. random (×0.001)")
    ax.set_title("Calibrated fusion by held-out source")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", alpha=.25)
    fig.savefig(OUT / "source_deltas.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
