"""Static scientific figure from measured curves; no synthetic model outputs."""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/20261114T035022Z-predicate-retention-dev-v1-032f30"


def main():
    result = json.loads((RUN / "result.json").read_text(encoding="utf-8"))
    assert not result["failures"] and len(result["cells"]) == 12
    arms = ("frozen", "readout_only", "current_hidden", "protected_hidden")
    labels = ("Frozen", "Readout only", "Current hidden", "Protected hidden")
    colors = ("#7b8794", "#397ac6", "#d18532", "#208a7a")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": (1.2, 1)})
    fig.patch.set_facecolor("white")
    for index, (arm, title, color) in enumerate(zip(arms, labels, colors)):
        cells = [c for c in result["cells"] if c["arm"] == arm]
        curve = np.mean([[s["accuracy"] for s in c["curves"]] for c in cells], axis=0) * 100
        axes[0].plot(range(1, 21), curve, label=title, color=color, lw=1.9)
        values = np.array([c["final_DEV"]["accuracy"] for c in cells]) * 100
        axes[1].scatter(index + np.linspace(-.09, .09, 3), values, color=color,
                        s=45, zorder=3, edgecolors="white", linewidths=.8)
        axes[1].plot([index-.2, index+.2], [values.mean()]*2, color=color, lw=3)
        axes[1].text(index, 61, f"{values.mean():.1f}%", ha="center", fontsize=10, color=color)
    for ax in axes:
        ax.axhline(50, ls=":", lw=1.1, color="#9aa3ad")
        ax.axhline(65, ls="--", lw=1.2, color="#b65353")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(35, 72)
    axes[0].set_xlim(1, 20)
    axes[0].set_xticks((1, 4, 8, 12, 16, 20))
    axes[0].set_xlabel("Stage (no boundary signal to learner)")
    axes[0].set_title("Mean unseen-probe score (zoomed)")
    axes[0].legend(loc="lower left", fontsize=9, ncol=2, frameon=False)
    axes[1].set_ylim(0, 100)
    axes[1].set_xticks(range(4), labels, rotation=15)
    axes[1].set_title("Final DEV: dots = three neural seeds")
    axes[1].text(3.52, 65, "65% floor", ha="right", va="bottom", fontsize=8, color="#b65353")
    fig.suptitle("Fresh query-predicate DEV · ALIF H128 · all joint gates failed", fontsize=14)
    fig.text(.5, .025,
             "320 new pairs + replay · 400 source-heldout DEV cases (2 encoded aliases) · no language/video claim\n"
             "Protection − current: +2.5 pp; paired seed/family 95% CI [−0.75, +6.33] pp. Not qualified.",
             ha="center", fontsize=9, color="#4b5563")
    fig.tight_layout(rect=(0, .13, 1, .93))
    destination = RUN / "predicate-retention-screen-v2.png"
    if destination.exists():
        raise FileExistsError("Refuse replacing a historical figure")
    fig.savefig(destination, dpi=180, facecolor="white")
    fig.savefig(RUN / "predicate-retention-screen-v2.svg", facecolor="white")
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
