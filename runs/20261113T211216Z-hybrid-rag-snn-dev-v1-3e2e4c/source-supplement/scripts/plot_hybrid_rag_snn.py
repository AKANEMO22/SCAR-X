"""Standalone scientific figure from audited predictions, no model execution."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(run, output=None):
    audit = json.loads((run / "independent-audit.json").read_text(encoding="utf-8"))
    keys = [
        "bm25",
        "source_rule",
        "episodic",
        "frozen",
        "readout_only",
        "hidden",
        "hidden_reset",
    ]
    labels = [
        "BM25 RAG",
        "RAG + source/time rules",
        "Episodic kNN",
        "SNN frozen",
        "SNN readout only",
        "SNN hidden learning",
        "Hidden reset ablation",
    ]
    scores = [100 * audit["metrics"][k]["grounded_QA"] for k in keys]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#f7f9fc")
    ax.barh(
        range(len(keys)),
        scores,
        color=[
            "#92a4b5",
            "#286b57",
            "#92a4b5",
            "#c2cad2",
            "#547db0",
            "#b76546",
            "#bba17d",
        ],
        height=0.62,
    )
    for i, key in enumerate(keys):
        vals = [
            100 * r["grounded_QA"] for r in audit["metrics"][key]["per_seed"].values()
        ]
        ax.scatter(vals, [i - 0.12, i, i + 0.12], color="#102538", s=24, zorder=3)
        ax.text(105, i, f"{scores[i]:.1f}%", va="center", fontsize=10)
    ax.set_yticks(range(len(keys)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Grounded exact QA (%)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    fig.suptitle(
        "Current SCAR-X SNN has no demonstrated advantage over the strongest RAG control",
        fontsize=12,
        y=0.97,
    )
    fig.text(
        0.02,
        0.035,
        "48 procedural event questions × 3 neural seeds. Dots: seed scores. Shared frozen 0.8B VLM responses.\nSource/time rule reaches a ceiling on this finite contract; this is not a general intelligence benchmark.\nParent run timed out at 900 s; fixed missing inference completed separately. All failures retained.",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.14, 1, 0.93])
    path = output or run / "grounded-qa-comparison.png"
    if path.exists():
        raise ValueError("Refuse overwrite scientific figure")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    main(a.run, a.out)
