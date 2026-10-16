import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_PATH = Path(__file__).resolve().parent
RUNS = [
    (
        "Logistic (1/4)",
        PROJECT_PATH / "models" / "logistic_run_20261015_200027",
        "#4C78A8",
        "o",
        1.8,
    ),
    (
        "GBT (1/16, selected)",
        PROJECT_PATH / "models" / "gbt_run_20261016_163426",
        "#54A24B",
        "^",
        2.6,
    ),
    (
        "GBT (1/64)",
        PROJECT_PATH / "models" / "gbt_run_20261015_213401",
        "#F58518",
        "s",
        1.8,
    ),
]


def main():
    """Compare validation metrics from the three preliminary runs."""

    results = []
    for label, run_path, colour, marker, width in RUNS:
        data = json.loads(
            (run_path / "test_results.json").read_text(encoding="utf-8")
        )
        results.append(
            (
                label,
                data["validation"]["threshold_results"],
                colour,
                marker,
                width,
            )
        )

    thresholds = [row["threshold_mm"] for row in results[0][1]]
    positions = list(range(len(thresholds)))
    metrics = [
        ("pr_auc", "PR-AUC", "Higher is better"),
        ("roc_auc", "ROC-AUC", "Higher is better"),
        ("brier_score", "Brier score", "Lower is better"),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for axis, (key, title, direction) in zip(axes, metrics):
        for label, rows, colour, marker, width in results:
            axis.plot(
                positions,
                [row[key] for row in rows],
                color=colour,
                marker=marker,
                linewidth=width,
                markersize=5.5,
                label=label,
            )
        axis.set_title(f"{title}\n({direction})")
        axis.set_xlabel("Rainfall threshold (mm)")
        axis.set_xticks(positions, [str(value) for value in thresholds])
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Preliminary model comparison on the 2023 validation set",
        y=1.13,
        fontsize=13,
    )
    figure.tight_layout()

    output_path = PROJECT_PATH / "figures" / "task2_model_comparison.png"
    output_path.parent.mkdir(exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
