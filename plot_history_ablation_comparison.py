"""Plot threshold-level metrics for the rainfall-history ablation run."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


VARIANTS = [
    (
        "full_history",
        "Full model (15 + 30 min)",
        "#4C78A8",
        "o",
    ),
    (
        "remove_30min",
        "Remove 30 min",
        "#F58518",
        "s",
    ),
    (
        "remove_15min_30min",
        "Remove 15 + 30 min",
        "#E45756",
        "^",
    ),
]


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare threshold-level PR-AUC, ROC-AUC, and Brier score "
            "for a history-ablation run."
        )
    )
    parser.add_argument("run_path", type=Path)
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(arguments)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(run_path, split):
    comparison_path = run_path / "comparison_results.json"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Missing {comparison_path}")

    comparison = read_json(comparison_path)
    full_history_source = comparison.get("full_history_source")
    if not full_history_source:
        raise ValueError(
            "comparison_results.json does not contain full_history_source"
        )

    full_results_path = Path(full_history_source) / "test_results.json"
    if not full_results_path.exists():
        portable_path = (
            run_path.parent
            / Path(full_history_source).name
            / "test_results.json"
        )
        if portable_path.exists():
            full_results_path = portable_path
        else:
            raise FileNotFoundError(f"Missing {full_results_path}")

    results = {
        "full_history": read_json(full_results_path)[split][
            "threshold_results"
        ]
    }
    for variant in ("remove_30min", "remove_15min_30min"):
        results_path = run_path / variant / "test_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Missing {results_path}")
        results[variant] = read_json(results_path)[split][
            "threshold_results"
        ]

    reference_thresholds = [
        row["threshold_mm"] for row in results["full_history"]
    ]
    for variant, rows in results.items():
        thresholds = [row["threshold_mm"] for row in rows]
        if thresholds != reference_thresholds:
            raise ValueError(
                f"Thresholds for {variant} do not match the full model"
            )

    return reference_thresholds, results


def plot_comparison(thresholds, results, split, output_path):
    positions = list(range(len(thresholds)))
    metrics = [
        ("pr_auc", "PR-AUC", "Higher is better"),
        ("roc_auc", "ROC-AUC", "Higher is better"),
        ("brier_score", "Brier score", "Lower is better"),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for axis, (metric, title, direction) in zip(axes, metrics):
        for variant, label, colour, marker in VARIANTS:
            axis.plot(
                positions,
                [row[metric] for row in results[variant]],
                color=colour,
                marker=marker,
                linewidth=2.2,
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
    split_title = "2024 test" if split == "test" else "2023 validation"
    figure.suptitle(
        f"Rainfall-history ablation by threshold ({split_title} set)",
        y=1.12,
        fontsize=13,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main(arguments=None):
    args = parse_args(arguments)
    run_path = args.run_path.resolve()
    output_path = args.output
    if output_path is None:
        output_path = run_path / (
            f"history_ablation_threshold_comparison_{args.split}.png"
        )
    else:
        output_path = output_path.resolve()

    thresholds, results = load_results(run_path, args.split)
    plot_comparison(
        thresholds,
        results,
        args.split,
        output_path,
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
