"""Run rainfall-history feature ablations for the final GBT model."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from pyspark import StorageLevel
from pyspark.sql import functions as F

from history_ablation_model import (
    ABLATION_CONFIGS,
    build_neighbor_pairs,
    make_examples,
    save_model,
    save_neighbor_mapping,
    train_model,
)
from logistic_model import THRESHOLDS as DEFAULT_THRESHOLDS
from model_evaluation import evaluate, plot_threshold_metrics, save_json
from plot_history_ablation_comparison import (
    load_results as load_threshold_results,
    plot_comparison as plot_threshold_comparison,
)
from preprocess import clean_rainfall_data, create_spark, load_rainfall_data


PROJECT_PATH = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_PATH / "data"
DEFAULT_MODEL_ROOT = PROJECT_PATH / "models"
DEFAULT_SPARK_TEMP_PATH = PROJECT_PATH / "spark_tmp"
DEFAULT_FULL_MODEL_PATH = (
    DEFAULT_MODEL_ROOT / "final_gbt_run_20261017_184044"
)


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(
        description="Run GBT rainfall-history ablation experiments."
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(ABLATION_CONFIGS),
        default=list(ABLATION_CONFIGS),
        help="Ablation variants to train; defaults to both variants.",
    )
    parser.add_argument("--sample-modulus", type=int, default=4)
    parser.add_argument("--training-sample-modulus", type=int, default=16)
    parser.add_argument("--neighbor-count", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--subsampling-rate", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=5208)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument("--spark-master", default="local[4]")
    parser.add_argument("--driver-memory", default="4g")
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
    )
    parser.add_argument(
        "--spark-temp-path",
        type=Path,
        default=DEFAULT_SPARK_TEMP_PATH,
    )
    parser.add_argument(
        "--full-model-run",
        type=Path,
        default=DEFAULT_FULL_MODEL_PATH,
        help="Existing full model run used as the comparison reference.",
    )
    parser.add_argument("--run-name")

    args = parser.parse_args(arguments)
    if args.sample_modulus < 1:
        parser.error("--sample-modulus must be at least 1")
    if args.training_sample_modulus < 1:
        parser.error("--training-sample-modulus must be at least 1")
    if args.training_sample_modulus % args.sample_modulus != 0:
        parser.error(
            "--training-sample-modulus must be a multiple of "
            "--sample-modulus"
        )
    if args.neighbor_count < 1:
        parser.error("--neighbor-count must be at least 1")
    if args.max_iter < 1 or args.max_depth < 1:
        parser.error("--max-iter and --max-depth must be at least 1")
    if args.step_size <= 0:
        parser.error("--step-size must be positive")
    if not 0 < args.subsampling_rate <= 1:
        parser.error("--subsampling-rate must be in (0, 1]")
    if args.shuffle_partitions < 1:
        parser.error("--shuffle-partitions must be at least 1")
    if any(value < 0 for value in args.thresholds):
        parser.error("--thresholds cannot contain negative values")
    if len(set(args.thresholds)) != len(args.thresholds):
        parser.error("--thresholds cannot contain duplicates")
    return args


def mean_brier(results):
    scores = [row["brier_score"] for row in results["threshold_results"]]
    return sum(scores) / len(scores)


def make_summary(variant, label, features, results):
    return {
        "variant": variant,
        "label": label,
        "features": ["station_id", *features],
        "validation_pr_auc": results["validation"]["pr_auc"],
        "validation_roc_auc": results["validation"]["roc_auc"],
        "validation_mean_brier": mean_brier(results["validation"]),
        "test_pr_auc": results["test"]["pr_auc"],
        "test_roc_auc": results["test"]["roc_auc"],
        "test_mean_brier": mean_brier(results["test"]),
    }


def load_full_model_summary(path, thresholds):
    results_path = path / "test_results.json"
    parameters_path = path / "training_parameters.json"
    if not results_path.exists() or not parameters_path.exists():
        print(f"Full-model reference not found at {path}; skipping it")
        return None

    results = json.loads(results_path.read_text(encoding="utf-8"))
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    saved_thresholds = parameters.get("thresholds_mm")
    if saved_thresholds != thresholds:
        print(
            "Full-model thresholds do not match this run; "
            "skipping the reference"
        )
        return None

    features = parameters.get("features", [])
    return make_summary(
        "full_history",
        "Full 15- and 30-minute history",
        [feature for feature in features if feature != "station_id"],
        results,
    )


def plot_comparison(summary, output_path):
    """Plot validation and test metrics for every available configuration."""

    labels = [row["label"] for row in summary]
    positions = list(range(len(labels)))
    panels = [
        ("validation_pr_auc", "test_pr_auc", "PR-AUC", "Higher is better"),
        (
            "validation_roc_auc",
            "test_roc_auc",
            "ROC-AUC",
            "Higher is better",
        ),
        (
            "validation_mean_brier",
            "test_mean_brier",
            "Mean Brier score",
            "Lower is better",
        ),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, (validation_key, test_key, title, direction) in zip(
        axes,
        panels,
    ):
        axis.plot(
            positions,
            [row[validation_key] for row in summary],
            marker="o",
            label="Validation 2023",
        )
        axis.plot(
            positions,
            [row[test_key] for row in summary],
            marker="s",
            label="Test 2024",
        )
        axis.set_title(f"{title}\n({direction})")
        axis.set_xticks(positions, labels, rotation=18, ha="right")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Metric value")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    figure.suptitle("Rainfall-history feature ablation", y=1.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main(arguments=None):
    args = parse_args(arguments)
    run_name = args.run_name or datetime.now().strftime(
        "history_ablation_run_%Y%m%d_%H%M%S"
    )
    run_path = args.model_root / run_name
    run_path.mkdir(parents=True, exist_ok=False)

    args.spark_temp_path.mkdir(parents=True, exist_ok=True)
    spark = create_spark(
        local_directory=args.spark_temp_path,
        master=args.spark_master,
        driver_memory=args.driver_memory,
        shuffle_partitions=args.shuffle_partitions,
    )
    spark.conf.set("spark.sql.session.timeZone", "Asia/Singapore")
    spark.sparkContext.setLogLevel("WARN")

    rainfall = load_rainfall_data(
        spark,
        args.data_path,
        years=range(2017, 2025),
    )
    rainfall = clean_rainfall_data(rainfall)
    neighbor_pairs = build_neighbor_pairs(rainfall, args.neighbor_count)
    print(
        f"Built {len(neighbor_pairs):,} station-neighbor pairs "
        f"using {args.neighbor_count} neighbors per station"
    )
    save_neighbor_mapping(neighbor_pairs, run_path / "neighbor_mapping.json")

    examples = make_examples(
        rainfall,
        neighbor_pairs,
        sample_modulus=args.sample_modulus,
        thresholds=args.thresholds,
    )
    training_data = (
        examples
        .filter(F.col("observation_time") < F.lit("2023-01-01"))
        .filter(
            F.pmod(
                F.xxhash64("station_id", "observation_time"),
                F.lit(args.training_sample_modulus),
            ) == 0
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    validation_data = (
        examples
        .filter(
            (F.col("observation_time") >= F.lit("2023-01-01"))
            & (F.col("observation_time") < F.lit("2024-01-01"))
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    test_data = (
        examples
        .filter(F.col("observation_time") >= F.lit("2024-01-01"))
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    training_rows = training_data.count()
    validation_rows = validation_data.count()
    test_rows = test_data.count()
    print(f"Training rows: {training_rows:,}")
    print(f"Validation rows: {validation_rows:,}")
    print(f"Test rows: {test_rows:,}")

    all_results = {}
    summary = []
    for variant in args.variants:
        configuration = ABLATION_CONFIGS[variant]
        features = configuration["features"]
        variant_path = run_path / variant
        variant_path.mkdir()
        print(f"\nTraining {variant}: {configuration['label']}")
        print(f"Features: {features}")

        model = train_model(
            training_data,
            numeric_features=features,
            max_iter=args.max_iter,
            max_depth=args.max_depth,
            step_size=args.step_size,
            subsampling_rate=args.subsampling_rate,
            seed=args.seed,
        )

        validation_predictions = model.transform(validation_data).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        validation_predictions.count()
        validation_results = evaluate(
            validation_predictions,
            f"{variant} validation",
            args.thresholds,
        )
        validation_predictions.unpersist(blocking=True)

        test_predictions = model.transform(test_data).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        test_predictions.count()
        test_results = evaluate(
            test_predictions,
            f"{variant} test",
            args.thresholds,
        )
        test_predictions.unpersist(blocking=True)

        variant_results = {
            "training_rows": training_rows,
            "validation_rows": validation_rows,
            "test_rows": test_rows,
            "validation": validation_results,
            "test": test_results,
        }
        all_results[variant] = variant_results
        summary.append(
            make_summary(
                variant,
                configuration["label"],
                features,
                variant_results,
            )
        )

        save_model(
            model,
            variant_path / "model.json",
            thresholds=args.thresholds,
            neighbor_count=args.neighbor_count,
            numeric_features=features,
        )
        save_json(
            {
                "algorithm": "Spark GBTClassifier history ablation",
                "variant": variant,
                "description": configuration["label"],
                "data_years": list(range(2017, 2025)),
                "training_period": ["2017-01-01", "2022-12-31"],
                "validation_period": ["2023-01-01", "2023-12-31"],
                "test_period": ["2024-01-01", "2024-12-31"],
                "thresholds_mm": args.thresholds,
                "sample_modulus": args.sample_modulus,
                "training_sample_modulus": args.training_sample_modulus,
                "prediction_minutes": [0, 30],
                "features": ["station_id", *features],
                "neighbor_count": args.neighbor_count,
                "max_iter": args.max_iter,
                "max_depth": args.max_depth,
                "step_size": args.step_size,
                "subsampling_rate": args.subsampling_rate,
                "seed": args.seed,
                "spark_master": args.spark_master,
                "driver_memory": args.driver_memory,
                "shuffle_partitions": args.shuffle_partitions,
            },
            variant_path / "training_parameters.json",
        )
        save_json(variant_results, variant_path / "test_results.json")
        plot_threshold_metrics(
            validation_results["threshold_results"],
            test_results["threshold_results"],
            variant_path / "threshold_metrics.png",
        )

    full_summary = load_full_model_summary(
        args.full_model_run,
        args.thresholds,
    )
    comparison_summary = []
    if full_summary is not None:
        comparison_summary.append(full_summary)
    comparison_summary.extend(summary)

    save_json(
        {
            "full_history_source": (
                str(args.full_model_run) if full_summary is not None else None
            ),
            "summary": comparison_summary,
            "results": all_results,
        },
        run_path / "comparison_results.json",
    )
    plot_comparison(
        comparison_summary,
        run_path / "history_ablation.png",
    )
    threshold_values, threshold_results = load_threshold_results(
        run_path,
        "test",
    )
    plot_threshold_comparison(
        threshold_values,
        threshold_results,
        "test",
        run_path / "history_ablation_threshold_comparison_test.png",
    )

    training_data.unpersist(blocking=True)
    validation_data.unpersist(blocking=True)
    test_data.unpersist(blocking=True)
    spark.stop()
    print(f"Saved history ablation run to {run_path}")


if __name__ == "__main__":
    main()
