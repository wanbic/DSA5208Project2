import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from pyspark import StorageLevel
from pyspark.sql import functions as F

from logistic_model import THRESHOLDS as DEFAULT_THRESHOLDS
from model_evaluation import evaluate, plot_threshold_metrics, save_json
from preprocess import clean_rainfall_data, create_spark, load_rainfall_data
from rainfall_model import (
    NUMERIC_FEATURES,
    build_neighbor_pairs,
    make_examples,
    predict_probability,
    save_model,
    save_neighbor_mapping,
    train_model,
)


PROJECT_PATH = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_PATH / "data"
DEFAULT_MODEL_ROOT = PROJECT_PATH / "models"
DEFAULT_SPARK_TEMP_PATH = PROJECT_PATH / "spark_tmp"
DEFAULT_BASELINE_PATH = (
    DEFAULT_MODEL_ROOT / "gbt_run_20261016_163426"
)


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(
        description="Train the final rainfall GBT model."
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
        "--baseline-run",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
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


def compare_with_baseline(
    validation_results,
    test_results,
    baseline_path,
):
    """Compare the final model with a previous run."""

    baseline_file = baseline_path / "test_results.json"
    if not baseline_file.exists():
        return None

    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    baseline_thresholds = [
        row["threshold_mm"]
        for row in baseline["test"]["threshold_results"]
    ]
    current_thresholds = [
        row["threshold_mm"] for row in test_results["threshold_results"]
    ]
    if current_thresholds != baseline_thresholds:
        return None

    return {
        "baseline_run": baseline_path.name,
        "validation_pr_auc_change": (
            validation_results["pr_auc"] - baseline["validation"]["pr_auc"]
        ),
        "validation_roc_auc_change": (
            validation_results["roc_auc"]
            - baseline["validation"]["roc_auc"]
        ),
        "validation_mean_brier_change": (
            mean_brier(validation_results)
            - mean_brier(baseline["validation"])
        ),
        "test_pr_auc_change": (
            test_results["pr_auc"] - baseline["test"]["pr_auc"]
        ),
        "test_roc_auc_change": (
            test_results["roc_auc"] - baseline["test"]["roc_auc"]
        ),
        "test_mean_brier_change": (
            mean_brier(test_results) - mean_brier(baseline["test"])
        ),
    }


def plot_baseline_comparison(test_results, baseline_path, output_path):
    """Plot test metrics for the baseline and final models."""

    baseline_file = baseline_path / "test_results.json"
    if not baseline_file.exists():
        return

    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    baseline_by_threshold = {
        row["threshold_mm"]: row
        for row in baseline["test"]["threshold_results"]
    }
    final_by_threshold = {
        row["threshold_mm"]: row
        for row in test_results["threshold_results"]
    }
    thresholds = [
        threshold
        for threshold in final_by_threshold
        if threshold in baseline_by_threshold
    ]
    if not thresholds:
        return

    baseline_rows = [baseline_by_threshold[value] for value in thresholds]
    final_rows = [final_by_threshold[value] for value in thresholds]
    positions = list(range(len(thresholds)))
    metrics = [
        ("pr_auc", "PR-AUC"),
        ("roc_auc", "ROC-AUC"),
        ("brier_score", "Brier score"),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, (key, title) in zip(axes, metrics):
        axis.plot(
            positions,
            [row[key] for row in baseline_rows],
            marker="o",
            label="GBT baseline",
        )
        axis.plot(
            positions,
            [row[key] for row in final_rows],
            marker="s",
            label="Final GBT",
        )
        axis.set_title(title)
        axis.set_xlabel("Rainfall threshold (mm)")
        axis.set_xticks(positions, thresholds)
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Metric value")
    axes[0].legend()
    figure.suptitle("Final model comparison on the 2024 test set")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main(arguments=None):
    """Train and evaluate the final neighbor-rainfall GBT model."""

    args = parse_args(arguments)
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
    examples = make_examples(
        rainfall,
        neighbor_pairs,
        sample_modulus=args.sample_modulus,
        thresholds=args.thresholds,
    )

    training_data = examples.filter(
        F.col("observation_time") < F.lit("2023-01-01")
    ).filter(
        F.pmod(
            F.xxhash64("station_id", "observation_time"),
            F.lit(args.training_sample_modulus),
        ) == 0
    ).persist(StorageLevel.MEMORY_AND_DISK)
    validation_data = examples.filter(
        (F.col("observation_time") >= F.lit("2023-01-01"))
        & (F.col("observation_time") < F.lit("2024-01-01"))
    )
    test_data = examples.filter(
        F.col("observation_time") >= F.lit("2024-01-01")
    )

    training_rows = training_data.count()
    print(f"Training rows: {training_rows:,}")
    model = train_model(
        training_data,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        step_size=args.step_size,
        subsampling_rate=args.subsampling_rate,
        seed=args.seed,
    )
    training_data.unpersist(blocking=True)

    validation_predictions = model.transform(validation_data).persist(
        StorageLevel.MEMORY_AND_DISK
    )
    validation_rows = validation_predictions.count()
    print(f"Validation rows: {validation_rows:,}")
    validation_results = evaluate(
        validation_predictions,
        "Validation",
        args.thresholds,
    )
    validation_predictions.unpersist(blocking=True)

    test_predictions = model.transform(test_data).persist(
        StorageLevel.MEMORY_AND_DISK
    )
    test_rows = test_predictions.count()
    print(f"Test rows: {test_rows:,}")
    test_results = evaluate(test_predictions, "Test", args.thresholds)

    example_threshold = min(args.thresholds, key=lambda value: abs(value - 1.0))
    example = (
        test_predictions
        .filter(F.col("threshold_mm") == example_threshold)
        .select(
            "station_id",
            "observation_time",
            "current_rainfall_mm",
            "recent_15min_rainfall_mm",
            "recent_30min_rainfall_mm",
            "neighbor_rainfall_mm",
        )
        .first()
    )
    test_predictions.unpersist(blocking=True)

    run_name = args.run_name or datetime.now().strftime(
        "final_gbt_run_%Y%m%d_%H%M%S"
    )
    run_path = args.model_root / run_name
    run_path.mkdir(parents=True, exist_ok=False)

    save_model(
        model,
        run_path / "model.json",
        thresholds=args.thresholds,
        neighbor_count=args.neighbor_count,
    )
    save_neighbor_mapping(neighbor_pairs, run_path / "neighbor_mapping.json")
    save_json(
        {
            "algorithm": "Spark GBTClassifier with neighbor rainfall",
            "data_years": list(range(2017, 2025)),
            "training_period": ["2017-01-01", "2022-12-31"],
            "validation_period": ["2023-01-01", "2023-12-31"],
            "test_period": ["2024-01-01", "2024-12-31"],
            "thresholds_mm": args.thresholds,
            "sample_modulus": args.sample_modulus,
            "training_sample_modulus": args.training_sample_modulus,
            "prediction_minutes": [0, 30],
            "features": ["station_id", *NUMERIC_FEATURES],
            "neighbor_count": args.neighbor_count,
            "neighbor_feature": (
                "mean current rainfall of the nearest stations"
            ),
            "max_iter": args.max_iter,
            "max_depth": args.max_depth,
            "step_size": args.step_size,
            "subsampling_rate": args.subsampling_rate,
            "seed": args.seed,
            "spark_master": args.spark_master,
            "driver_memory": args.driver_memory,
            "shuffle_partitions": args.shuffle_partitions,
        },
        run_path / "training_parameters.json",
    )
    plot_threshold_metrics(
        validation_results["threshold_results"],
        test_results["threshold_results"],
        run_path / "threshold_metrics.png",
    )
    plot_baseline_comparison(
        test_results,
        args.baseline_run,
        run_path / "baseline_comparison.png",
    )

    prediction_time = example["observation_time"].isoformat(sep=" ")
    probability = predict_probability(
        model,
        spark,
        station_id=example["station_id"],
        prediction_time=prediction_time,
        threshold_mm=example_threshold,
        current_rainfall_mm=example["current_rainfall_mm"],
        recent_15min_rainfall_mm=example["recent_15min_rainfall_mm"],
        recent_30min_rainfall_mm=example["recent_30min_rainfall_mm"],
        neighbor_rainfall_mm=example["neighbor_rainfall_mm"],
    )

    comparison = compare_with_baseline(
        validation_results,
        test_results,
        args.baseline_run,
    )
    saved_results = {
        "training_rows": training_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "validation": validation_results,
        "test": test_results,
        "baseline_comparison": comparison,
        "example_prediction": {
            "station_id": example["station_id"],
            "prediction_time": prediction_time,
            "threshold_mm": example_threshold,
            "current_rainfall_mm": example["current_rainfall_mm"],
            "recent_15min_rainfall_mm": (
                example["recent_15min_rainfall_mm"]
            ),
            "recent_30min_rainfall_mm": (
                example["recent_30min_rainfall_mm"]
            ),
            "neighbor_rainfall_mm": example["neighbor_rainfall_mm"],
            "probability": probability,
        },
    }
    save_json(saved_results, run_path / "test_results.json")
    if comparison is not None:
        print(json.dumps(comparison, indent=2))
    print(f"Saved training run to {run_path}")

    spark.stop()


if __name__ == "__main__":
    main()
