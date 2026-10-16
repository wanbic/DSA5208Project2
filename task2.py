import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from pyspark import StorageLevel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from preprocess import clean_rainfall_data, create_spark, load_rainfall_data
from logistic_model import (
    MAX_ITER,
    NUMERIC_FEATURES,
    SAMPLE_MODULUS,
    THRESHOLDS,
    make_examples,
    predict_probability,
    save_model,
    train_model,
)


PROJECT_PATH = Path(__file__).resolve().parent
DATA_PATH = PROJECT_PATH / "data"
MODEL_ROOT = PROJECT_PATH / "models"


def save_json(data, path):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def evaluate(predictions, name):
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    pr_auc = evaluator.setMetricName("areaUnderPR").evaluate(predictions)
    roc_auc = evaluator.setMetricName("areaUnderROC").evaluate(predictions)

    print(f"{name} PR-AUC: {pr_auc:.4f}")
    print(f"{name} ROC-AUC: {roc_auc:.4f}")

    threshold_results = []
    for threshold in THRESHOLDS:
        subset = predictions.filter(F.col("threshold_mm") == threshold)
        threshold_pr_auc = (
            evaluator.setMetricName("areaUnderPR").evaluate(subset)
        )
        threshold_roc_auc = (
            evaluator.setMetricName("areaUnderROC").evaluate(subset)
        )
        statistics = (
            subset
            .withColumn(
                "probability_of_rain",
                vector_to_array("probability")[1],
            )
            .agg(
                F.count("*").alias("rows"),
                F.avg("label").alias("positive_rate"),
                F.avg(
                    (F.col("probability_of_rain") - F.col("label")) ** 2
                ).alias("brier_score"),
            )
            .first()
        )

        result = {
            "threshold_mm": threshold,
            "rows": statistics["rows"],
            "positive_rate": statistics["positive_rate"],
            "pr_auc": threshold_pr_auc,
            "roc_auc": threshold_roc_auc,
            "brier_score": statistics["brier_score"],
        }
        threshold_results.append(result)
        print(
            f"{name} threshold {threshold:>4.1f} mm: "
            f"PR-AUC={threshold_pr_auc:.4f}, "
            f"ROC-AUC={threshold_roc_auc:.4f}, "
            f"Brier={statistics['brier_score']:.4f}"
        )

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "threshold_results": threshold_results,
    }


def plot_threshold_metrics(validation_results, test_results, output_path):
    thresholds = [row["threshold_mm"] for row in test_results]
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
            [row[key] for row in validation_results],
            marker="o",
            label="Validation 2023",
        )
        axis.plot(
            positions,
            [row[key] for row in test_results],
            marker="o",
            label="Test 2024",
        )
        axis.set_title(title)
        axis.set_xlabel("Rainfall threshold (mm)")
        axis.set_xticks(positions, thresholds)
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Metric value")
    axes[0].legend()
    figure.suptitle("Model performance at different rainfall thresholds")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main():
    """Train and evaluate the logistic regression baseline."""

    spark = create_spark()
    spark.conf.set("spark.sql.session.timeZone", "Asia/Singapore")
    spark.sparkContext.setLogLevel("WARN")

    rainfall = load_rainfall_data(
        spark,
        DATA_PATH,
        years=range(2017, 2025),
    )
    rainfall = clean_rainfall_data(rainfall)

    examples = make_examples(rainfall)
    examples = examples.persist(StorageLevel.DISK_ONLY)

    training_data = examples.filter(
        F.col("observation_time") < F.lit("2023-01-01")
    )
    validation_data = examples.filter(
        (F.col("observation_time") >= F.lit("2023-01-01"))
        & (F.col("observation_time") < F.lit("2024-01-01"))
    )
    test_data = examples.filter(
        F.col("observation_time") >= F.lit("2024-01-01")
    )

    training_rows = training_data.count()
    validation_rows = validation_data.count()
    test_rows = test_data.count()
    print(f"Training rows: {training_rows:,}")
    print(f"Validation rows: {validation_rows:,}")
    print(f"Test rows: {test_rows:,}")

    model = train_model(training_data)
    validation_predictions = model.transform(validation_data).cache()
    test_predictions = model.transform(test_data).cache()

    validation_results = evaluate(
        validation_predictions,
        "Validation",
    )
    test_results = evaluate(test_predictions, "Test")

    run_name = datetime.now().strftime("logistic_run_%Y%m%d_%H%M%S")
    run_path = MODEL_ROOT / run_name
    run_path.mkdir(parents=True)

    save_model(model, run_path / "model.json")

    training_parameters = {
        "algorithm": "Spark LogisticRegression baseline",
        "data_years": list(range(2017, 2025)),
        "training_period": ["2017-01-01", "2022-12-31"],
        "validation_period": ["2023-01-01", "2023-12-31"],
        "test_period": ["2024-01-01", "2024-12-31"],
        "thresholds_mm": THRESHOLDS,
        "sample_modulus": SAMPLE_MODULUS,
        "prediction_minutes": [0, 30],
        "features": ["station_id", *NUMERIC_FEATURES],
        "max_iter": MAX_ITER,
    }
    save_json(
        training_parameters,
        run_path / "training_parameters.json",
    )
    plot_threshold_metrics(
        validation_results["threshold_results"],
        test_results["threshold_results"],
        run_path / "threshold_metrics.png",
    )

    example = (
        test_data
        .filter(F.col("threshold_mm") == 1.0)
        .select(
            "station_id",
            "observation_time",
            "current_rainfall_mm",
            "recent_15min_rainfall_mm",
            "recent_30min_rainfall_mm",
        )
        .first()
    )
    prediction_time = example["observation_time"].isoformat(sep=" ")
    probability = predict_probability(
        model,
        spark,
        station_id=example["station_id"],
        prediction_time=prediction_time,
        threshold_mm=1.0,
        current_rainfall_mm=example["current_rainfall_mm"],
        recent_15min_rainfall_mm=example["recent_15min_rainfall_mm"],
        recent_30min_rainfall_mm=example["recent_30min_rainfall_mm"],
    )
    print(
        f"P(next 30 min > 1.0 mm | {example['station_id']}, "
        f"{prediction_time}) = {probability:.4f}"
    )

    saved_results = {
        "training_rows": training_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "validation": validation_results,
        "test": test_results,
        "example_prediction": {
            "station_id": example["station_id"],
            "prediction_time": prediction_time,
            "threshold_mm": 1.0,
            "current_rainfall_mm": example["current_rainfall_mm"],
            "recent_15min_rainfall_mm": (
                example["recent_15min_rainfall_mm"]
            ),
            "recent_30min_rainfall_mm": (
                example["recent_30min_rainfall_mm"]
            ),
            "probability": probability,
        },
    }
    save_json(saved_results, run_path / "test_results.json")
    print(f"Saved training run to {run_path}")

    validation_predictions.unpersist(blocking=True)
    test_predictions.unpersist(blocking=True)
    examples.unpersist(blocking=True)
    spark.stop()


if __name__ == "__main__":
    main()
