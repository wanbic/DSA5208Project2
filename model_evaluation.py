import json

import matplotlib.pyplot as plt
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from logistic_model import THRESHOLDS


def save_json(data, path):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def evaluate(predictions, name, thresholds=THRESHOLDS):
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    pr_auc = evaluator.setMetricName("areaUnderPR").evaluate(predictions)
    roc_auc = evaluator.setMetricName("areaUnderROC").evaluate(predictions)

    print(f"{name} PR-AUC: {pr_auc:.4f}")
    print(f"{name} ROC-AUC: {roc_auc:.4f}")

    threshold_results = []
    for threshold in thresholds:
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
