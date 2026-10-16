from datetime import datetime
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import functions as F

from model_evaluation import evaluate, plot_threshold_metrics, save_json
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
