from datetime import datetime
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import functions as F

from utils.gbt_baseline_model import (
    MAX_DEPTH,
    MAX_ITER,
    SAMPLE_MODULUS,
    SEED,
    STEP_SIZE,
    SUBSAMPLING_RATE,
    TRAIN_SAMPLE_MODULUS,
    make_examples,
    predict_probability,
    save_model,
    train_model,
)
from utils.logistic_model import NUMERIC_FEATURES, THRESHOLDS
from utils.preprocess import clean_rainfall_data, create_spark, load_rainfall_data
from utils.model_evaluation import evaluate, plot_threshold_metrics, save_json


PROJECT_PATH = Path(__file__).resolve().parent
DATA_PATH = PROJECT_PATH / "data"
MODEL_ROOT = PROJECT_PATH / "models"
SPARK_TEMP_PATH = PROJECT_PATH / "spark_tmp"


def main():
    """Train and evaluate the GBT model."""

    SPARK_TEMP_PATH.mkdir(exist_ok=True)
    spark = create_spark(SPARK_TEMP_PATH)
    spark.conf.set("spark.sql.session.timeZone", "Asia/Singapore")
    spark.sparkContext.setLogLevel("WARN")

    rainfall = load_rainfall_data(
        spark,
        DATA_PATH,
        years=range(2017, 2025),
    )
    rainfall = clean_rainfall_data(rainfall)

    examples = make_examples(rainfall)

    training_data = examples.filter(
        F.col("observation_time") < F.lit("2023-01-01")
    ).filter(
        F.pmod(
            F.xxhash64("station_id", "observation_time"),
            F.lit(TRAIN_SAMPLE_MODULUS),
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

    model = train_model(training_data)
    training_data.unpersist(blocking=True)

    validation_predictions = model.transform(validation_data).persist(
        StorageLevel.MEMORY_AND_DISK
    )
    validation_rows = validation_predictions.count()
    print(f"Validation rows: {validation_rows:,}")

    validation_results = evaluate(
        validation_predictions,
        "Validation",
    )
    validation_predictions.unpersist(blocking=True)

    test_predictions = model.transform(test_data).persist(
        StorageLevel.MEMORY_AND_DISK
    )
    test_rows = test_predictions.count()
    print(f"Test rows: {test_rows:,}")
    test_results = evaluate(test_predictions, "Test")

    example = (
        test_predictions
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
    test_predictions.unpersist(blocking=True)

    run_name = datetime.now().strftime("gbt_run_%Y%m%d_%H%M%S")
    run_path = MODEL_ROOT / run_name
    run_path.mkdir(parents=True)

    save_model(model, run_path / "model.json")

    training_parameters = {
        "algorithm": "Spark GBTClassifier",
        "data_years": list(range(2017, 2025)),
        "training_period": ["2017-01-01", "2022-12-31"],
        "validation_period": ["2023-01-01", "2023-12-31"],
        "test_period": ["2024-01-01", "2024-12-31"],
        "thresholds_mm": THRESHOLDS,
        "sample_modulus": SAMPLE_MODULUS,
        "training_sample_modulus": TRAIN_SAMPLE_MODULUS,
        "prediction_minutes": [0, 30],
        "features": ["station_id", *NUMERIC_FEATURES],
        "max_iter": MAX_ITER,
        "max_depth": MAX_DEPTH,
        "step_size": STEP_SIZE,
        "subsampling_rate": SUBSAMPLING_RATE,
        "seed": SEED,
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

    spark.stop()


if __name__ == "__main__":
    main()
