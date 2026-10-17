import json
import math
from datetime import datetime
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import functions as F
from pyspark.sql.window import Window


THRESHOLDS = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
SAMPLE_MODULUS = 4
MAX_ITER = 20
NUMERIC_FEATURES = [
    "current_rainfall_mm",
    "recent_15min_rainfall_mm",
    "recent_30min_rainfall_mm",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "threshold_mm",
]


def make_examples(
    rainfall,
    sample_modulus=SAMPLE_MODULUS,
    thresholds=THRESHOLDS,
):
    """Create examples for predicting the next 30 minutes."""

    window = Window.partitionBy("station_id").orderBy("observation_time")
    data = rainfall.select(
        "station_id",
        "observation_time",
        F.col("reading_value").alias("current_rainfall_mm"),
    )

    current_time = F.col("observation_time").cast("long")
    recent_15min = F.col("current_rainfall_mm")
    recent_30min = F.col("current_rainfall_mm")
    complete_history = F.lit(True)

    for offset in range(1, 6):
        rain_column = f"past_rain_{offset}"
        time_column = f"past_time_{offset}"

        data = data.withColumn(
            rain_column,
            F.lag("current_rainfall_mm", offset).over(window),
        )
        data = data.withColumn(
            time_column,
            F.lag("observation_time", offset).over(window),
        )

        recent_30min = recent_30min + F.col(rain_column)
        if offset <= 2:
            recent_15min = recent_15min + F.col(rain_column)

        complete_history = complete_history & (
            F.col(time_column).cast("long") == current_time - offset * 300
        )

    future_rainfall = F.lit(0.0)
    complete_future = F.lit(True)

    for offset in range(1, 7):
        rain_column = f"future_rain_{offset}"
        time_column = f"future_time_{offset}"

        data = data.withColumn(
            rain_column,
            F.lead("current_rainfall_mm", offset).over(window),
        )
        data = data.withColumn(
            time_column,
            F.lead("observation_time", offset).over(window),
        )

        future_rainfall = future_rainfall + F.col(rain_column)
        complete_future = complete_future & (
            F.col(time_column).cast("long") == current_time + offset * 300
        )

    threshold_values = F.array(*[F.lit(value) for value in thresholds])
    month_angle = (
        (F.month("observation_time") - F.lit(1))
        * F.lit(2.0 * math.pi / 12.0)
    )
    hour_angle = (
        F.hour("observation_time")
        * F.lit(2.0 * math.pi / 24.0)
    )

    return (
        data
        .filter(complete_history & complete_future)
        .filter(
            F.year("future_time_6") == F.year("observation_time")
        )
        .filter(F.minute("observation_time").isin(0, 30))
        .filter(
            F.pmod(
                F.xxhash64("station_id", "observation_time"),
                F.lit(sample_modulus),
            ) == 0
        )
        .withColumn("recent_15min_rainfall_mm", recent_15min)
        .withColumn("recent_30min_rainfall_mm", recent_30min)
        .withColumn("future_rainfall_mm", future_rainfall)
        .withColumn("month_sin", F.sin(month_angle))
        .withColumn("month_cos", F.cos(month_angle))
        .withColumn("hour_sin", F.sin(hour_angle))
        .withColumn("hour_cos", F.cos(hour_angle))
        .withColumn("threshold_mm", F.explode(threshold_values))
        .withColumn(
            "label",
            (F.col("future_rainfall_mm") > F.col("threshold_mm"))
            .cast("double"),
        )
        .select(
            "station_id",
            "observation_time",
            *NUMERIC_FEATURES,
            "future_rainfall_mm",
            "label",
        )
    )


def train_model(training_data):
    """Train the logistic regression baseline."""

    station_indexer = StringIndexer(
        inputCol="station_id",
        outputCol="station_index",
        handleInvalid="keep",
    )
    station_encoder = OneHotEncoder(
        inputCol="station_index",
        outputCol="station_vector",
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=["station_vector", *NUMERIC_FEATURES],
        outputCol="features",
    )
    classifier = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=MAX_ITER,
    )

    pipeline = Pipeline(
        stages=[station_indexer, station_encoder, assembler, classifier]
    )
    return pipeline.fit(training_data)


def predict_probability(
    model,
    spark,
    station_id,
    prediction_time,
    threshold_mm,
    current_rainfall_mm,
    recent_15min_rainfall_mm,
    recent_30min_rainfall_mm,
):
    """Predict from the station, time, threshold and recent rainfall."""

    prediction_input = (
        spark.range(1)
        .select(
            F.lit(station_id).alias("station_id"),
            F.lit(prediction_time).alias("prediction_time"),
            F.lit(float(threshold_mm)).alias("threshold_mm"),
            F.lit(float(current_rainfall_mm))
            .alias("current_rainfall_mm"),
            F.lit(float(recent_15min_rainfall_mm))
            .alias("recent_15min_rainfall_mm"),
            F.lit(float(recent_30min_rainfall_mm))
            .alias("recent_30min_rainfall_mm"),
        )
        .withColumn(
            "observation_time",
            F.to_timestamp("prediction_time"),
        )
        .withColumn(
            "month_angle",
            (F.month("observation_time") - F.lit(1))
            * F.lit(2.0 * math.pi / 12.0),
        )
        .withColumn(
            "hour_angle",
            F.hour("observation_time")
            * F.lit(2.0 * math.pi / 24.0),
        )
        .withColumn("month_sin", F.sin("month_angle"))
        .withColumn("month_cos", F.cos("month_angle"))
        .withColumn("hour_sin", F.sin("hour_angle"))
        .withColumn("hour_cos", F.cos("hour_angle"))
    )

    result = model.transform(prediction_input).select("probability").first()
    return float(result["probability"][1])


def save_model(model, path):
    """Save the fitted logistic regression parameters as JSON."""

    station_labels = list(model.stages[0].labels)
    classifier = model.stages[-1]
    coefficients = [float(value) for value in classifier.coefficients]

    saved_model = {
        "station_labels": station_labels,
        "station_feature_size": len(coefficients) - len(NUMERIC_FEATURES),
        "feature_order": ["station", *NUMERIC_FEATURES],
        "coefficients": coefficients,
        "intercept": float(classifier.intercept),
        "thresholds": THRESHOLDS,
    }

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(saved_model, indent=2),
        encoding="utf-8",
    )


def predict_saved_probability(
    path,
    station_id,
    prediction_time,
    threshold_mm,
    current_rainfall_mm,
    recent_15min_rainfall_mm,
    recent_30min_rainfall_mm,
):
    """Use the saved JSON model without starting Spark."""

    saved_model = json.loads(Path(path).read_text(encoding="utf-8"))
    coefficients = saved_model["coefficients"]
    station_labels = saved_model["station_labels"]
    station_size = saved_model["station_feature_size"]
    time = datetime.fromisoformat(prediction_time)

    score = saved_model["intercept"]
    if station_id in station_labels:
        station_index = station_labels.index(station_id)
        if station_index < station_size:
            score += coefficients[station_index]

    month_angle = 2.0 * math.pi * (time.month - 1) / 12.0
    hour_angle = 2.0 * math.pi * time.hour / 24.0
    numeric_values = [
        float(current_rainfall_mm),
        float(recent_15min_rainfall_mm),
        float(recent_30min_rainfall_mm),
        math.sin(month_angle),
        math.cos(month_angle),
        math.sin(hour_angle),
        math.cos(hour_angle),
        float(threshold_mm),
    ]

    for index, value in enumerate(numeric_values):
        score += coefficients[station_size + index] * value

    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))

    exponential = math.exp(score)
    return exponential / (1.0 + exponential)
