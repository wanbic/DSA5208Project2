"""GBT model helpers for rainfall-history ablation experiments.

This module is intentionally separate from ``rainfall_model.py`` so that
ablation-only feature changes cannot alter the selected final model.
"""

import json
import math
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import functions as F

from logistic_model import (
    NUMERIC_FEATURES as BASE_NUMERIC_FEATURES,
    make_examples as make_base_examples,
)


NEIGHBOR_FEATURE = "neighbor_rainfall_mm"
FULL_NUMERIC_FEATURES = [*BASE_NUMERIC_FEATURES, NEIGHBOR_FEATURE]

ABLATION_CONFIGS = {
    "remove_30min": {
        "label": "Remove 30-minute history",
        "features": [
            feature
            for feature in FULL_NUMERIC_FEATURES
            if feature != "recent_30min_rainfall_mm"
        ],
    },
    "remove_15min_30min": {
        "label": "Remove 15- and 30-minute history",
        "features": [
            feature
            for feature in FULL_NUMERIC_FEATURES
            if feature
            not in {
                "recent_15min_rainfall_mm",
                "recent_30min_rainfall_mm",
            }
        ],
    },
}


def distance_km(latitude_1, longitude_1, latitude_2, longitude_2):
    """Calculate the geographic distance between two stations."""

    radius = 6371.0
    latitude_1 = math.radians(latitude_1)
    latitude_2 = math.radians(latitude_2)
    latitude_difference = latitude_2 - latitude_1
    longitude_difference = math.radians(longitude_2 - longitude_1)

    value = (
        math.sin(latitude_difference / 2.0) ** 2
        + math.cos(latitude_1)
        * math.cos(latitude_2)
        * math.sin(longitude_difference / 2.0) ** 2
    )
    return 2.0 * radius * math.asin(math.sqrt(value))


def build_neighbor_pairs(rainfall, neighbor_count):
    """Find the nearest stations from their median coordinates."""

    locations = (
        rainfall
        .groupBy("station_id")
        .agg(
            F.expr("percentile_approx(location_latitude, 0.5)")
            .alias("latitude"),
            F.expr("percentile_approx(location_longitude, 0.5)")
            .alias("longitude"),
        )
        .filter(F.col("latitude").isNotNull())
        .filter(F.col("longitude").isNotNull())
        .collect()
    )

    neighbor_pairs = []
    for station in locations:
        distances = []
        for candidate in locations:
            if candidate["station_id"] == station["station_id"]:
                continue
            distance = distance_km(
                station["latitude"],
                station["longitude"],
                candidate["latitude"],
                candidate["longitude"],
            )
            distances.append((distance, candidate["station_id"]))

        distances.sort()
        for distance, neighbor_id in distances[:neighbor_count]:
            neighbor_pairs.append(
                (station["station_id"], neighbor_id, distance)
            )

    return neighbor_pairs


def make_examples(
    rainfall,
    neighbor_pairs,
    sample_modulus,
    thresholds,
):
    """Create the same examples and neighbor feature as the final model."""

    spark = rainfall.sparkSession
    mapping_values = ", ".join(
        f"('{station_id}', '{neighbor_id}', {distance})"
        for station_id, neighbor_id, distance in neighbor_pairs
    )
    mapping = spark.sql(
        "SELECT * FROM VALUES "
        f"{mapping_values} "
        "AS neighbors(station_id, neighbor_id, distance_km)"
    )

    anchor_requests = (
        rainfall
        .filter(F.minute("observation_time").isin(0, 30))
        .filter(
            F.pmod(
                F.xxhash64("station_id", "observation_time"),
                F.lit(sample_modulus),
            ) == 0
        )
        .select("station_id", "observation_time")
        .join(
            F.broadcast(mapping.select("station_id", "neighbor_id")),
            "station_id",
        )
    )
    neighbor_readings = rainfall.select(
        F.col("station_id").alias("neighbor_id"),
        "observation_time",
        F.col("reading_value").alias("neighbor_reading_mm"),
    )
    neighbor_rainfall = (
        anchor_requests
        .join(
            neighbor_readings,
            ["neighbor_id", "observation_time"],
        )
        .groupBy("station_id", "observation_time")
        .agg(F.avg("neighbor_reading_mm").alias(NEIGHBOR_FEATURE))
    )

    return (
        make_base_examples(rainfall, sample_modulus, thresholds)
        .join(
            neighbor_rainfall,
            ["station_id", "observation_time"],
            "left",
        )
        .fillna({NEIGHBOR_FEATURE: 0.0})
    )


def train_model(
    training_data,
    numeric_features,
    max_iter,
    max_depth,
    step_size,
    subsampling_rate,
    seed,
):
    """Train a GBT using only the selected ablation features."""

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
        inputCols=["station_vector", *numeric_features],
        outputCol="features",
    )
    classifier = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=max_iter,
        maxDepth=max_depth,
        stepSize=step_size,
        subsamplingRate=subsampling_rate,
        seed=seed,
    )

    return Pipeline(
        stages=[station_indexer, station_encoder, assembler, classifier]
    ).fit(training_data)


def save_model(
    model,
    path,
    thresholds,
    neighbor_count,
    numeric_features,
):
    """Save the fitted ablation model and its exact feature order."""

    classifier = model.stages[-1]
    saved_model = {
        "model_type": "Spark GBTClassifier history ablation",
        "station_labels": list(model.stages[0].labels),
        "feature_order": ["station", *numeric_features],
        "thresholds": thresholds,
        "neighbor_count": neighbor_count,
        "tree_weights": [
            float(weight) for weight in classifier.treeWeights
        ],
        "feature_importances": [
            float(value) for value in classifier.featureImportances
        ],
        "trees": [tree.toDebugString for tree in classifier.trees],
    }

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(saved_model, indent=2),
        encoding="utf-8",
    )


def save_neighbor_mapping(neighbor_pairs, path):
    """Save the station-neighbor mapping shared by all variants."""

    rows = [
        {
            "station_id": station_id,
            "neighbor_id": neighbor_id,
            "distance_km": distance,
        }
        for station_id, neighbor_id, distance in neighbor_pairs
    ]
    Path(path).write_text(json.dumps(rows, indent=2), encoding="utf-8")
