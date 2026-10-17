"""Model helpers for month/hour feature ablation experiments."""

import json
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import functions as F

from .rainfall_model import (
    NEIGHBOR_FEATURE,
    NUMERIC_FEATURES as FULL_NUMERIC_FEATURES,
    build_neighbor_pairs,
    make_examples as make_neighbor_examples,
    save_neighbor_mapping,
)


CYCLIC_TIME_FEATURES = {
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
}

TIME_ABLATION_CONFIGS = {
    "remove_time": {
        "label": "Remove month and hour",
        "features": [
            feature
            for feature in FULL_NUMERIC_FEATURES
            if feature not in CYCLIC_TIME_FEATURES
        ],
    },
    "noncyclic_time": {
        "label": "Non-cyclic month and hour",
        "features": [
            "current_rainfall_mm",
            "recent_15min_rainfall_mm",
            "recent_30min_rainfall_mm",
            "month",
            "hour",
            "threshold_mm",
            NEIGHBOR_FEATURE,
        ],
    },
}


def make_examples(
    rainfall,
    neighbor_pairs,
    sample_modulus,
    thresholds,
):
    """Create final-model examples and add raw month and hour columns."""

    return (
        make_neighbor_examples(
            rainfall,
            neighbor_pairs,
            sample_modulus=sample_modulus,
            thresholds=thresholds,
        )
        .withColumn("month", F.month("observation_time").cast("double"))
        .withColumn("hour", F.hour("observation_time").cast("double"))
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
    """Train a GBT using the selected time-feature configuration."""

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
    """Save the fitted model and its exact time-feature representation."""

    classifier = model.stages[-1]
    saved_model = {
        "model_type": "Spark GBTClassifier time-feature ablation",
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


__all__ = [
    "TIME_ABLATION_CONFIGS",
    "build_neighbor_pairs",
    "make_examples",
    "save_model",
    "save_neighbor_mapping",
    "train_model",
]
