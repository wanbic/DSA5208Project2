import json
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

from .logistic_model import (
    NUMERIC_FEATURES,
    THRESHOLDS,
    make_examples as make_logistic_examples,
    predict_probability,
)


SAMPLE_MODULUS = 4
TRAIN_SAMPLE_MODULUS = 16
MAX_ITER = 20
MAX_DEPTH = 4
STEP_SIZE = 0.1
SUBSAMPLING_RATE = 0.8
SEED = 5208


def make_examples(rainfall, sample_modulus=SAMPLE_MODULUS):
    """Use exactly the same features as the logistic baseline."""

    return make_logistic_examples(rainfall, sample_modulus)


def train_model(training_data):
    """Train a gradient-boosted tree classifier."""

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
    classifier = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=MAX_ITER,
        maxDepth=MAX_DEPTH,
        stepSize=STEP_SIZE,
        subsamplingRate=SUBSAMPLING_RATE,
        seed=SEED,
    )

    pipeline = Pipeline(
        stages=[station_indexer, station_encoder, assembler, classifier]
    )
    return pipeline.fit(training_data)


def save_model(model, path):
    """Save the fitted trees and their weights as JSON."""

    classifier = model.stages[-1]
    saved_model = {
        "model_type": "Spark GBTClassifier",
        "station_labels": list(model.stages[0].labels),
        "feature_order": ["station", *NUMERIC_FEATURES],
        "thresholds": THRESHOLDS,
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
