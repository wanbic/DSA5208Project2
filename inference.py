"""Run one prediction with a saved final rainfall model."""

import argparse
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path

from pyspark.sql import functions as F

from utils.preprocess import clean_rainfall_data, create_spark, load_rainfall_data


PROJECT_PATH = Path(__file__).resolve().parent


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(
        description="Predict next-30-minute rainfall exceedance probability."
    )
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument("--station-id", required=True)
    parser.add_argument(
        "--prediction-time",
        required=True,
        help='Singapore local time, for example "2024-06-01 14:30:00".',
    )
    parser.add_argument("--threshold-mm", type=float, required=True)
    parser.add_argument("--data-path", type=Path, default=PROJECT_PATH / "data")
    parser.add_argument("--spark-master", default="local[4]")
    parser.add_argument("--driver-memory", default="4g")
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    parser.add_argument(
        "--spark-temp-path",
        type=Path,
        default=PROJECT_PATH / "spark_tmp",
    )
    args = parser.parse_args(arguments)

    try:
        args.prediction_time = datetime.fromisoformat(args.prediction_time)
    except ValueError:
        parser.error("--prediction-time must be an ISO date and time")
    if args.prediction_time.tzinfo is not None:
        parser.error(
            "--prediction-time must be Singapore local time without an offset"
        )
    if (
        args.prediction_time.minute not in (0, 30)
        or args.prediction_time.second != 0
        or args.prediction_time.microsecond != 0
    ):
        parser.error("--prediction-time must be exactly on minute 00 or 30")
    if args.threshold_mm < 0:
        parser.error("--threshold-mm cannot be negative")
    return args


def load_neighbors(mapping_path, station_id):
    rows = json.loads(mapping_path.read_text(encoding="utf-8"))
    neighbors = [
        row["neighbor_id"]
        for row in rows
        if row["station_id"] == station_id
    ]
    if not neighbors:
        raise ValueError(
            f"Station {station_id!r} is absent from the neighbor mapping"
        )
    return neighbors


def required_years(prediction_time):
    years = {prediction_time.year}
    if (prediction_time - timedelta(minutes=25)).year != prediction_time.year:
        years.add(prediction_time.year - 1)
    return sorted(years)


def parse_tree(debug_string):
    """Parse the Spark decision-tree text stored in model.json."""

    lines = [line for line in debug_string.splitlines()[1:] if line.strip()]

    def parse_node(position):
        text = lines[position].strip()
        if text.startswith("Predict: "):
            return ("predict", float(text.removeprefix("Predict: "))), position + 1

        continuous = re.fullmatch(
            r"If \(feature (\d+) <= ([-+0-9.eE]+)\)", text
        )
        categorical = re.fullmatch(
            r"If \(feature (\d+) in \{([^}]*)\}\)", text
        )
        if continuous:
            condition = (
                "continuous",
                int(continuous.group(1)),
                float(continuous.group(2)),
            )
        elif categorical:
            condition = (
                "categorical",
                int(categorical.group(1)),
                frozenset(
                    float(value)
                    for value in categorical.group(2).split(",")
                    if value
                ),
            )
        else:
            raise ValueError(f"Unsupported tree line: {text}")

        left, position = parse_node(position + 1)
        if (
            position >= len(lines)
            or not lines[position].strip().startswith("Else ")
        ):
            raise ValueError("Invalid Spark tree: matching Else branch not found")
        right, position = parse_node(position + 1)
        return ("branch", condition, left, right), position

    tree, final_position = parse_node(0)
    if final_position != len(lines):
        raise ValueError("Invalid Spark tree: unparsed lines remain")
    return tree


def evaluate_tree(tree, feature_vector):
    while tree[0] == "branch":
        _, condition, left, right = tree
        condition_type, feature_index, split_value = condition
        value = feature_vector[feature_index]
        if condition_type == "continuous":
            go_left = value <= split_value
        else:
            go_left = value in split_value
        tree = left if go_left else right
    return tree[1]


def predict_from_json(
    model_data,
    station_id,
    prediction_time,
    threshold_mm,
    features,
):
    """Evaluate the exported Spark GBT without loading a Hadoop model."""

    numeric_names = model_data["feature_order"][1:]
    station_feature_size = (
        len(model_data["feature_importances"]) - len(numeric_names)
    )
    feature_vector = [0.0] * len(model_data["feature_importances"])
    try:
        station_index = model_data["station_labels"].index(station_id)
    except ValueError:
        station_index = len(model_data["station_labels"])
    if station_index >= station_feature_size:
        raise ValueError(f"Station {station_id!r} cannot be encoded by this model")
    feature_vector[station_index] = 1.0

    month_angle = (prediction_time.month - 1) * 2.0 * math.pi / 12.0
    hour_angle = prediction_time.hour * 2.0 * math.pi / 24.0
    values = {
        **features,
        "month_sin": math.sin(month_angle),
        "month_cos": math.cos(month_angle),
        "hour_sin": math.sin(hour_angle),
        "hour_cos": math.cos(hour_angle),
        "threshold_mm": float(threshold_mm),
    }
    for offset, name in enumerate(numeric_names, start=station_feature_size):
        feature_vector[offset] = float(values[name])

    margin = sum(
        float(weight) * evaluate_tree(parse_tree(tree), feature_vector)
        for weight, tree in zip(model_data["tree_weights"], model_data["trees"])
    )
    if margin >= 0:
        return 1.0 / (1.0 + math.exp(-2.0 * margin))
    exp_value = math.exp(2.0 * margin)
    return exp_value / (1.0 + exp_value)


def extract_features(rainfall, station_id, prediction_time, neighbor_ids):
    """Extract the rainfall inputs using the same definitions as training."""

    expected_times = [
        prediction_time - timedelta(minutes=offset)
        for offset in range(0, 30, 5)
    ]
    station_rows = (
        rainfall
        .filter(F.col("station_id") == station_id)
        .filter(F.col("observation_time").isin(expected_times))
        .groupBy("observation_time")
        .agg(F.avg("reading_value").alias("reading_value"))
        .collect()
    )
    readings = {
        row["observation_time"]: float(row["reading_value"])
        for row in station_rows
    }
    missing = [value for value in expected_times if value not in readings]
    if missing:
        missing_text = ", ".join(value.isoformat(sep=" ") for value in missing)
        raise ValueError(
            f"Missing 5-minute readings for station {station_id}: {missing_text}"
        )

    neighbor_row = (
        rainfall
        .filter(F.col("station_id").isin(neighbor_ids))
        .filter(F.col("observation_time") == F.lit(prediction_time))
        .agg(F.avg("reading_value").alias("neighbor_rainfall_mm"))
        .first()
    )
    neighbor_value = neighbor_row["neighbor_rainfall_mm"]
    return {
        "current_rainfall_mm": readings[expected_times[0]],
        "recent_15min_rainfall_mm": sum(
            readings[value] for value in expected_times[:3]
        ),
        "recent_30min_rainfall_mm": sum(
            readings[value] for value in expected_times
        ),
        "neighbor_rainfall_mm": (
            float(neighbor_value) if neighbor_value is not None else 0.0
        ),
    }


def main(arguments=None):
    args = parse_args(arguments)
    model_path = args.model_run / "model.json"
    mapping_path = args.model_run / "neighbor_mapping.json"
    if not model_path.exists():
        raise SystemExit(f"Model export not found at {model_path}")
    if not mapping_path.exists():
        raise SystemExit(f"Neighbor mapping not found at {mapping_path}")

    neighbors = load_neighbors(mapping_path, args.station_id)
    years = required_years(args.prediction_time)
    missing_files = [
        args.data_path / f"rainfall_{year}.csv"
        for year in years
        if not (args.data_path / f"rainfall_{year}.csv").exists()
    ]
    if missing_files:
        raise SystemExit(f"Data file not found: {missing_files[0]}")

    args.spark_temp_path.mkdir(parents=True, exist_ok=True)
    spark = create_spark(
        local_directory=args.spark_temp_path,
        master=args.spark_master,
        driver_memory=args.driver_memory,
        shuffle_partitions=args.shuffle_partitions,
    )
    spark.conf.set("spark.sql.session.timeZone", "Asia/Singapore")
    spark.sparkContext.setLogLevel("WARN")

    try:
        rainfall = clean_rainfall_data(
            load_rainfall_data(spark, args.data_path, years=years)
        )
        features = extract_features(
            rainfall,
            args.station_id,
            args.prediction_time,
            neighbors,
        )
        prediction_time_text = args.prediction_time.isoformat(sep=" ")
        model_data = json.loads(model_path.read_text(encoding="utf-8"))
        probability = predict_from_json(
            model_data,
            args.station_id,
            args.prediction_time,
            args.threshold_mm,
            features,
        )
        result = {
            "station_id": args.station_id,
            "prediction_time": prediction_time_text,
            "threshold_mm": args.threshold_mm,
            **features,
            "probability": probability,
        }
        print(json.dumps(result, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
