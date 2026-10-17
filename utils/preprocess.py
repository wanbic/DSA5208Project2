import os
import tempfile
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType


def create_spark(
    local_directory=None,
    master="local[4]",
    driver_memory="4g",
    shuffle_partitions=64,
):
    """Create the Spark session used by the project."""

    local_path = None
    if local_directory is not None:
        local_path = Path(local_directory).resolve()
        local_path.mkdir(parents=True, exist_ok=True)
        os.environ["TEMP"] = str(local_path)
        os.environ["TMP"] = str(local_path)
        tempfile.tempdir = str(local_path)

    builder = (
        SparkSession.builder
        .appName("Singapore Rainfall Task 1")
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.files.maxPartitionBytes", 64 * 1024 * 1024)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
    )
    if local_path is not None:
        local_path_text = str(local_path).replace("\\", "/")
        builder = builder.config(
            "spark.local.dir",
            local_path_text,
        ).config(
            "spark.driver.extraJavaOptions",
            f"-Djava.io.tmpdir={local_path_text}",
        )

    return builder.getOrCreate()


def load_rainfall_data(spark, data_path="data", years=None):
    """Read and combine the annual rainfall CSV files."""

    data_directory = Path(data_path).resolve()
    if years is None:
        csv_files = sorted(data_directory.glob("rainfall_*.csv"))
    else:
        csv_files = [
            data_directory / f"rainfall_{year}.csv"
            for year in years
        ]
    file_uris = [path.as_uri() for path in csv_files]

    schema = StructType([
        StructField("date", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("update_timestamp", StringType(), True),
        StructField("station_id", StringType(), True),
        StructField("station_name", StringType(), True),
        StructField("station_device_id", StringType(), True),
        StructField("location_longitude", DoubleType(), True),
        StructField("location_latitude", DoubleType(), True),
        StructField("reading_update_timestamp", StringType(), True),
        StructField("reading_value", DoubleType(), True),
        StructField("reading_type", StringType(), True),
        StructField("reading_unit", StringType(), True),
    ])

    return (
        spark.read
        .option("header", True)
        .schema(schema)
        .csv(file_uris)
    )


def clean_rainfall_data(df):
    """Parse time, remove invalid rows and add time columns."""

    return (
        df
        .withColumn(
            "observation_time",
            F.to_timestamp("timestamp", "yyyy-MM-dd'T'HH:mm:ssXXX"),
        )
        .filter(F.col("observation_time").isNotNull())
        .filter(F.col("station_id").isNotNull())
        .filter(F.col("reading_value").isNotNull())
        .filter(F.col("reading_value") >= 0)
        .withColumn("date_local", F.to_date("observation_time"))
        .withColumn("year", F.year("observation_time"))
        .withColumn("month", F.month("observation_time"))
        .withColumn("hour", F.hour("observation_time"))
        .filter(F.col("year").between(2017, 2024))
    )


def make_station_day(df, minimum_coverage=0.95):
    """Create one row per station and day."""

    return (
        df.groupBy("station_id", "date_local", "year", "month")
        .agg(
            F.sum("reading_value").alias("daily_rainfall_mm"),
            F.count("*").alias("slot_count"),
            F.expr("percentile_approx(location_longitude, 0.5)")
            .alias("longitude"),
            F.expr("percentile_approx(location_latitude, 0.5)")
            .alias("latitude"),
        )
        .withColumn("coverage", F.col("slot_count") / F.lit(288.0))
        .withColumn("valid_day", F.col("coverage") >= minimum_coverage)
        .withColumn(
            "adjusted_daily_rainfall_mm",
            F.col("daily_rainfall_mm") / F.col("coverage"),
        )
    )


def make_national_day(station_day):
    """Average valid station totals to obtain one value per day."""

    return (
        station_day
        .filter(F.col("valid_day"))
        .groupBy("date_local", "year", "month")
        .agg(
            F.avg("adjusted_daily_rainfall_mm")
            .alias("mean_daily_rainfall_mm"),
            F.expr(
                "percentile_approx(adjusted_daily_rainfall_mm, 0.5)"
            )
            .alias("median_daily_rainfall_mm"),
            F.countDistinct("station_id").alias("station_count"),
        )
        .filter(F.col("station_count") >= 30)
    )


def make_station_hour(df):
    """Create one row per station and hour, keeping complete hours."""

    return (
        df
        .withColumn("hour_start", F.date_trunc("hour", "observation_time"))
        .groupBy("station_id", "hour_start", "year", "month", "hour")
        .agg(
            F.sum("reading_value").alias("hourly_rainfall_mm"),
            F.count("*").alias("slot_count"),
        )
        .filter(F.col("slot_count") == 12)
    )
