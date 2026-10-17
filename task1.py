import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from pyspark import StorageLevel
from pyspark.sql import functions as F

from utils.preprocess import (
    clean_rainfall_data,
    create_spark,
    load_rainfall_data,
    make_national_day,
    make_station_day,
    make_station_hour,
)


PROJECT_PATH = Path(__file__).resolve().parent
DATA_PATH = PROJECT_PATH / "data"
FIGURE_PATH = PROJECT_PATH / "figures"
TABLE_PATH = PROJECT_PATH / "tables"
BOUNDARY_PATH = PROJECT_PATH / "data" / "MasterPlan2025RegionBoundaryNoSea.geojson"


def read_with_progress(df):
    """Materialize a Spark DataFrame and show read progress."""

    spark_context = df.sparkSession.sparkContext
    tracker = spark_context.statusTracker()
    job_group = "task1_read_rainfall"

    def count_rows():
        spark_context.setJobGroup(job_group, "Read rainfall CSV files")
        return df.count()

    last_percent = 0

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(count_rows)

        while not future.done():
            total_tasks = 0
            completed_tasks = 0
            stage_ids = set()

            for job_id in tracker.getJobIdsForGroup(job_group):
                job_info = tracker.getJobInfo(job_id)
                if job_info is not None:
                    stage_ids.update(job_info.stageIds)

            for stage_id in stage_ids:
                stage_info = tracker.getStageInfo(stage_id)
                if stage_info is not None:
                    total_tasks += stage_info.numTasks
                    completed_tasks += stage_info.numCompletedTasks

            if total_tasks > 0:
                percent = int(completed_tasks / total_tasks * 100)
                last_percent = max(last_percent, min(percent, 99))

            bar_width = 30
            filled = int(bar_width * last_percent / 100)
            bar = "=" * filled + "-" * (bar_width - filled)
            print(
                f"\rReading data [{bar}] {last_percent:3d}%",
                end="",
                flush=True,
            )
            time.sleep(0.5)

        future.result()

    print(f"\rReading data [{'=' * 30}] 100%", flush=True)


def plot_annual_rainfall(national_day):
    """Plot coverage-adjusted annual rainfall."""

    annual = (
        national_day.groupBy("year")
        .agg(
            F.avg("mean_daily_rainfall_mm").alias("mean_daily_mm"),
            F.countDistinct("date_local").alias("valid_dates"),
        )
        .withColumn(
            "days_in_year",
            F.when(F.col("year").isin(2020, 2024), 366).otherwise(365),
        )
        .withColumn(
            "estimated_annual_mm",
            F.col("mean_daily_mm") * F.col("days_in_year"),
        )
        .orderBy("year")
        .toPandas()
    )
    annual.to_csv(TABLE_PATH / "annual_rainfall.csv", index=False)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(
        annual["year"].astype(str),
        annual["estimated_annual_mm"],
        color="#4C78A8",
    )
    for bar, valid_dates in zip(bars, annual["valid_dates"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{valid_dates} days",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.xlabel("Year")
    plt.ylabel("Coverage-adjusted annual rainfall estimate (mm)")
    plt.title("Estimated annual rainfall")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH / "annual_rainfall.png", dpi=300)
    plt.close()


def plot_monthly_rainfall(national_day):
    """Plot average daily rainfall for each calendar month."""

    monthly = (
        national_day.groupBy("month")
        .agg(F.avg("mean_daily_rainfall_mm").alias("mean_daily_mm"))
        .orderBy("month")
        .toPandas()
    )
    monthly.to_csv(TABLE_PATH / "monthly_rainfall.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(
        monthly["month"],
        monthly["mean_daily_mm"],
        marker="o",
        color="#4C78A8",
    )
    plt.xticks(range(1, 13))
    plt.xlabel("Month")
    plt.ylabel("Mean daily rainfall (mm/day)")
    plt.title("Monthly rainfall pattern")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH / "monthly_rainfall.png", dpi=300)
    plt.close()


def plot_year_month_heatmap(national_day):
    """Plot a year-by-month rainfall heatmap."""

    year_month = (
        national_day.groupBy("year", "month")
        .agg(
            F.avg("mean_daily_rainfall_mm").alias("mean_daily_mm"),
            F.countDistinct("date_local").alias("valid_dates"),
        )
        .filter(F.col("valid_dates") >= 15)
        .orderBy("year", "month")
        .toPandas()
    )
    year_month.to_csv(TABLE_PATH / "year_month_rainfall.csv", index=False)
    matrix = year_month.pivot(
        index="year",
        columns="month",
        values="mean_daily_mm",
    )

    plt.figure(figsize=(11, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar_kws={"label": "Mean daily rainfall (mm/day)"},
    )
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.title("Rainfall by year and month")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH / "year_month_heatmap.png", dpi=300)
    plt.close()


def plot_hourly_rainfall(station_hour):
    """Plot the average rainfall pattern over the 24 hours of a day."""

    hourly = (
        station_hour.groupBy("hour")
        .agg(
            F.avg("hourly_rainfall_mm").alias("mean_hourly_mm"),
            F.avg(
                (F.col("hourly_rainfall_mm") > 0).cast("double")
            ).alias("wet_hour_probability"),
        )
        .orderBy("hour")
        .toPandas()
    )
    hourly.to_csv(TABLE_PATH / "hourly_rainfall.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(
        hourly["hour"],
        hourly["mean_hourly_mm"],
        marker="o",
        color="#4C78A8",
        label="Mean rainfall",
    )
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Mean hourly rainfall (mm)", color="#4C78A8")
    ax1.set_xticks(range(24))

    ax2 = ax1.twinx()
    ax2.plot(
        hourly["hour"],
        hourly["wet_hour_probability"],
        color="#E45756",
        label="Wet-hour probability",
    )
    ax2.set_ylabel("Wet-hour probability", color="#E45756")
    ax2.set_ylim(bottom=0)

    plt.title("Hourly rainfall pattern")
    fig.tight_layout()
    plt.savefig(FIGURE_PATH / "hourly_rainfall.png", dpi=300)
    plt.close()


def plot_station_map(station_day):
    """Plot monitoring stations and their long-term mean rainfall."""

    stations = (
        station_day
        .filter(F.col("valid_day"))
        .groupBy("station_id")
        .agg(
            F.expr("percentile_approx(longitude, 0.5)").alias("longitude"),
            F.expr("percentile_approx(latitude, 0.5)").alias("latitude"),
            F.avg("adjusted_daily_rainfall_mm").alias("mean_daily_mm"),
            F.countDistinct("date_local").alias("valid_days"),
        )
        .filter(F.col("valid_days") >= 365)
        .filter(F.col("longitude").isNotNull())
        .filter(F.col("latitude").isNotNull())
        .toPandas()
    )
    stations.to_csv(TABLE_PATH / "station_rainfall.csv", index=False)

    with BOUNDARY_PATH.open(encoding="utf-8") as file:
        boundary = json.load(file)

    fig, ax = plt.subplots(figsize=(9, 6))
    for feature in boundary["features"]:
        for polygon in feature["geometry"]["coordinates"]:
            for ring in polygon:
                longitude, latitude = zip(*ring)
                ax.plot(
                    longitude,
                    latitude,
                    color="#777777",
                    linewidth=0.6,
                    zorder=1,
                )

    points = ax.scatter(
        stations["longitude"],
        stations["latitude"],
        c=stations["mean_daily_mm"],
        cmap="Blues",
        s=55,
        edgecolor="black",
        linewidth=0.3,
        zorder=2,
    )
    fig.colorbar(points, ax=ax, label="Mean daily rainfall (mm/day)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Rainfall monitoring stations in Singapore")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH / "station_map.png", dpi=300)
    plt.close()


def main():
    """Run preprocessing and create all Task 1 figures."""

    FIGURE_PATH.mkdir(exist_ok=True)
    TABLE_PATH.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    spark = create_spark()
    spark.conf.set("spark.sql.session.timeZone", "Asia/Singapore")
    spark.sparkContext.setLogLevel("WARN")

    rainfall = load_rainfall_data(spark, DATA_PATH)
    rainfall = clean_rainfall_data(rainfall)
    rainfall = rainfall.persist(StorageLevel.DISK_ONLY)
    read_with_progress(rainfall)

    station_day = make_station_day(rainfall).cache()
    national_day = make_national_day(station_day).cache()
    station_hour = make_station_hour(rainfall)

    plot_annual_rainfall(national_day)
    plot_monthly_rainfall(national_day)
    plot_year_month_heatmap(national_day)
    plot_hourly_rainfall(station_hour)
    plot_station_map(station_day)

    spark.stop()


if __name__ == "__main__":
    main()
