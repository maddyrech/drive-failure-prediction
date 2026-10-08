"""Turn the raw daily CSV files into analysis-ready tables with PySpark.

Built to run on an ordinary laptop (8 GB RAM), in two stages:

  Stage 1  CSV -> Parquet. Reads the raw files once, keeps only the columns
           we need, and writes them in Parquet, a compressed column format.
           This step streams through the data and needs very little memory.

  Stage 2  Parquet -> tables. Everything heavy works from the Parquet copy,
           and the model features use one snapshot per drive per week, which
           is 7x less data to sort than the full daily records.

Input:  data/raw/data_<quarter>/*.csv   (one row per drive per day)
Output: data/processed/
    drive_summary/    one row per drive: model, maker, age, failed or not
    daily_fleet/      drives running and failures per day and maker
    features/         weekly snapshots with SMART signals, 7-day changes
                      and the label "fails within HORIZON_DAYS"
    latest_snapshot/  every drive still running on the last day, ready to score

Usage:
    python -m src.spark_etl
"""
import os
from collections import defaultdict

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.config import (
    DATA_PROCESSED,
    DATA_RAW,
    HORIZON_DAYS,
    MANUFACTURERS,
    NEGATIVE_SAMPLE_RATE,
    SMART_CONTEXT,
    SMART_SIGNALS,
    SNAPSHOT_EVERY_DAYS,
)

SMART_COLS = list(SMART_SIGNALS) + list(SMART_CONTEXT)
DAILY_PARQUET = DATA_PROCESSED.parent / "interim" / "daily"


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("drive-failure-etl")
        # Fewer tasks at once means more memory for each one
        .master(os.getenv("SPARK_MASTER", "local[4]"))
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "2g"))
        # More, smaller pieces when sorting, so each fits in memory
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "200"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


# ---------------------------------------------------------------- Stage 1

def read_raw(spark: SparkSession) -> DataFrame:
    """Read every daily file, coping with Backblaze's changing file layout.

    Backblaze added columns over the years, so files can have different
    headers. Spark matches CSV columns by position, so we group files by
    their header, read each group on its own, and join them by column name.
    """
    files = sorted(p for p in DATA_RAW.rglob("*.csv") if not p.name.startswith("._"))
    if not files:
        raise SystemExit("No CSV files in data/raw. Run src.download or src.make_sample_data first.")

    by_header = defaultdict(list)
    for path in files:
        with open(path, encoding="utf-8", errors="ignore") as f:
            by_header[f.readline().strip()].append(str(path))
    print(f"Found {len(files)} daily files in {len(by_header)} layout(s)")

    parts = []
    for header, paths in by_header.items():
        columns = header.split(",")
        df = spark.read.option("header", True).csv(paths)
        selected = [F.col(c) for c in ["date", "serial_number", "model", "capacity_bytes", "failure"]]
        for c in SMART_COLS:
            selected.append(F.col(c) if c in columns else F.lit(None).cast("string").alias(c))
        parts.append(df.select(*selected))

    raw = parts[0]
    for part in parts[1:]:
        raw = raw.unionByName(part)
    return raw


def clean(raw: DataFrame) -> DataFrame:
    model = F.upper(F.trim(F.col("model")))
    manufacturer = (
        F.when(model.startswith("ST") | model.startswith("SEAGATE"), "Seagate")
        .when(
            model.startswith("HGST") | model.startswith("HITACHI")
            | model.startswith("HUH") | model.startswith("HMS") | model.startswith("HDS"),
            "HGST",
        )
        .when(model.startswith("WDC") | model.startswith("WUH") | model.startswith("WD"), "WDC")
        .when(model.startswith("TOSHIBA") | model.startswith("MG") | model.startswith("MD"), "Toshiba")
        .otherwise("Other")
    )
    df = raw.select(
        F.to_date("date").alias("date"),
        F.trim("serial_number").alias("serial_number"),
        F.trim("model").alias("model"),
        manufacturer.alias("manufacturer"),
        F.when(F.col("capacity_bytes").cast("double") > 0,
               F.round(F.col("capacity_bytes").cast("double") / 1e12, 1)).alias("capacity_tb"),
        F.col("failure").cast("int").alias("failure"),
        *[F.col(c).cast("double").alias(c) for c in SMART_COLS],
    )
    df = df.filter(F.col("manufacturer").isin(MANUFACTURERS))
    # Data drives only: Backblaze's boot drives (small HDDs and SSDs, under
    # 1 TB) do a different job and would distort the failure rates
    df = df.filter(F.col("capacity_tb") >= 1.0)
    return df.filter(F.col("date").isNotNull() & F.col("serial_number").isNotNull())


def csv_to_parquet(spark: SparkSession) -> None:
    clean(read_raw(spark)).write.mode("overwrite").parquet(str(DAILY_PARQUET))
    print(f"Stage 1: cleaned daily records saved as Parquet in {DAILY_PARQUET}")


# ---------------------------------------------------------------- Stage 2

def attach_failure_dates(df: DataFrame) -> DataFrame:
    """Each drive's first failure date. Rows after it are dropped
    (a drive is removed from service when it fails). Failures are rare,
    so this small table is broadcast instead of shuffling every row."""
    failures = (
        df.filter(F.col("failure") == 1)
        .groupBy("serial_number")
        .agg(F.min("date").alias("failure_date"))
    )
    df = df.join(F.broadcast(failures), "serial_number", "left")
    return df.filter(F.col("failure_date").isNull() | (F.col("date") <= F.col("failure_date")))


def drive_summary(df: DataFrame) -> DataFrame:
    # Text columns go in the grouping keys rather than being aggregated,
    # which lets Spark use its fast, low-memory hash aggregation
    return df.groupBy("serial_number", "model", "manufacturer").agg(
        F.max("capacity_tb").alias("capacity_tb"),
        F.min("date").alias("first_date"),
        F.max("date").alias("last_date"),
        F.max("failure").alias("failed"),
        F.max("failure_date").alias("failure_date"),
        F.count("*").alias("drive_days"),
        (F.min("smart_9_raw") / 24).alias("age_days_start"),
        (F.max("smart_9_raw") / 24).alias("age_days_end"),
        F.max("smart_5_raw").alias("max_reallocated_sectors"),
        F.max("smart_197_raw").alias("max_pending_sectors"),
    )


def daily_fleet(df: DataFrame) -> DataFrame:
    return df.groupBy("date", "manufacturer").agg(
        F.count("*").alias("drives"),
        F.sum("failure").alias("failures"),
    )


def weekly_features(df: DataFrame, max_date) -> DataFrame:
    """One snapshot per drive per week, counted back from the last day,
    with each SMART signal, how much it moved since last week, and the label."""
    on_snapshot_day = F.datediff(F.lit(max_date), "date") % SNAPSHOT_EVERY_DAYS == 0
    weekly = df.filter(on_snapshot_day)

    w = Window.partitionBy("serial_number").orderBy("date")
    prev_date = F.lag("date").over(w)
    one_week_apart = F.datediff("date", prev_date) == SNAPSHOT_EVERY_DAYS

    feats = weekly.withColumn("age_days", F.col("smart_9_raw") / 24)
    for col, name in SMART_SIGNALS.items():
        feats = feats.withColumnRenamed(col, name)
        # Change since last week's snapshot (left empty if the drive missed it)
        feats = feats.withColumn(
            f"{name}_change_7d",
            F.when(one_week_apart, F.col(name) - F.lag(name).over(w)),
        )
    for col, name in SMART_CONTEXT.items():
        if col != "smart_9_raw":
            feats = feats.withColumnRenamed(col, name)

    days_to_failure = F.datediff("failure_date", "date")
    feats = feats.withColumn("label", F.when(days_to_failure.between(0, HORIZON_DAYS), 1).otherwise(0))
    # We only know whether a drive fails in the next 30 days if we can see
    # 30 days past it. The last 30 days are left out for EVERY drive: keeping
    # only the ones that went on to fail would make those weeks look far
    # riskier than they were.
    known = F.col("date") <= F.date_sub(F.lit(max_date), HORIZON_DAYS)
    return feats.withColumn("label_known", known)


def build_tables(spark: SparkSession) -> None:
    df = attach_failure_dates(spark.read.parquet(str(DAILY_PARQUET)))
    bounds = df.agg(F.min("date").alias("min_date"), F.max("date").alias("max_date")).first()
    max_date = bounds["max_date"]
    print(f"Data runs from {bounds['min_date']} to {max_date}")

    out = DATA_PROCESSED
    drive_summary(df).write.mode("overwrite").parquet(str(out / "drive_summary"))
    daily_fleet(df).write.mode("overwrite").parquet(str(out / "daily_fleet"))
    print("Wrote drive_summary and daily_fleet")

    # Save the weekly snapshots once, then build both outputs from them
    weekly_path = DATA_PROCESSED.parent / "interim" / "weekly"
    weekly_features(df, max_date).write.mode("overwrite").parquet(str(weekly_path))
    feats = spark.read.parquet(str(weekly_path))

    # Every drive still running on the last day: what the dashboard scores
    latest = feats.filter((F.col("date") == F.lit(max_date)) & F.col("failure_date").isNull())
    latest.drop("label", "label_known").write.mode("overwrite").parquet(str(out / "latest_snapshot"))

    # Keep every soon-to-fail row and a share of the healthy ones
    labelled = feats.filter(F.col("label_known"))
    keep = (F.col("label") == 1) | (F.rand(seed=42) < NEGATIVE_SAMPLE_RATE)
    sampled = labelled.filter(keep).withColumn(
        "sample_weight",
        F.when(F.col("label") == 1, F.lit(1.0)).otherwise(F.lit(1.0 / NEGATIVE_SAMPLE_RATE)),
    )
    sampled.drop("label_known", "failure").write.mode("overwrite").parquet(str(out / "features"))

    counts = spark.read.parquet(str(out / "features")).groupBy("label").count().collect()
    print("Training rows by label:", {r["label"]: r["count"] for r in counts})


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("ERROR")
    csv_to_parquet(spark)
    build_tables(spark)
    spark.stop()


if __name__ == "__main__":
    main()
