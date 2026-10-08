"""Tests for the PySpark cleaning and labelling steps, on tiny hand-made data."""
from datetime import date, timedelta

import pytest
from pyspark.sql import functions as F

from src.config import HORIZON_DAYS, SMART_CONTEXT, SMART_SIGNALS
from src.spark_etl import attach_failure_dates, clean, drive_summary, weekly_features

pytestmark = pytest.mark.spark
SMART_COLS = list(SMART_SIGNALS) + list(SMART_CONTEXT)
START = date(2024, 1, 1)


def raw_row(day, serial, model="ST4000DM000", capacity=4e12, failure=0, realloc=0, hours=10000):
    row = {"date": (START + timedelta(days=day)).isoformat(), "serial_number": serial, "model": model,
           "capacity_bytes": str(int(capacity)), "failure": str(failure)}
    for c in SMART_COLS:
        row[c] = "0"
    row["smart_5_raw"] = str(realloc)
    row["smart_9_raw"] = str(hours + day * 24)
    return row


def to_df(spark, rows):
    return spark.createDataFrame(rows)


def test_manufacturers_are_recognised_and_boot_drives_removed(spark):
    rows = [
        raw_row(0, "a", model="ST4000DM000"),
        raw_row(0, "b", model="HGST HMS5C4040BLE640"),
        raw_row(0, "c", model="WDC WUH721816ALE6L4"),
        raw_row(0, "d", model="TOSHIBA MG07ACA14TA"),
        raw_row(0, "e", model="CT250MX500SSD1", capacity=250e9),  # boot SSD
        raw_row(0, "f", model="ST500LM012 HN", capacity=500e9),  # small boot HDD
    ]
    out = {r["serial_number"]: r["manufacturer"] for r in clean(to_df(spark, rows)).collect()}
    assert out == {"a": "Seagate", "b": "HGST", "c": "WDC", "d": "Toshiba"}


def test_rows_after_a_failure_are_dropped(spark):
    rows = [raw_row(d, "x", failure=int(d == 5)) for d in range(10)]
    df = attach_failure_dates(clean(to_df(spark, rows)))
    dates = sorted(r["date"] for r in df.collect())
    assert dates[-1] == START + timedelta(days=5)
    assert all(r["failure_date"] == START + timedelta(days=5) for r in df.collect())


def test_drive_summary_counts_days_and_failures(spark):
    rows = [raw_row(d, "x", failure=int(d == 5)) for d in range(10)] + [raw_row(d, "y") for d in range(10)]
    summary = {r["serial_number"]: r for r in drive_summary(attach_failure_dates(clean(to_df(spark, rows)))).collect()}
    assert summary["x"]["failed"] == 1 and summary["x"]["drive_days"] == 6
    assert summary["y"]["failed"] == 0 and summary["y"]["drive_days"] == 10


def test_labels_change_and_end_of_data_rule(spark):
    days = 90
    fail_day = 75  # inside the last HORIZON_DAYS, where the end-of-data rule applies
    rows = [raw_row(d, "fails", failure=int(d == fail_day), realloc=d) for d in range(fail_day + 1)]
    rows += [raw_row(d, "healthy") for d in range(days)]
    df = attach_failure_dates(clean(to_df(spark, rows)))
    max_date = START + timedelta(days=days - 1)
    feats = {(r["serial_number"], r["date"]): r for r in weekly_features(df, max_date).collect()}

    # Snapshots are weekly, counted back from the last day
    assert all((max_date - d).days % 7 == 0 for _, d in feats)

    fail_date = START + timedelta(days=fail_day)
    for (serial, d), r in feats.items():
        if serial == "fails":
            expected = 1 if 0 <= (fail_date - d).days <= HORIZON_DAYS else 0
            assert r["label"] == expected, d
            # 7-day change is filled in once there's a snapshot a week earlier
            if r["reallocated_sectors_change_7d"] is not None:
                assert r["reallocated_sectors_change_7d"] == 7
        # Nobody's label is trusted in the last HORIZON_DAYS of the data
        assert r["label_known"] == ((max_date - d).days >= HORIZON_DAYS)
