"""Tests for the failure-rate statistics."""
import numpy as np
import pandas as pd
import pytest

from src.stats_analysis import (
    failure_rates,
    poisson_ci,
    test_manufacturer_difference as manufacturer_difference,
    test_reallocated_sectors as reallocated_sectors,
)


def make_drives(maker, n, failures, days=365, flagged_failed=0, flagged_ok=0):
    rows = []
    for i in range(n):
        failed = int(i < failures)
        flagged = (i < flagged_failed) if failed else (i - failures < flagged_ok)
        rows.append({"serial_number": f"{maker}{i}", "manufacturer": maker, "drive_days": days,
                     "failed": failed, "max_reallocated_sectors": 5.0 if flagged else 0.0})
    return pd.DataFrame(rows)


def test_poisson_interval_contains_the_count():
    low, high = poisson_ci(10)
    assert low < 10 < high


def test_poisson_interval_with_zero_failures_starts_at_zero():
    low, high = poisson_ci(0)
    assert low == 0.0
    assert high > 0


def test_annualised_failure_rate_formula():
    # 100 drives running a full year with 2 failures is an AFR of 2%
    afr = failure_rates(make_drives("Seagate", 100, 2))
    row = afr.iloc[0]
    assert row["drive_years"] == pytest.approx(100)
    assert row["afr_pct"] == pytest.approx(2.0)
    assert row["afr_low_pct"] < row["afr_pct"] < row["afr_high_pct"]


def test_equal_failure_rates_are_not_flagged_as_different():
    drives = pd.concat([make_drives("Seagate", 1000, 20), make_drives("WDC", 1000, 20)])
    result = manufacturer_difference(failure_rates(drives))
    assert result["p_value"] > 0.5


def test_very_different_failure_rates_are_flagged():
    drives = pd.concat([make_drives("Seagate", 1000, 80), make_drives("WDC", 1000, 5)])
    result = manufacturer_difference(failure_rates(drives))
    assert result["p_value"] < 0.001


def test_warning_sign_raises_relative_risk():
    # 40 of 50 failed drives showed sector errors, but only 10 of 950 healthy ones
    drives = make_drives("Seagate", 1000, 50, flagged_failed=40, flagged_ok=10)
    result = reallocated_sectors(drives)
    assert result["relative_risk"] > 10
    low, high = result["relative_risk_ci95"]
    assert low < result["relative_risk"] < high
    assert result["p_value"] < 0.001
