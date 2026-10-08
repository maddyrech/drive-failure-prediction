"""Tests for the model evaluation logic. These guard the decisions that
keep the reported results honest: time-based splits with a gap, sampling
weights, and the weekly check-budget metric."""
import numpy as np
import pandas as pd
import pytest

from src.config import HORIZON_DAYS
from src.train_model import (
    add_derived,
    budget_metrics,
    ranked_rule_scores,
    rule_flags,
    signed_log,
    split_before,
)


def weekly_rows(weeks=10, drives=100, positives_per_week=2):
    rows = []
    start = pd.Timestamp("2024-01-01")
    for w in range(weeks):
        for d in range(drives):
            rows.append({"date": start + pd.Timedelta(weeks=w), "label": int(d < positives_per_week),
                         "sample_weight": 1.0})
    return pd.DataFrame(rows)


def test_signed_log_keeps_sign_and_shrinks_large_values():
    out = signed_log(np.array([-1000.0, 0.0, 1000.0]))
    assert out[0] < 0 and out[1] == 0 and out[2] > 0
    assert abs(out[2]) < 10


def test_add_derived_sums_errors_and_treats_missing_as_zero():
    df = pd.DataFrame({"reallocated_sectors": [0, 3, np.nan], "uncorrectable_errors": [0, 1, np.nan],
                       "pending_sectors": [0, 0, 2], "offline_uncorrectable": [0, np.nan, 0]})
    out = add_derived(df)
    assert out["total_sector_errors"].tolist() == [0, 4, 2]
    assert out["any_sector_error"].tolist() == [0.0, 1.0, 1.0]
    assert rule_flags(out).tolist() == [False, True, True]


def test_ranked_rule_puts_drives_with_more_errors_first():
    df = pd.DataFrame({"total_sector_errors": [0, 50, 5], "age_days": [100, 100, 100]})
    scores = ranked_rule_scores(df)
    assert scores[1] > scores[2] > scores[0]


def test_perfect_scores_catch_every_failure_within_budget():
    df = weekly_rows()
    scores = df["label"].to_numpy(dtype=float)  # the model knows exactly which drives fail
    result = budget_metrics(df, scores, budget=0.02)  # 2 of 100 drives checked each week
    assert result["recall_pct"] == pytest.approx(100.0)
    assert result["precision_pct"] == pytest.approx(100.0)


def test_random_scores_catch_roughly_the_budget_share():
    df = weekly_rows(weeks=200, drives=200, positives_per_week=20)
    scores = np.random.default_rng(0).random(len(df))
    result = budget_metrics(df, scores, budget=0.10)
    assert 5 < result["recall_pct"] < 15  # about 10%, as chance would give


def test_sample_weights_are_respected():
    # One positive row, plus one sampled healthy row standing in for 20 healthy drives
    df = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")] * 2, "label": [1, 0], "sample_weight": [1.0, 20.0]})
    result = budget_metrics(df, np.array([0.9, 0.1]), budget=1 / 21)
    assert result["recall_pct"] == pytest.approx(100.0)
    assert result["precision_pct"] == pytest.approx(100.0)


def test_time_split_leaves_a_gap_so_labels_cannot_leak():
    df = weekly_rows(weeks=30)
    earlier, later, later_start, earlier_end = split_before(df, 0.3)
    assert earlier["date"].max() < later["date"].min()
    gap = later["date"].min() - earlier["date"].max()
    assert gap >= pd.Timedelta(days=HORIZON_DAYS)
    assert set(earlier.index).isdisjoint(later.index)
