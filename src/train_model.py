"""Predict which drives will fail in the next 30 days.

Baselines (what an ops team could do without machine learning):
  - Simple rule: flag every drive that reports any sector errors
  - Ranked rule: check drives in order of how many sector errors they report
Models:
  - Logistic regression (interpretable)
  - Gradient boosting, with and without class balancing

How models are chosen and tested, so nothing peeks at the future:
  |---- inner train ----| gap |-- validation --| gap |------ test ------|
  Candidates are compared on the validation weeks. The winner is refit on
  everything before the test period and scored ONCE on the test weeks.
  Each gap is HORIZON_DAYS long, so no label overlaps the next period.

Outputs:
    models/model.joblib
    reports/model_metrics.json
    reports/figures/precision_recall.png, alert_budget.png, feature_importance.png
    data/processed/budget_curve.parquet, at_risk_drives.parquet

Usage:
    python -m src.train_model
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import DATA_PROCESSED, FIGURES, HORIZON_DAYS, MODELS, REPORTS, SMART_SIGNALS

SIGNALS = list(SMART_SIGNALS.values())
# The error counters behind the classic "replace it if it reports bad sectors" rule
RULE_SIGNALS = ["reallocated_sectors", "uncorrectable_errors", "pending_sectors", "offline_uncorrectable"]
NUMERIC = (
    ["age_days", "capacity_tb", "power_cycles", "temperature_c"]
    + SIGNALS
    + [f"{s}_change_7d" for s in SIGNALS]
    + ["total_sector_errors", "any_sector_error"]
)
CATEGORICAL = ["manufacturer"]
FEATURES = NUMERIC + CATEGORICAL
BUDGETS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Give the models the rule's signal directly: total errors and a yes/no flag."""
    df = df.copy()
    df["total_sector_errors"] = df[RULE_SIGNALS].fillna(0).sum(axis=1)
    df["any_sector_error"] = (df["total_sector_errors"] > 0).astype(float)
    return df


def signed_log(x):
    """SMART counters are heavily skewed (mostly 0, sometimes thousands)."""
    return np.sign(x) * np.log1p(np.abs(x))


def logistic_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("log", FunctionTransformer(signed_log, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])


def boosting_pipeline(balanced: bool) -> Pipeline:
    # Gradient boosting copes with missing values and skew on its own
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CATEGORICAL),
    ])
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=100,
        l2_regularization=1.0, categorical_features=[len(NUMERIC)],
        class_weight="balanced" if balanced else None, random_state=42,
    )
    return Pipeline([("prep", pre), ("clf", clf)])


CANDIDATES = {
    "Logistic regression": logistic_pipeline,
    "Gradient boosting": lambda: boosting_pipeline(balanced=False),
    "Gradient boosting (balanced)": lambda: boosting_pipeline(balanced=True),
}


def fit(make, df: pd.DataFrame) -> Pipeline:
    pipe = make()
    pipe.fit(df[FEATURES], df["label"], clf__sample_weight=df["sample_weight"])
    return pipe


def rule_flags(df: pd.DataFrame) -> np.ndarray:
    return (df["total_sector_errors"] > 0).to_numpy()


def ranked_rule_scores(df: pd.DataFrame) -> np.ndarray:
    """More sector errors = check first (ties broken by age)."""
    return (df["total_sector_errors"] + df["age_days"].fillna(0) / 1e7).to_numpy()


def budget_metrics(df: pd.DataFrame, scores: np.ndarray, budget: float) -> dict:
    """If the team can check the riskiest `budget` share of drives each week,
    what share of soon-to-fail drives do they catch, and how many checks
    find a real problem? Weights undo the sampling of healthy drives."""
    d = df[["date", "label", "sample_weight"]].copy()
    d["score"] = scores
    flagged_w = flagged_pos = 0.0
    total_pos = (d["label"] * d["sample_weight"]).sum()
    for _, week in d.groupby("date"):
        week = week.sort_values("score", ascending=False)
        cum = week["sample_weight"].cumsum()
        top = week[cum - week["sample_weight"] < budget * week["sample_weight"].sum()]
        flagged_w += top["sample_weight"].sum()
        flagged_pos += (top["label"] * top["sample_weight"]).sum()
    return {
        "budget_pct": budget * 100,
        "recall_pct": round(100 * flagged_pos / total_pos, 2) if total_pos else 0.0,
        "precision_pct": round(100 * flagged_pos / flagged_w, 2) if flagged_w else 0.0,
    }


def evaluate(df: pd.DataFrame, scores: np.ndarray) -> dict:
    y, w = df["label"].to_numpy(), df["sample_weight"].to_numpy()
    return {
        "pr_auc": round(float(average_precision_score(y, scores, sample_weight=w)), 4),
        "roc_auc": round(float(roc_auc_score(y, scores, sample_weight=w)), 4),
        "budgets": [budget_metrics(df, scores, b) for b in BUDGETS],
    }


def recall_at(result: dict, budget_pct: float) -> float:
    return next(b["recall_pct"] for b in result["budgets"] if b["budget_pct"] == budget_pct)


def split_before(df: pd.DataFrame, share: float):
    """Last `share` of dates becomes the later period; the earlier period
    stops HORIZON_DAYS before it so labels can't overlap."""
    dates = np.sort(df["date"].unique())
    later_start = pd.Timestamp(dates[int(len(dates) * (1 - share))])
    earlier_end = later_start - pd.Timedelta(days=HORIZON_DAYS)
    return df[df["date"] < earlier_end], df[df["date"] >= later_start], later_start, earlier_end


def main() -> None:
    df = add_derived(pd.read_parquet(DATA_PROCESSED / "features"))
    df["date"] = pd.to_datetime(df["date"])

    dev, test, test_start, train_end = split_before(df, 0.3)
    inner, val, val_start, _ = split_before(dev, 0.3)
    for name, part in [("Inner train", inner), ("Validation", val), ("Train (all before test)", dev), ("Test", test)]:
        print(f"{name:24s} {part['date'].min().date()} to {part['date'].max().date()}: "
              f"{len(part):,} rows, {int(part['label'].sum())} soon-to-fail")
    if min(inner["label"].sum(), val["label"].sum(), test["label"].sum()) < 10:
        raise SystemExit("Too few failures in a period. Add more quarters of data.")

    # 1. Choose on the validation weeks
    validation = {}
    for name, make in CANDIDATES.items():
        p = fit(make, inner).predict_proba(val[FEATURES])[:, 1]
        validation[name] = evaluate(val, p)
        print(f"  validation  {name:30s} PR-AUC {validation[name]['pr_auc']:.4f}  "
              f"recall@1% {recall_at(validation[name], 1.0):.1f}%")
    best = max(validation, key=lambda k: validation[k]["pr_auc"])
    print(f"Chosen on validation: {best}")

    # 2. Score every candidate on the test weeks (reported for transparency),
    #    but only the one chosen above counts as "the model"
    results, test_scores, fitted = {}, {}, {}
    for name, make in CANDIDATES.items():
        fitted[name] = fit(make, dev)
        test_scores[name] = fitted[name].predict_proba(test[FEATURES])[:, 1]
        results[name] = evaluate(test, test_scores[name])

    ranked = evaluate(test, ranked_rule_scores(test))
    y_test, w_test = test["label"].to_numpy(), test["sample_weight"].to_numpy()
    flags = rule_flags(test)
    rule_rate = float((flags * w_test).sum() / w_test.sum())
    rule = {
        "flag_rate_pct": round(rule_rate * 100, 2),
        "recall_pct": round(100 * float((flags * y_test * w_test).sum() / (y_test * w_test).sum()), 2),
        "precision_pct": round(100 * float((flags * y_test * w_test).sum() / max((flags * w_test).sum(), 1e-9)), 2),
    }
    model_at_rule_rate = budget_metrics(test, test_scores[best], rule_rate)

    print(f"\nTest weeks from {test_start.date()}:")
    print(f"  Ranked rule         PR-AUC {ranked['pr_auc']:.4f}  recall@1% {recall_at(ranked, 1.0):.1f}%")
    print(f"  {best:19s} PR-AUC {results[best]['pr_auc']:.4f}  recall@1% {recall_at(results[best], 1.0):.1f}%")
    print(f"  Simple rule flags {rule['flag_rate_pct']}% and catches {rule['recall_pct']}%; "
          f"{best} checking the same number catches {model_at_rule_rate['recall_pct']}%")

    plot_precision_recall(test, test_scores, best, rule)
    plot_budget(results, ranked, best)
    plot_importance(fitted[best], test, best)

    # 3. Refit the chosen model on all labelled data before scoring today's fleet
    final = fit(CANDIDATES[best], df)
    joblib.dump(final, MODELS / "model.joblib")
    latest = add_derived(pd.read_parquet(DATA_PROCESSED / "latest_snapshot"))
    latest["risk_score"] = final.predict_proba(latest[FEATURES])[:, 1]
    latest = latest.sort_values("risk_score", ascending=False).reset_index(drop=True)
    latest["risk_rank"] = latest.index + 1
    keep = ["risk_rank", "serial_number", "model", "manufacturer", "capacity_tb", "age_days",
            "risk_score", "date"] + SIGNALS
    latest.head(3000)[keep].to_parquet(DATA_PROCESSED / "at_risk_drives.parquet", index=False)
    pd.DataFrame(results[best]["budgets"]).to_parquet(DATA_PROCESSED / "budget_curve.parquet", index=False)

    metrics = {
        "horizon_days": HORIZON_DAYS,
        "validation_start": str(val_start.date()),
        "train_end": str(train_end.date()),
        "test_start": str(test_start.date()),
        "train_rows": len(dev), "test_rows": len(test),
        "test_failure_weeks": int(y_test.sum()),
        "best_model": best,
        "chosen_on": "validation PR-AUC",
        "validation": {k: {"pr_auc": v["pr_auc"], "recall_at_1pct": recall_at(v, 1.0)} for k, v in validation.items()},
        "models": results,
        "ranked_rule": ranked,
        "simple_rule": rule,
        "best_model_at_rule_rate": model_at_rule_rate,
        "active_drives_scored": len(latest),
        "scored_date": str(pd.to_datetime(latest["date"]).max().date()),
    }
    (REPORTS / "model_metrics.json").write_text(json.dumps(metrics, indent=2))


def plot_precision_recall(test, scores, best, rule):
    y, w = test["label"].to_numpy(), test["sample_weight"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    curves = {"Ranked rule (error count)": ranked_rule_scores(test), best: scores[best]}
    for (name, p), color in zip(curves.items(), ["#C0392B", "#2F6690"]):
        prec, rec, _ = precision_recall_curve(y, p, sample_weight=w)
        ax.plot(rec, prec, color=color,
                label=f"{name} (PR-AUC {average_precision_score(y, p, sample_weight=w):.2f})")
    ax.scatter([rule["recall_pct"] / 100], [rule["precision_pct"] / 100], color="#C0392B", marker="s",
               zorder=5, label="Simple rule: any sector errors")
    ax.set_xlabel("Recall: share of failing drives caught")
    ax.set_ylabel("Precision: share of alerts that are real")
    ax.set_title(f"Predicting failure within {HORIZON_DAYS} days (test period)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "precision_recall.png", dpi=150)
    plt.close(fig)


def plot_budget(results, ranked, best):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, r, color in [("Ranked rule (error count)", ranked, "#C0392B"), (best, results[best], "#2F6690")]:
        b = pd.DataFrame(r["budgets"])
        ax.plot(b["budget_pct"], b["recall_pct"], marker="o", label=name, color=color)
    budget_pct = [b * 100 for b in BUDGETS]
    ax.set_xscale("log")
    ax.set_xticks(budget_pct)
    ax.set_xticklabels([f"{x:g}%" for x in budget_pct])
    ax.set_xlabel("Share of fleet checked each week")
    ax.set_ylabel("Share of soon-to-fail drives caught (%)")
    ax.set_title("What a weekly check budget buys (test period)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "alert_budget.png", dpi=150)
    plt.close(fig)


def plot_importance(pipe, test, name):
    sample = test.sample(min(len(test), 20000), random_state=0)
    imp = permutation_importance(
        pipe, sample[FEATURES], sample["label"], sample_weight=sample["sample_weight"],
        scoring="average_precision", n_repeats=5, random_state=0,
    )
    order = pd.Series(imp.importances_mean, index=FEATURES).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh([c.replace("_", " ") for c in order.index], order.values, color="#2F6690")
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.set_title(f"What the {name.lower()} relies on")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
