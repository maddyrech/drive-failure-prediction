"""Statistics on the drive fleet.

Answers three questions a fleet manager would ask:
  1. Do some manufacturers really fail more often, or is it noise?
  2. Are reallocated sectors a real warning sign?
  3. How long do drives from each manufacturer tend to last?

Outputs:
    reports/stats_results.json
    reports/figures/afr_by_manufacturer.png
    reports/figures/survival_by_manufacturer.png
    data/processed/afr_by_manufacturer.parquet

Usage:
    python -m src.stats_analysis
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy import stats

from src.config import DATA_PROCESSED, FIGURES, REPORTS

COLORS = {"Seagate": "#2F6690", "HGST": "#3A7D44", "WDC": "#8C5E58", "Toshiba": "#C48A1A"}


def poisson_ci(failures: int, alpha: float = 0.05):
    """Exact 95% interval for a Poisson count (standard for failure rates)."""
    lower = stats.chi2.ppf(alpha / 2, 2 * failures) / 2 if failures > 0 else 0.0
    upper = stats.chi2.ppf(1 - alpha / 2, 2 * failures + 2) / 2
    return lower, upper


def failure_rates(drives: pd.DataFrame) -> pd.DataFrame:
    """Annualised failure rate (AFR), the industry-standard measure:
    failures per drive-year of running time, as a percentage."""
    grouped = drives.groupby("manufacturer").agg(
        drives=("serial_number", "count"),
        drive_days=("drive_days", "sum"),
        failures=("failed", "sum"),
    )
    grouped["drive_years"] = grouped["drive_days"] / 365
    grouped["afr_pct"] = grouped["failures"] / grouped["drive_years"] * 100
    cis = grouped["failures"].apply(lambda k: poisson_ci(int(k)))
    grouped["afr_low_pct"] = [lo for lo, _ in cis] / grouped["drive_years"] * 100
    grouped["afr_high_pct"] = [hi for _, hi in cis] / grouped["drive_years"] * 100
    return grouped.reset_index().sort_values("afr_pct", ascending=False)


def test_manufacturer_difference(afr: pd.DataFrame) -> dict:
    """Chi-square test, adjusted for exposure: if every manufacturer failed at
    the same rate, failures would be spread in proportion to running time."""
    observed = afr["failures"].to_numpy(dtype=float)
    expected = observed.sum() * afr["drive_days"] / afr["drive_days"].sum()
    chi2, p = stats.chisquare(observed, expected)
    return {"chi2": round(float(chi2), 2), "dof": len(afr) - 1, "p_value": float(p)}


def test_reallocated_sectors(drives: pd.DataFrame) -> dict:
    """Do drives that ever reallocated a sector fail more often?"""
    flagged = drives["max_reallocated_sectors"].fillna(0) > 0
    failed = drives["failed"] == 1
    table = pd.crosstab(flagged, failed).reindex(index=[False, True], columns=[False, True], fill_value=0)
    chi2, p, _, _ = stats.chi2_contingency(table)

    a, b = table.loc[True, True], table.loc[True, False]
    c, d = table.loc[False, True], table.loc[False, False]
    risk_flagged, risk_clean = a / (a + b), c / (c + d)
    rr = risk_flagged / risk_clean if risk_clean > 0 else float("inf")
    # 95% CI for a relative risk, on the log scale
    se = np.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d)) if min(a, c) > 0 else np.nan
    rr_low, rr_high = np.exp(np.log(rr) - 1.96 * se), np.exp(np.log(rr) + 1.96 * se)

    # Non-parametric comparison of how many sectors were reallocated
    mw = stats.mannwhitneyu(
        drives.loc[failed, "max_reallocated_sectors"].fillna(0),
        drives.loc[~failed, "max_reallocated_sectors"].fillna(0),
        alternative="greater",
    )
    return {
        "share_flagged_pct": round(float(flagged.mean() * 100), 2),
        "failure_risk_flagged_pct": round(float(risk_flagged * 100), 2),
        "failure_risk_clean_pct": round(float(risk_clean * 100), 2),
        "relative_risk": round(float(rr), 1),
        "relative_risk_ci95": [round(float(rr_low), 1), round(float(rr_high), 1)],
        "chi2": round(float(chi2), 1),
        "p_value": float(p),
        "mann_whitney_p_value": float(mw.pvalue),
    }


def survival_curves(drives: pd.DataFrame) -> dict:
    """Kaplan-Meier survival by drive age.

    Drives enter the data already partway through their life, so we tell the
    model each drive's age when we first saw it ("left truncation"). Without
    that, old drives would look unrealistically reliable.
    """
    d = drives.dropna(subset=["age_days_start", "age_days_end"])
    d = d[d["age_days_end"] > d["age_days_start"]]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    five_year = {}
    for maker, group in d.groupby("manufacturer"):
        kmf = KaplanMeierFitter(label=maker)
        kmf.fit(
            group["age_days_end"] / 365,
            event_observed=group["failed"],
            entry=group["age_days_start"] / 365,
        )
        kmf.plot_survival_function(ax=ax, ci_show=True, color=COLORS.get(maker))
        if kmf.timeline.max() >= 5:
            five_year[maker] = round(float(kmf.predict(5.0)) * 100, 1)
    ax.set_xlabel("Drive age (years)")
    ax.set_ylabel("Share of drives still running")
    ax.set_title("How long drives last, by manufacturer")
    ax.set_xlim(left=0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "survival_by_manufacturer.png", dpi=150)
    plt.close(fig)
    return {"survival_at_5_years_pct": five_year}


def plot_afr(afr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    order = afr.sort_values("afr_pct")
    errors = [order["afr_pct"] - order["afr_low_pct"], order["afr_high_pct"] - order["afr_pct"]]
    ax.barh(order["manufacturer"], order["afr_pct"],
            color=[COLORS.get(m, "#777") for m in order["manufacturer"]],
            xerr=errors, capsize=4)
    ax.set_xlabel("Annualised failure rate (%), with 95% interval")
    ax.set_title("Failure rate by manufacturer")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "afr_by_manufacturer.png", dpi=150)
    plt.close(fig)


def main() -> None:
    drives = pd.read_parquet(DATA_PROCESSED / "drive_summary")
    afr = failure_rates(drives)
    afr.to_parquet(DATA_PROCESSED / "afr_by_manufacturer.parquet", index=False)
    plot_afr(afr)

    fleet_failures = int(drives["failed"].sum())
    fleet_years = drives["drive_days"].sum() / 365
    results = {
        "drives": int(len(drives)),
        "failures": fleet_failures,
        "drive_days": int(drives["drive_days"].sum()),
        "fleet_afr_pct": round(fleet_failures / fleet_years * 100, 2),
        "date_range": [str(drives["first_date"].min()), str(drives["last_date"].max())],
        "afr_by_manufacturer": afr.round(3).to_dict(orient="records"),
        "manufacturer_test": test_manufacturer_difference(afr),
        "reallocated_sectors_test": test_reallocated_sectors(drives),
        **survival_curves(drives),
    }
    (REPORTS / "stats_results.json").write_text(json.dumps(results, indent=2, default=str))

    print(f"Fleet AFR: {results['fleet_afr_pct']}% across {results['drives']:,} drives")
    print(afr[["manufacturer", "drives", "failures", "afr_pct", "afr_low_pct", "afr_high_pct"]].round(2).to_string(index=False))
    m = results["manufacturer_test"]
    print(f"Manufacturers differ? chi2={m['chi2']}, p={m['p_value']:.2g}")
    r = results["reallocated_sectors_test"]
    print(f"Drives with reallocated sectors were {r['relative_risk']}x more likely to fail "
          f"(95% CI {r['relative_risk_ci95']}, p={r['p_value']:.2g})")


if __name__ == "__main__":
    main()
