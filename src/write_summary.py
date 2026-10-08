"""Collect the headline numbers into reports/results_summary.md.

This is the file to paste into the README's Results section, and the one
to use when updating your CV.

Usage:
    python -m src.write_summary
"""
import json

from src.config import REPORTS


def p_text(p: float) -> str:
    """Very small p-values underflow to 0; report them as a bound instead."""
    return "< 1e-300" if p == 0 else f"= {p:.2g}"


def recall_at(result: dict, budget_pct: float) -> dict:
    return next(b for b in result["budgets"] if b["budget_pct"] == budget_pct)


def main() -> None:
    s = json.loads((REPORTS / "stats_results.json").read_text())
    m = json.loads((REPORTS / "model_metrics.json").read_text())
    best_name = m["best_model"]
    best = m["models"][best_name]
    model_1 = recall_at(best, 1.0)
    rule_1 = recall_at(m["ranked_rule"], 1.0)
    rule, matched = m["simple_rule"], m["best_model_at_rule_rate"]
    mt, rs = s["manufacturer_test"], s["reallocated_sectors_test"]
    records = s["drive_days"]

    lines = [
        "# Results summary",
        "",
        f"- Data: {s['drives']:,} data drives, {records:,} daily records, {s['date_range'][0]} to {s['date_range'][1]}",
        f"- Fleet annualised failure rate: {s['fleet_afr_pct']}% ({s['failures']:,} failures)",
        "- Failure rate by manufacturer: "
        + ", ".join(f"{r['manufacturer']} {r['afr_pct']:.2f}%" for r in s["afr_by_manufacturer"]),
        f"- Manufacturers differ (exposure-adjusted chi-square): chi2 = {mt['chi2']}, p {p_text(mt['p_value'])}",
        f"- Drives that reallocated a sector failed {rs['relative_risk']}x as often "
        f"(95% CI {rs['relative_risk_ci95'][0]}-{rs['relative_risk_ci95'][1]}x, p {p_text(rs['p_value'])})",
        f"- Model chosen on validation weeks: {best_name}. On untouched test weeks from {m['test_start']}: "
        f"PR-AUC {best['pr_auc']} vs {m['ranked_rule']['pr_auc']} for ranking by error count",
        f"- Checking the riskiest 1% of drives each week: model catches {model_1['recall_pct']}% of drives "
        f"that fail within {m['horizon_days']} days ({model_1['precision_pct']}% of checks find one); "
        f"ranking by error count catches {rule_1['recall_pct']}% ({rule_1['precision_pct']}%)",
        f"- Simple rule (any sector errors): checks {rule['flag_rate_pct']}% of drives, catches {rule['recall_pct']}%. "
        f"The model checking the same number catches {matched['recall_pct']}%",
        "",
        "## Verdict",
        "",
    ]
    gain = model_1["recall_pct"] - rule_1["recall_pct"]
    if gain > 1:
        lines.append(
            f"The model beats the best non-ML baseline at a 1% weekly budget by {gain:.1f} percentage points "
            f"({model_1['recall_pct'] / max(rule_1['recall_pct'], 0.01):.2f}x the failures caught)."
        )
    elif gain >= -1:
        lines.append("The model and the error-count ranking perform about the same at a 1% weekly budget. "
                     "The simple, explainable rule is the sensible choice; the model adds little.")
    else:
        lines.append(f"Ranking by error count beats the model at a 1% weekly budget by {-gain:.1f} points. "
                     "Report this honestly: the simple rule is the better tool here.")
    lines += [
        "",
        "## Check every number before putting it on a CV",
    ]
    out = REPORTS / "results_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
