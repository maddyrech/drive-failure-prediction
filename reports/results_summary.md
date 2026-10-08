# Results summary

- Data: 310,999 data drives, 76,842,879 daily records, 2024-01-01 to 2024-09-30
- Fleet annualised failure rate: 1.68% (3,530 failures)
- Failure rate by manufacturer: HGST 2.98%, Seagate 1.99%, Toshiba 1.23%, WDC 0.72%
- Manufacturers differ (exposure-adjusted chi-square): chi2 = 602.72, p = 2.6e-130
- Drives that reallocated a sector failed 29.4x as often (95% CI 27.5-31.3x, p < 1e-300)
- Model chosen on validation weeks: Gradient boosting. On untouched test weeks from 2024-06-17: PR-AUC 0.1644 vs 0.0536 for ranking by error count
- Checking the riskiest 1% of drives each week: model catches 53.58% of drives that fail within 30 days (9.0% of checks find one); ranking by error count catches 35.39% (5.95%)
- Simple rule (any sector errors): checks 5.19% of drives, catches 71.7%. The model checking the same number catches 71.49%

## Verdict

The model beats the best non-ML baseline at a 1% weekly budget by 18.2 percentage points (1.51x the failures caught).

## Check every number before putting it on a CV
