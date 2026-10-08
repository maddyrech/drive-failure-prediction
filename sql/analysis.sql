-- Fleet analysis queries. Run them all with: python -m src.run_sql
-- AFR (annualised failure rate) = failures per drive-year of running time, in %.

-- name: fleet_overview
SELECT
    COUNT(*)                                               AS drives,
    SUM(failed)                                            AS failures,
    SUM(drive_days)                                        AS drive_days,
    ROUND(100.0 * SUM(failed) / (SUM(drive_days) / 365.0), 2) AS afr_pct
FROM drive_summary;

-- name: afr_by_model
-- Models with at least 100 drives, worst first
SELECT
    model,
    manufacturer,
    MAX(capacity_tb)                                       AS capacity_tb,
    COUNT(*)                                               AS drives,
    SUM(failed)                                            AS failures,
    ROUND(100.0 * SUM(failed) / (SUM(drive_days) / 365.0), 2) AS afr_pct
FROM drive_summary
GROUP BY model, manufacturer
HAVING COUNT(*) >= 100
ORDER BY afr_pct DESC
LIMIT 15;

-- name: afr_by_age_band
-- Bands use each drive's age at the end of the period (an approximation)
WITH banded AS (
    SELECT
        failed,
        drive_days,
        CASE
            WHEN age_days_end < 365      THEN 1
            WHEN age_days_end < 2 * 365  THEN 2
            WHEN age_days_end < 3 * 365  THEN 3
            WHEN age_days_end < 5 * 365  THEN 4
            ELSE 5
        END AS band
    FROM drive_summary
    WHERE age_days_end IS NOT NULL
)
SELECT
    CASE band
        WHEN 1 THEN 'Under 1 year'
        WHEN 2 THEN '1-2 years'
        WHEN 3 THEN '2-3 years'
        WHEN 4 THEN '3-5 years'
        ELSE '5+ years'
    END                                                    AS age_band,
    COUNT(*)                                               AS drives,
    SUM(failed)                                            AS failures,
    ROUND(100.0 * SUM(failed) / (SUM(drive_days) / 365.0), 2) AS afr_pct
FROM banded
GROUP BY band
ORDER BY band;

-- name: afr_by_capacity
SELECT
    capacity_tb,
    COUNT(*)                                               AS drives,
    SUM(failed)                                            AS failures,
    ROUND(100.0 * SUM(failed) / (SUM(drive_days) / 365.0), 2) AS afr_pct
FROM drive_summary
GROUP BY capacity_tb
HAVING COUNT(*) >= 100
ORDER BY capacity_tb;

-- name: warning_signs
-- How often drives failed, split by whether they ever showed sector errors
SELECT
    CASE
        WHEN COALESCE(max_reallocated_sectors, 0) > 0
          OR COALESCE(max_pending_sectors, 0) > 0 THEN 'Showed sector errors'
        ELSE 'No sector errors'
    END                                                    AS drive_group,
    COUNT(*)                                               AS drives,
    SUM(failed)                                            AS failures,
    ROUND(100.0 * AVG(failed), 2)                          AS pct_failed
FROM drive_summary
GROUP BY 1
ORDER BY pct_failed DESC;

-- name: weekly_failures
SELECT
    DATE_TRUNC('week', date)::date                         AS week,
    SUM(failures)                                          AS failures,
    ROUND(SUM(drives)::numeric / COUNT(DISTINCT date))     AS avg_drives_running
FROM daily_fleet
GROUP BY 1
ORDER BY 1;

-- name: top_risk_drives
SELECT
    risk_rank,
    serial_number,
    model,
    ROUND((age_days / 365.0)::numeric, 1)                  AS age_years,
    reallocated_sectors,
    pending_sectors,
    ROUND(risk_score::numeric, 3)                          AS risk_score
FROM at_risk_drives
ORDER BY risk_rank
LIMIT 10;
