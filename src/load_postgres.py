"""Load the processed tables into PostgreSQL.

Works the same against the local Docker database and Azure Database for
PostgreSQL. Only DATABASE_URL changes (see .env.example).

Usage:
    python -m src.load_postgres
"""
import json

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import DATA_PROCESSED, DATABASE_URL, REPORTS

TABLES = {
    "drive_summary": DATA_PROCESSED / "drive_summary",
    "daily_fleet": DATA_PROCESSED / "daily_fleet",
    "afr_by_manufacturer": DATA_PROCESSED / "afr_by_manufacturer.parquet",
    "at_risk_drives": DATA_PROCESSED / "at_risk_drives.parquet",
    "budget_curve": DATA_PROCESSED / "budget_curve.parquet",
}
INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_drive_summary_model ON drive_summary (model)",
    "CREATE INDEX IF NOT EXISTS idx_drive_summary_maker ON drive_summary (manufacturer)",
    "CREATE INDEX IF NOT EXISTS idx_daily_fleet_date ON daily_fleet (date)",
]


def main() -> None:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    host = engine.url.host
    print(f"Loading into PostgreSQL at {host}")

    for table, path in TABLES.items():
        if not path.exists():
            print(f"  {table}: not found at {path}, skipping (run the earlier steps first)")
            continue
        df = pd.read_parquet(path)
        # PostgreSQL allows at most 65,535 values per statement, so size the batches to fit
        chunk = max(1, 60_000 // max(1, len(df.columns)))
        df.to_sql(table, engine, if_exists="replace", index=False, chunksize=chunk, method="multi")
        print(f"  {table}: {len(df):,} rows")

    # Small key-value table of headline results for the dashboard
    facts = {}
    for name in ("stats_results.json", "model_metrics.json"):
        path = REPORTS / name
        if path.exists():
            facts[name.replace(".json", "")] = json.loads(path.read_text())
    if facts:
        pd.DataFrame(
            [{"name": k, "payload": json.dumps(v)} for k, v in facts.items()]
        ).to_sql("project_results", engine, if_exists="replace", index=False)
        print(f"  project_results: {len(facts)} rows")

    with engine.begin() as conn:
        for statement in INDEXES:
            conn.execute(text(statement))
    print("Done")


if __name__ == "__main__":
    main()
