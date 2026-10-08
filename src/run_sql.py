"""Run every query in sql/analysis.sql and save the results to
reports/sql_results.md (handy for the README and for interviews).

Usage:
    python -m src.run_sql
"""
import re

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import DATABASE_URL, REPORTS, ROOT


def load_queries() -> dict:
    sql = (ROOT / "sql" / "analysis.sql").read_text()
    parts = re.split(r"^-- name:\s*(\w+)\s*$", sql, flags=re.MULTILINE)
    return {name: body.strip() for name, body in zip(parts[1::2], parts[2::2])}


def to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def main() -> None:
    engine = create_engine(DATABASE_URL)
    sections = ["# SQL analysis results\n"]
    with engine.connect() as conn:
        for name, query in load_queries().items():
            df = pd.read_sql(text(query), conn)
            print(f"\n== {name} ==\n{df.to_string(index=False)}")
            sections.append(f"## {name.replace('_', ' ').capitalize()}\n\n{to_markdown(df)}\n")
    (REPORTS / "sql_results.md").write_text("\n".join(sections))
    print("\nSaved reports/sql_results.md")


if __name__ == "__main__":
    main()
