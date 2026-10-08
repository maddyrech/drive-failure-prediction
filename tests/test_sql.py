"""Tests for the SQL helpers."""
import pandas as pd

from src.run_sql import load_queries, to_markdown


def test_every_query_is_found_and_named():
    queries = load_queries()
    assert {"fleet_overview", "afr_by_model", "top_risk_drives"} <= set(queries)
    assert all(q.upper().lstrip().startswith(("SELECT", "WITH", "--")) for q in queries.values())


def test_markdown_table_has_header_divider_and_rows():
    md = to_markdown(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})).splitlines()
    assert md[0] == "| a | b |"
    assert md[1] == "| --- | --- |"
    assert md[2:] == ["| 1 | x |", "| 2 | y |"]
