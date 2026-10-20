"""
Integration tests for SQL queries executed on SQLite engine.
"""

from pathlib import Path
import re
import sqlite3
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "cleaned" / "churn_cleaned.csv"
SQL_PATH = ROOT_DIR / "sql" / "queries.sql"


@pytest.fixture(scope="module")
def db_conn():
    """In-memory SQLite database populated with cleaned churn dataset."""
    df = pd.read_csv(DATA_PATH)
    conn = sqlite3.connect(":memory:")
    df.to_sql("customers", conn, index=False)
    yield conn
    conn.close()


def test_sql_file_exists():
    assert SQL_PATH.exists()


def test_sql_execution(db_conn):
    """Verify all queries in queries.sql execute without error and return records."""
    with open(SQL_PATH, "r", encoding="utf-8") as f:
        sql_content = f.read()

    # Remove block comments and parse statements
    sql_clean = re.sub(r"/\*.*?\*/", "", sql_content, flags=re.DOTALL)
    statements = [s.strip() for s in sql_clean.split(";") if s.strip()]
    assert len(statements) >= 5

    for i, stmt in enumerate(statements, 1):
        lines = [l for l in stmt.splitlines() if not l.strip().startswith("--")]
        executable_sql = "\n".join(lines).strip()
        if not executable_sql:
            continue
        df_res = pd.read_sql_query(executable_sql, db_conn)
        assert not df_res.empty, f"Query {i} returned empty DataFrame"
        assert len(df_res) > 0
