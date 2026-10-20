"""
ChurnLens SQL Analytics Runner.
Executes all SQL queries from sql/queries.sql against the Telco Customer dataset
using an in-memory SQLite database, demonstrating the complete BI analysis workflow.
"""

from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned" / "churn_cleaned.csv"
SQL_PATH = BASE_DIR / "sql" / "queries.sql"


def run_queries(data_path=DATA_PATH, sql_path=SQL_PATH):
    print(f"Loading data from {data_path} into in-memory SQLite database...")
    df = pd.read_csv(data_path)
    conn = sqlite3.connect(":memory:")
    df.to_sql("customers", conn, index=False)

    with open(sql_path, "r", encoding="utf-8") as f:
        sql_content = f.read()

    # Parse individual queries
    statements = [stmt.strip() for stmt in sql_content.split(";") if stmt.strip()]

    results = []
    query_num = 0
    for stmt in statements:
        # Extract comment header if present
        lines = stmt.splitlines()
        title_lines = [l.strip().lstrip("-/* ").rstrip(" */") for l in lines if l.strip().startswith(("--", "/*", "*"))]
        title = title_lines[0] if title_lines else f"Query {query_num + 1}"

        # Clean query of pure block comments
        query_sql = "\n".join([l for l in lines if not l.strip().startswith(("/*", "*", "*/"))]).strip()
        if not query_sql:
            continue

        query_num += 1
        print(f"\n{'='*70}")
        print(f"📊 Query {query_num}: {title}")
        print(f"{'='*70}")

        try:
            res_df = pd.read_sql_query(stmt, conn)
            print(res_df.to_string(index=False))
            results.append((title, res_df))
        except Exception as e:
            print(f"❌ Error executing query: {e}")

    conn.close()
    return results


if __name__ == "__main__":
    run_queries()
