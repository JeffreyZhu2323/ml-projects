import argparse
from pathlib import Path

from google.cloud import bigquery

def run_sql_file(filename: str, project: str | None = None) -> Path:
    project_root = Path.cwd()              # <-- current project folder
    client = bigquery.Client(project=project)

    # sql/<name>.sql under the current project
    sql_path = project_root / "sql" / filename
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    with open(sql_path, "r", encoding="utf-8") as f:
        query = f.read()

    print(f"Running query from {sql_path} ...")
    df = client.query(query).to_dataframe()
    print(f"Query returned {len(df):,} rows.")

    out_dir = project_root / "reports" / "sql_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = sql_path.stem
    out_path = out_dir / f"{stem}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Run a SQL file from ./sql and save results to ./reports/sql_reports."
    )
    parser.add_argument(
        "name",
        help="Base name or filename of the SQL file (e.g. 01_label_distribution or 01_label_distribution.sql)",
    )
    parser.add_argument(
        "--project",
        "-p",
        help="GCP project ID (if omitted, BigQuery default credentials' project is used).",
    )
    args = parser.parse_args()

    name = args.name
    if not name.endswith(".sql"):
        name = f"{name}.sql"

    run_sql_file(name, project=args.project)


if __name__ == "__main__":
    main()