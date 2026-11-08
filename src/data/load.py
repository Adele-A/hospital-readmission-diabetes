"""Load the raw CSVs into a DuckDB database and parse ``IDS_mapping.csv``.

``IDS_mapping.csv`` holds three stacked blocks (admission_type_id,
discharge_disposition_id, admission_source_id), each starting with a
``<id_col>,description`` header and separated by a blank ``,`` line. They are
flattened into one long table ``ids_mapping(mapping_type, id, description)``.

The encounters table is loaded with every column as VARCHAR (``all_varchar``) so
the dataset's ``"?"`` (missing) and ``"None"`` (a real category) are preserved
verbatim for the SQL audit; typing/NA handling happens later in the modelling code.

Run:  python -m src.data.load
"""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb
import pandas as pd

from src.config import load_config
from src.paths import repo_path

_ID_HEADERS = {"admission_type_id", "discharge_disposition_id", "admission_source_id"}


def parse_ids_mapping(path: str | Path) -> pd.DataFrame:
    """Parse the stacked IDS_mapping.csv into long form.

    Returns a DataFrame with columns ``mapping_type``, ``id`` (int), ``description`` (str).
    """
    rows: list[dict] = []
    current: str | None = None
    with open(path, encoding="utf-8", newline="") as fh:
        for fields in csv.reader(fh):
            if not fields or all(c.strip() == "" for c in fields):
                current = None  # blank separator line
                continue
            first = fields[0].strip()
            if first in _ID_HEADERS:
                current = first
                continue
            if current is None or first == "":
                continue
            rows.append(
                {
                    "mapping_type": current,
                    "id": int(first),
                    "description": fields[1].strip() if len(fields) > 1 else "",
                }
            )
    return pd.DataFrame(rows, columns=["mapping_type", "id", "description"])


def build_database(con: duckdb.DuckDBPyConnection, cfg: dict) -> None:
    """(Re)create ``encounters_raw`` and ``ids_mapping`` in the given connection."""
    raw_dir = repo_path(cfg["paths"]["raw_dir"])
    enc_csv = (raw_dir / cfg["data"]["encounters_csv"]).as_posix()
    ids_csv = raw_dir / cfg["data"]["ids_mapping_csv"]

    con.execute("DROP TABLE IF EXISTS encounters_raw;")
    con.execute(
        "CREATE TABLE encounters_raw AS "
        "SELECT * FROM read_csv(?, all_varchar=true, header=true);",
        [enc_csv],
    )

    ids_df = parse_ids_mapping(ids_csv)  # noqa: F841 - referenced by name in SQL below
    con.execute("DROP TABLE IF EXISTS ids_mapping;")
    con.execute("CREATE TABLE ids_mapping AS SELECT * FROM ids_df;")


def connect(cfg: dict | None = None) -> duckdb.DuckDBPyConnection:
    """Open (creating the parent dir) the project's DuckDB file."""
    cfg = cfg or load_config()
    db_path = repo_path(cfg["paths"]["duckdb_file"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def main() -> None:
    cfg = load_config()
    con = connect(cfg)
    try:
        build_database(con, cfg)
        n_enc = con.execute("SELECT count(*) FROM encounters_raw;").fetchone()[0]
        n_ids = con.execute("SELECT count(*) FROM ids_mapping;").fetchone()[0]
        print(f"Loaded encounters_raw: {n_enc:,} rows; ids_mapping: {n_ids} rows.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
