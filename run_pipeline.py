"""Single entry point for the readmission pipeline.

Runs the stages that are implemented so far, in order:
    1. download  — fetch raw data into data/raw/ (skipped if already present)
    2. load      — build the DuckDB tables
    3. audit     — write reports/data_quality_report.md

Later stages (cohort, features, train, evaluate, export) are appended here as
they are built. Cross-platform: one command, no shell scripts.

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-download
"""

from __future__ import annotations

import argparse

from src.config import load_config
from src.data import cohort as cohort_mod
from src.data import download as download_mod
from src.data import load as load_mod
from src.data import quality_audit
from src.eda import descriptive as eda_mod


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the readmission pipeline.")
    parser.add_argument(
        "--skip-download", action="store_true", help="Assume data/raw/ is already populated."
    )
    args = parser.parse_args()

    cfg = load_config()

    if not args.skip_download:
        print("[1/4] Download")
        download_mod.download()

    print("[2/4] Load into DuckDB")
    con = load_mod.connect(cfg)
    try:
        load_mod.build_database(con, cfg)
    finally:
        con.close()

    print("[3/5] Data quality audit")
    quality_audit.main()

    print("[4/5] Cohort definition + exclusion flow")
    cohort_mod.main()

    print("[5/5] Exploratory analysis")
    eda_mod.main()

    print("Pipeline (Stages 1-3) complete.")


if __name__ == "__main__":
    main()
