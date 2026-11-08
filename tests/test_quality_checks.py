"""Quality-audit checks run against the small synthetic fixture (no real data).

Keeps CI independent of the UCI download.
"""

from pathlib import Path

import duckdb
import pytest

from src.data.load import parse_ids_mapping
from src.data.quality_audit import (
    column_missingness,
    numeric_ranges,
    parse_named_queries,
)
from src.paths import repo_path

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def con():
    """In-memory DuckDB with encounters_raw + ids_mapping from the fixtures."""
    c = duckdb.connect()
    c.execute(
        "CREATE TABLE encounters_raw AS "
        "SELECT * FROM read_csv(?, all_varchar=true, header=true);",
        [(FIX / "diabetic_data_sample.csv").as_posix()],
    )
    ids_df = parse_ids_mapping(FIX / "ids_mapping_sample.csv")  # noqa: F841 - used in SQL
    c.execute("CREATE TABLE ids_mapping AS SELECT * FROM ids_df;")
    yield c
    c.close()


@pytest.fixture
def queries():
    text = repo_path("sql", "02_quality_checks.sql").read_text(encoding="utf-8")
    return parse_named_queries(text)


def test_row_and_patient_counts(con, queries):
    assert con.execute(queries["row_count"]).fetchone()[0] == 8
    assert con.execute(queries["unique_patients"]).fetchone()[0] == 7
    assert con.execute(queries["duplicate_encounter_ids"]).fetchone()[0] == 0


def test_target_distribution(con, queries):
    df = con.execute(queries["readmitted_distribution"]).df()
    counts = dict(zip(df["readmitted"], df["n"], strict=True))
    assert counts["<30"] == 2
    assert counts["NO"] == 5
    assert counts[">30"] == 1


def test_death_hospice_exclusion_count(con, queries):
    df = con.execute(queries["death_hospice_discharges"]).df()
    # Fixture has discharge ids 11, 13, 14 (one encounter each).
    assert int(df["n"].sum()) == 3
    assert set(df["discharge_disposition_id"]) == {11, 13, 14}


def test_none_is_preserved_as_category(con, queries):
    df = con.execute(queries["a1cresult_distribution"]).df()
    assert "None" in set(df["value"])  # not converted to NULL/missing


def test_missingness_counts(con):
    miss = column_missingness(con).set_index("column")["n_missing"].to_dict()
    assert miss["weight"] == 8  # fully missing in the fixture
    assert miss["race"] == 1
    assert miss["payer_code"] == 3
    assert "diag_1" not in miss  # no '?' in diag_1 in the fixture


def test_numeric_ranges_sane(con):
    r = numeric_ranges(con).set_index("feature")
    assert r.loc["time_in_hospital", "min"] == 1
    assert r.loc["time_in_hospital", "max"] == 6
    assert r.loc["num_procedures", "min"] == 0
