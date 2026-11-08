"""Stage 2 cohort logic tested on the synthetic fixture (no real data)."""

from pathlib import Path

import duckdb
import pytest

from src.data.cohort import build_cohort_table, exclusion_flow
from src.data.load import parse_ids_mapping

FIX = Path(__file__).parent / "fixtures"

# Fixture has 8 encounters / 7 patients. Discharge 11,13,14 -> 3 death/hospice
# encounters removed (patients 101, 102, 106). One gender Unknown/Invalid (enc 6,
# patient 104) removed next. Expected cohort: 4 encounters, 4 patients.
CFG = {
    "cohort": {
        "death_hospice_discharge_ids": [11, 13, 14, 19, 20, 21],
        "drop_gender_unknown": True,
        "positive_label": "<30",
    }
}


@pytest.fixture
def con():
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


def test_exclusion_flow_counts(con):
    flow = exclusion_flow(con, CFG).set_index("step")
    assert flow.iloc[0]["encounters_remaining"] == 8
    assert flow.iloc[0]["patients_remaining"] == 7
    # death/hospice removes 3 encounters (discharge 11, 13, 14)
    death_row = flow.iloc[1]
    assert death_row["encounters_remaining"] == 5
    assert death_row["encounters_removed"] == 3
    # gender Unknown/Invalid removes 1 more
    gender_row = flow.iloc[2]
    assert gender_row["encounters_remaining"] == 4
    assert gender_row["encounters_removed"] == 1


def test_cohort_table_target_and_first_flag(con):
    build_cohort_table(con, CFG)
    rows = con.execute(
        "SELECT encounter_id, patient_nbr, readmitted, y, is_first_encounter "
        "FROM cohort ORDER BY CAST(encounter_id AS BIGINT);"
    ).df()
    # Surviving encounters: 1 (pat100, NO), 2 (pat100, <30), 5 (pat103, NO), 7 (pat105, NO)
    assert list(rows["encounter_id"].astype(int)) == [1, 2, 5, 7]
    # Target: only encounter 2 is <30
    ymap = dict(zip(rows["encounter_id"].astype(int), rows["y"], strict=True))
    assert ymap == {1: 0, 2: 1, 5: 0, 7: 0}
    # First encounter per patient: enc 1 for patient 100 (not enc 2), enc 5, enc 7
    first = dict(zip(rows["encounter_id"].astype(int), rows["is_first_encounter"], strict=True))
    assert bool(first[1])
    assert not bool(first[2])


def test_no_excluded_dispositions_remain(con):
    build_cohort_table(con, CFG)
    bad = con.execute(
        "SELECT count(*) FROM cohort "
        "WHERE CAST(discharge_disposition_id AS INTEGER) IN (11,13,14,19,20,21) "
        "OR gender = 'Unknown/Invalid';"
    ).fetchone()[0]
    assert bad == 0
