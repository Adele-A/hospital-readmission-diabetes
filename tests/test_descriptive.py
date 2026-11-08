"""Stage 3 descriptive-analysis helpers, tested without real data."""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.data.cohort import build_cohort_table
from src.data.load import parse_ids_mapping
from src.eda.descriptive import load_cohort_df, rate_by, wilson_ci

FIX = Path(__file__).parent / "fixtures"
CFG = {
    "cohort": {
        "death_hospice_discharge_ids": [11, 13, 14, 19, 20, 21],
        "drop_gender_unknown": True,
        "positive_label": "<30",
    }
}


def test_wilson_ci_symmetric_case():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi
    # Known Wilson 95% interval for 50/100 is about (0.404, 0.596).
    assert round(lo, 3) == pytest.approx(0.404, abs=0.003)
    assert round(hi, 3) == pytest.approx(0.596, abs=0.003)


def test_wilson_ci_degenerate():
    import math

    lo0, hi0 = wilson_ci(0, 0)
    assert math.isnan(lo0) and math.isnan(hi0)  # n == 0 -> nan, no crash
    lo, hi = wilson_ci(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert 0 < hi < 1


def test_rate_by_counts_and_ci():
    df = pd.DataFrame({"grp": ["a", "a", "a", "a", "b", "b"], "y": [1, 0, 0, 0, 1, 1]})
    out = rate_by(df, "grp").set_index("category")
    assert out.loc["a", "n"] == 4
    assert out.loc["a", "n_readmit_lt30"] == 1
    assert out.loc["a", "rate_pct"] == 25.0
    assert out.loc["b", "rate_pct"] == 100.0
    # CI low/high are within [0, 100].
    assert 0 <= out.loc["a", "ci95_low_pct"] <= out.loc["a", "ci95_high_pct"] <= 100


def test_rate_by_respects_order():
    df = pd.DataFrame({"g": ["x", "y", "z", "x"], "y": [1, 0, 1, 0]})
    out = rate_by(df, "g", order=["z", "y", "x"])
    assert list(out["category"]) == ["z", "y", "x"]


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
    build_cohort_table(c, CFG)
    yield c
    c.close()


def test_load_cohort_df_derived_columns(con):
    df = load_cohort_df(con)
    # Surviving fixture encounters: 1 (diag_1 250.01 -> diabetes), 2 (414 -> circulatory),
    # 5 (715 -> musculoskeletal), 7 (786 -> respiratory).
    groups = dict(zip(df["encounter_id"].astype(int), df["diag_1_group"], strict=True))
    assert groups[1] == "diabetes"
    assert groups[2] == "circulatory"
    assert groups[5] == "musculoskeletal"
    assert groups[7] == "respiratory"
    # Race "?" becomes Missing (encounter 5).
    race = dict(zip(df["encounter_id"].astype(int), df["race"], strict=True))
    assert race[5] == "Missing"
    # Prior inpatient bucket exists.
    assert "number_inpatient_grp" in df.columns
