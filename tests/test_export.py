"""Tests for the Stage 7 Power BI export helpers (no models / no I/O)."""

import numpy as np
import pandas as pd

from src.export.powerbi import _count_bucket, _discharge_group, build_fact_encounters


def test_discharge_group_mapping():
    assert _discharge_group(1) == "Home / home health"
    assert _discharge_group(6) == "Home / home health"
    assert _discharge_group(3) == "SNF / ICF"
    assert _discharge_group(22) == "Transfer / other facility"
    assert _discharge_group(25) == "Other/Unknown"
    assert _discharge_group("?") == "Other/Unknown"


def test_count_bucket():
    assert _count_bucket(0, 3) == "0"
    assert _count_bucket(2, 3) == "2"
    assert _count_bucket(3, 3) == "3+"
    assert _count_bucket(9, 3) == "3+"


def _synth():
    return pd.DataFrame(
        {
            "encounter_id": [1, 2, 3],
            "split": ["train", "train", "test"],
            "age": ["[50-60)", "[60-70)", "[70-80)"],
            "gender": ["Female", "Male", "Female"],
            "race": ["Caucasian", "?", "AfricanAmerican"],
            "admission_type_id": ["1", "2", "1"],
            "discharge_disposition_id": ["1", "3", "22"],
            "diag_1": ["250.01", "414", "786"],
            "number_inpatient": ["0", "2", "5"],
            "A1Cresult": ["None", ">8", "Norm"],
            "change": ["No", "Ch", "No"],
            "diabetesMed": ["Yes", "Yes", "No"],
            "insulin": ["No", "Steady", "Up"],
            "time_in_hospital": ["3", "5", "2"],
            "number_emergency": ["0", "1", "0"],
            "number_diagnoses": ["9", "9", "6"],
            "y": [0, 1, 0],
        }
    )


def test_build_fact_encounters_predictions_only_on_test():
    df = _synth()
    test_prob = {"Gradient boosting": np.array([0.42])}  # one test row
    high_risk = np.array([True])
    fact = build_fact_encounters(df, test_prob, high_risk)

    assert len(fact) == 3
    # Train rows have no prediction; the test row does.
    assert fact.loc[fact["split"] == "train", "predicted_risk"].isna().all()
    assert fact.loc[fact["split"] == "test", "predicted_risk"].iloc[0] == 0.42
    assert fact.loc[fact["split"] == "test", "high_risk_top20"].iloc[0] == 1
    # Derived dimensions.
    assert fact["race"].tolist() == ["Caucasian", "Missing", "AfricanAmerican"]
    assert fact["discharge_group"].tolist() == [
        "Home / home health", "SNF / ICF", "Transfer / other facility",
    ]
    assert fact["diag_1_group"].tolist() == ["diabetes", "circulatory", "respiratory"]
    assert fact["prior_inpatient_grp"].tolist() == ["0", "2", "3+"]
    assert fact["readmitted_lt30"].tolist() == [0, 1, 0]
