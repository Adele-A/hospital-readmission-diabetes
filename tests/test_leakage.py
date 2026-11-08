"""Leakage guards for Stage 4 splits and preprocessing.

Checks:
  1. No patient appears in both train and test (GroupShuffleSplit).
  2. No patient appears in both train and validation of any CV fold.
  3. The preprocessor fit on train never sees test categories.
  4. Identifiers and the outcome are excluded from the feature frame.
"""

import numpy as np
import pandas as pd

from src.features.feature_defs import DROP_ALWAYS, build_feature_frame
from src.models.pipeline import build_preprocessor
from src.models.splits import cv_splitter, holdout_split

CFG = {
    "seed": 42,
    "split": {"test_size": 0.2, "n_cv_folds": 5, "group_column": "patient_nbr"},
}


def make_synth(n_patients: int = 120, seed: int = 0) -> pd.DataFrame:
    """Synthetic cohort-shaped frame: patients with 1-3 encounters each."""
    rng = np.random.default_rng(seed)
    rows = []
    eid = 1
    for p in range(1000, 1000 + n_patients):
        for _ in range(int(rng.integers(1, 4))):
            rows.append(
                {
                    "encounter_id": eid,
                    "patient_nbr": p,
                    "y": int(rng.random() < 0.3),
                    "readmitted": "<30",
                    "is_first_encounter": True,
                    "weight": "?",
                    "time_in_hospital": int(rng.integers(1, 14)),
                    "num_lab_procedures": int(rng.integers(1, 100)),
                    "num_procedures": int(rng.integers(0, 6)),
                    "num_medications": int(rng.integers(1, 40)),
                    "number_outpatient": int(rng.integers(0, 5)),
                    "number_emergency": int(rng.integers(0, 5)),
                    "number_inpatient": int(rng.integers(0, 5)),
                    "number_diagnoses": int(rng.integers(1, 16)),
                    "diag_1": "250.01",
                    "diag_2": "414",
                    "diag_3": "?",
                    "race": rng.choice(["Caucasian", "AfricanAmerican", "?"]),
                    "gender": rng.choice(["Male", "Female"]),
                }
            )
            eid += 1
    return pd.DataFrame(rows)


def test_holdout_has_no_patient_overlap():
    df = make_synth()
    train_idx, test_idx = holdout_split(df, CFG)
    train_p = set(df.iloc[train_idx]["patient_nbr"])
    test_p = set(df.iloc[test_idx]["patient_nbr"])
    assert train_p & test_p == set()
    # Roughly a 20% patient hold-out.
    frac = len(test_p) / df["patient_nbr"].nunique()
    assert 0.1 < frac < 0.3


def test_cv_folds_have_no_group_overlap():
    df = make_synth()
    train_idx, _ = holdout_split(df, CFG)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    skf = cv_splitter(CFG)
    for f_tr, f_va in skf.split(train_df, train_df["y"], groups=train_df["patient_nbr"]):
        g_tr = set(train_df.iloc[f_tr]["patient_nbr"])
        g_va = set(train_df.iloc[f_va]["patient_nbr"])
        assert g_tr & g_va == set()


def test_preprocessor_fit_does_not_see_test_categories():
    # Train categories {A, B}; a category 'ONLY_TEST' exists only in the test rows.
    train = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "cat": ["A", "B", "A", "B"]})
    test = pd.DataFrame({"num": [5.0, 6.0], "cat": ["A", "ONLY_TEST"]})
    pre = build_preprocessor(["num"], ["cat"], min_frequency=1)
    pre.fit(train)  # fit on train only
    ohe = pre.named_transformers_["cat"].named_steps["ohe"]
    learned = set(np.concatenate([c for c in ohe.categories_]))
    assert "ONLY_TEST" not in learned  # fit never saw the test-only category
    # Transforming unseen test categories must not raise.
    out = pre.transform(test)
    assert out.shape[0] == 2


def test_feature_frame_excludes_identifiers_and_outcome():
    df = make_synth(n_patients=10)
    x, numeric, categorical = build_feature_frame(df)
    for forbidden in DROP_ALWAYS:
        assert forbidden not in x.columns
    # Diagnosis groups replace the raw codes.
    assert "diag_1_group" in x.columns
    assert "diag_1" not in x.columns
    # Numeric features are present and numeric.
    assert set(numeric).issubset(x.columns)
    assert x[numeric].dtypes.map(lambda d: d.kind == "f").all()
