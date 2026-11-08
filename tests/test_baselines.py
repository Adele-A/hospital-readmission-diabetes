"""Tests for the baseline strategies."""

import numpy as np
import pandas as pd

from src.models.baselines import InpatientRateClassifier, PrevalenceClassifier


def test_prevalence_predicts_constant_train_rate():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0])  # 40% positive
    clf = PrevalenceClassifier().fit(pd.DataFrame({"a": range(10)}), y)
    proba = clf.predict_proba(pd.DataFrame({"a": range(10)}))[:, 1]
    assert np.allclose(proba, 0.4)
    # predict uses 0.5 threshold -> all zeros here.
    assert clf.predict(pd.DataFrame({"a": range(10)})).sum() == 0


def test_inpatient_rate_uses_bucket_empirical_rate():
    df = pd.DataFrame(
        {
            "number_inpatient": [0, 0, 0, 0, 1, 1, 2, 3, 5],
            "other": list(range(9)),
        }
    )
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1])
    clf = InpatientRateClassifier().fit(df, y)
    # bucket 0 -> 1/4, bucket 1 -> 1/2, bucket 2 -> 1, bucket 3+ (3 and 5) -> 1
    probe = pd.DataFrame({"number_inpatient": [0, 1, 2, 4], "other": [0, 0, 0, 0]})
    proba = clf.predict_proba(probe)[:, 1]
    assert proba[0] == 0.25
    assert proba[1] == 0.5
    assert proba[2] == 1.0
    assert proba[3] == 1.0  # 4 falls in the capped 3+ bucket


def test_inpatient_rate_unseen_bucket_falls_back_to_global():
    df = pd.DataFrame({"number_inpatient": [0, 0, 1, 1]})
    y = np.array([0, 1, 0, 1])  # global 0.5; only buckets 0 and 1 seen
    clf = InpatientRateClassifier().fit(df, y)
    probe = pd.DataFrame({"number_inpatient": [3]})  # unseen bucket -> global rate
    assert clf.predict_proba(probe)[0, 1] == 0.5
