"""Baseline strategies to compare the real models against.

Both are scikit-learn compatible classifiers so they run through the same CV and
metric code as the models:

  - ``PrevalenceClassifier``: predicts the constant train prevalence for every
    encounter (the "no-information" baseline; ROC-AUC = 0.5 by construction).
  - ``InpatientRateClassifier``: a simple, interpretable rule — bucket the number
    of prior inpatient visits (0, 1, 2, 3+) and predict the empirical 30-day
    readmission rate of that bucket learned on the training data. Captures "more
    past admissions -> higher risk" without any modelling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class PrevalenceClassifier(ClassifierMixin, BaseEstimator):
    """Predict the constant training-set prevalence as the positive probability."""

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.prevalence_ = float(y.mean())
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "prevalence_")
        n = len(X)
        p = np.full(n, self.prevalence_)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


class InpatientRateClassifier(ClassifierMixin, BaseEstimator):
    """Predict the empirical readmission rate of the prior-inpatient-visits bucket.

    Parameters
    ----------
    column:
        Name of the prior-inpatient-visits column in the input DataFrame.
    cap:
        Values >= ``cap`` share one ``cap+`` bucket.
    """

    def __init__(self, column: str = "number_inpatient", cap: int = 3):
        self.column = column
        self.cap = cap

    def _bucket(self, series: pd.Series) -> pd.Series:
        vals = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
        return vals.clip(upper=self.cap)

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("InpatientRateClassifier requires a pandas DataFrame X")
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.global_rate_ = float(y.mean())
        b = self._bucket(X[self.column])
        rates = pd.Series(y).groupby(b.values).mean()
        self.rates_ = rates.to_dict()
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "rates_")
        b = self._bucket(X[self.column])
        p = b.map(self.rates_).fillna(self.global_rate_).to_numpy(dtype=float)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
