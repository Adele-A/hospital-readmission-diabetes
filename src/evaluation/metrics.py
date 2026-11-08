"""Discrimination and calibration metrics for the hold-out evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

EPS = 1e-6


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y, p))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Cox calibration: fit y ~ a + b * logit(p).

    Returns (intercept a, slope b). Ideal calibration is intercept 0, slope 1.
    Requires variation in p and both outcome classes; returns (nan, nan) otherwise.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2 or np.ptp(p) < EPS:
        return (float("nan"), float("nan"))
    # C=1e10 ~ unpenalised (penalty=None is deprecated in scikit-learn >= 1.8).
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(_logit(p).reshape(-1, 1), y)
    return (float(lr.intercept_[0]), float(lr.coef_[0][0]))


def calibration_bins(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Quantile-binned calibration curve: mean predicted vs observed per bin."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    # Quantile edges; de-duplicate for ties.
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = bins == b
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": b,
                "n": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()),
                "observed_rate": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)
