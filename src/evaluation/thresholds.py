"""Decision thresholds — chosen on the training CV, applied to the test set.

Thresholds must never be tuned on the test set. We take out-of-fold predicted
probabilities from the patient-grouped CV on train, derive the thresholds there,
then apply them unchanged to the held-out test predictions.

Three operating points:
  - ``top10`` / ``top20`` — flag the highest-risk 10% / 20% (risk-stratification
    use case): the 90th / 80th percentile of the train OOF probabilities.
  - ``sens50`` — the threshold achieving ~50% sensitivity on the train OOF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict


def oof_probabilities(estimator, x, y, groups, cv) -> np.ndarray:
    """Out-of-fold positive-class probabilities on the training split."""
    proba = cross_val_predict(
        estimator, x, y, groups=groups, cv=cv, method="predict_proba", n_jobs=1
    )
    return proba[:, 1]


def choose_thresholds(oof_p: np.ndarray, y: np.ndarray, target_sens: float = 0.50) -> dict:
    """Derive named thresholds from train out-of-fold probabilities."""
    thr = {
        "top10 (highest-risk 10%)": float(np.quantile(oof_p, 0.90)),
        "top20 (highest-risk 20%)": float(np.quantile(oof_p, 0.80)),
    }
    # Highest threshold whose sensitivity on the OOF set is >= target.
    order = np.argsort(-oof_p)
    y_sorted = np.asarray(y)[order]
    p_sorted = oof_p[order]
    total_pos = y_sorted.sum()
    cum_tp = np.cumsum(y_sorted)
    sens = cum_tp / total_pos
    reach = np.searchsorted(sens, target_sens)
    reach = min(reach, len(p_sorted) - 1)
    thr[f"sens~{int(target_sens * 100)}%"] = float(p_sorted[reach])
    return thr


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Confusion-matrix-derived metrics at a probability threshold (on test)."""
    y = np.asarray(y)
    flag = (np.asarray(p) >= threshold).astype(int)
    tp = int(((flag == 1) & (y == 1)).sum())
    fp = int(((flag == 1) & (y == 0)).sum())
    tn = int(((flag == 0) & (y == 0)).sum())
    fn = int(((flag == 0) & (y == 1)).sum())
    n = len(y)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {
        "threshold": round(threshold, 4),
        "flagged": tp + fp,
        "flagged_per_1000": round(1000.0 * (tp + fp) / n, 1),
        "sensitivity": round(sens, 3),
        "specificity": round(spec, 3),
        "ppv": round(ppv, 3),
        "npv": round(npv, 3),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def threshold_table(y: np.ndarray, p: np.ndarray, thresholds: dict) -> pd.DataFrame:
    rows = []
    for name, thr in thresholds.items():
        m = threshold_metrics(y, p, thr)
        m = {"operating_point": name, **m}
        rows.append(m)
    return pd.DataFrame(rows)
