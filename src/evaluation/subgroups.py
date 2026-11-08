"""Subgroup performance — a robustness check, not a full fairness audit.

Reports discrimination (ROC-AUC) and calibration (intercept / slope) within age,
sex, and race subgroups on the test set. Small or single-class subgroups are
reported with their size and left without an undefined metric. Wording is kept
careful: these describe stability of performance across groups, not a verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import calibration_slope_intercept, roc_auc

MIN_SUBGROUP = 100


def subgroup_table(
    df_test: pd.DataFrame, y: np.ndarray, p: np.ndarray, column: str
) -> pd.DataFrame:
    """Per-category n, positive rate, ROC-AUC, calibration intercept/slope."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    rows = []
    for level, idx in df_test.groupby(column, dropna=False).groups.items():
        pos = df_test.index.get_indexer(idx)
        yy, pp = y[pos], p[pos]
        n = len(yy)
        auc = roc_auc(yy, pp) if (n >= MIN_SUBGROUP and len(np.unique(yy)) == 2) else np.nan
        cal_int, cal_slope = (
            calibration_slope_intercept(yy, pp)
            if (n >= MIN_SUBGROUP and len(np.unique(yy)) == 2)
            else (np.nan, np.nan)
        )
        rows.append(
            {
                "subgroup": level,
                "n": n,
                "positive_pct": round(100.0 * yy.mean(), 2),
                "roc_auc": round(auc, 4) if auc == auc else np.nan,
                "calib_intercept": round(cal_int, 3) if cal_int == cal_int else np.nan,
                "calib_slope": round(cal_slope, 3) if cal_slope == cal_slope else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
