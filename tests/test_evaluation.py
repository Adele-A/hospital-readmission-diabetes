"""Tests for the Stage 6 evaluation helpers (metrics, bootstrap, thresholds)."""

import numpy as np

from src.evaluation.bootstrap import cluster_bootstrap_ci, paired_delta_ci
from src.evaluation.metrics import (
    brier,
    calibration_slope_intercept,
    pr_auc,
    roc_auc,
)
from src.evaluation.thresholds import choose_thresholds, threshold_metrics


def test_point_metrics_basic():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert 0.0 <= roc_auc(y, p) <= 1.0
    assert 0.0 <= pr_auc(y, p) <= 1.0
    assert brier(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0


def test_calibration_recovers_ideal_on_calibrated_data():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, size=40000)
    y = (rng.uniform(size=p.size) < p).astype(int)  # perfectly calibrated by construction
    intercept, slope = calibration_slope_intercept(y, p)
    assert abs(slope - 1.0) < 0.15
    assert abs(intercept) < 0.15


def test_calibration_degenerate_returns_nan():
    y = np.array([0, 1, 0, 1])
    p = np.full(4, 0.3)  # no variation
    intercept, slope = calibration_slope_intercept(y, p)
    assert np.isnan(intercept) and np.isnan(slope)


def test_cluster_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(300), 2)  # 300 patients, 2 encounters each
    p = rng.uniform(size=600)
    y = (rng.uniform(size=600) < p).astype(int)
    point, lo, hi = cluster_bootstrap_ci(roc_auc, y, p, groups, n_boot=200, seed=2)
    assert lo <= point <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_paired_delta_zero_for_identical_models():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(200), 2)
    p = rng.uniform(size=400)
    y = (rng.uniform(size=400) < 0.3).astype(int)
    point, lo, hi = paired_delta_ci(roc_auc, y, p, p, groups, n_boot=100, seed=4)
    assert point == 0.0
    assert lo == 0.0 and hi == 0.0


def test_choose_thresholds_quantiles():
    oof = np.linspace(0, 1, 1001)
    y = (oof > 0.8).astype(int)
    thr = choose_thresholds(oof, y, target_sens=0.5)
    assert abs(thr["top10 (highest-risk 10%)"] - np.quantile(oof, 0.90)) < 1e-9
    assert abs(thr["top20 (highest-risk 20%)"] - np.quantile(oof, 0.80)) < 1e-9
    assert "sens~50%" in thr


def test_threshold_metrics_confusion_counts():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    m = threshold_metrics(y, p, threshold=0.75)
    # flagged: p>=0.75 -> indices 0,1 (both positive)
    assert m["tp"] == 2 and m["fp"] == 0
    assert m["fn"] == 1 and m["tn"] == 3
    assert m["flagged"] == 2
    assert m["sensitivity"] == round(2 / 3, 3)
    assert m["specificity"] == 1.0
