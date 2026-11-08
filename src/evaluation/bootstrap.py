"""Cluster bootstrap by patient for 95% confidence intervals.

Encounters of the same patient are correlated, so a naive bootstrap over rows
would understate uncertainty. Here each bootstrap replicate resamples **patients**
(clusters) with replacement and takes all their encounters, preserving the
within-patient correlation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

StatFn = Callable[[np.ndarray, np.ndarray], float]


def _patient_index(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (unique_patients, list of row-index arrays per patient)."""
    uniq, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inv = inverse[order]
    boundaries = np.searchsorted(sorted_inv, np.arange(len(uniq)))
    idx_per_patient = np.split(order, boundaries[1:])
    return uniq, idx_per_patient


def cluster_bootstrap_ci(
    stat_fn: StatFn,
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) for ``stat_fn`` via patient bootstrap."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    groups = np.asarray(groups)
    point = stat_fn(y, p)

    _, idx_per_patient = _patient_index(groups)
    n_pat = len(idx_per_patient)
    rng = np.random.default_rng(seed)

    stats = []
    for _ in range(n_boot):
        chosen = rng.integers(0, n_pat, size=n_pat)
        rows = np.concatenate([idx_per_patient[c] for c in chosen])
        yb, pb = y[rows], p[rows]
        if len(np.unique(yb)) < 2:
            continue  # metric undefined for a single-class resample
        stats.append(stat_fn(yb, pb))

    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return (float(point), lo, hi)


def paired_delta_ci(
    stat_fn: StatFn,
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """95% CI for stat(model A) - stat(model B), same resampled patients each draw."""
    y = np.asarray(y)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    groups = np.asarray(groups)
    point = stat_fn(y, p_a) - stat_fn(y, p_b)

    _, idx_per_patient = _patient_index(groups)
    n_pat = len(idx_per_patient)
    rng = np.random.default_rng(seed)

    deltas = []
    for _ in range(n_boot):
        chosen = rng.integers(0, n_pat, size=n_pat)
        rows = np.concatenate([idx_per_patient[c] for c in chosen])
        yb = y[rows]
        if len(np.unique(yb)) < 2:
            continue
        deltas.append(stat_fn(yb, p_a[rows]) - stat_fn(yb, p_b[rows]))

    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return (float(point), lo, hi)
