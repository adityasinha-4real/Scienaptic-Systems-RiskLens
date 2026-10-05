"""Discrimination and stability metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


def auc(y_true: np.ndarray, pd_hat: np.ndarray) -> float:
    return float(roc_auc_score(y_true, pd_hat))


def gini(y_true: np.ndarray, pd_hat: np.ndarray) -> float:
    return 2.0 * auc(y_true, pd_hat) - 1.0


def ks(y_true: np.ndarray, pd_hat: np.ndarray) -> float:
    """Max distance between the score CDFs of bads and goods."""
    y = np.asarray(y_true)
    s = np.asarray(pd_hat)
    return float(ks_2samp(s[y == 1], s[y == 0]).statistic)


def discrimination(y_true: np.ndarray, pd_hat: np.ndarray) -> dict[str, float]:
    return {
        "auc": round(auc(y_true, pd_hat), 5),
        "gini": round(gini(y_true, pd_hat), 5),
        "ks": round(ks(y_true, pd_hat), 5),
    }


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index of `actual` vs `expected`, bins from expected deciles."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, eps, None)
    a = np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))
