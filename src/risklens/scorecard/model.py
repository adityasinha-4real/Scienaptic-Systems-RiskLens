"""WoE-binned logistic regression scorecard (primary model) via optbinning.

Pipeline:
1. Optimal WoE binning of every feature on train, monotonic trend enforced for numerics.
2. Keep features with IV in [IV_MIN, IV_MAX].
3. Drop the lower-IV feature from any pair with |corr(WoE)| above a threshold.
4. Drop features whose logistic coefficient has the counter-intuitive sign.
5. Fit the final optbinning Scorecard with PDO/odds points scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from optbinning import BinningProcess, Scorecard
from sklearn.linear_model import LogisticRegression

from risklens.config import BASE_ODDS, BASE_SCORE, IV_MAX, IV_MIN, PDO, RANDOM_STATE

MAX_WOE_CORR: float = 0.7
MAX_PREBINS: int = 20


@dataclass
class SelectionLog:
    iv: dict[str, float] = field(default_factory=dict)
    iv_selected: list[str] = field(default_factory=list)
    dropped_correlation: list[str] = field(default_factory=list)
    dropped_sign: list[str] = field(default_factory=list)
    final: list[str] = field(default_factory=list)


def _binning_process(variables: list[str], categorical: list[str], select_iv: bool) -> BinningProcess:
    fit_params = {
        v: {"monotonic_trend": "auto_asc_desc"} for v in variables if v not in categorical
    }
    return BinningProcess(
        variable_names=variables,
        categorical_variables=[c for c in categorical if c in variables],
        max_n_prebins=MAX_PREBINS,
        selection_criteria={"iv": {"min": IV_MIN, "max": IV_MAX}} if select_iv else None,
        binning_fit_params=fit_params,
        n_jobs=-1,
    )


def _logreg() -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)


def _drop_correlated(woe: pd.DataFrame, iv: dict[str, float], threshold: float) -> list[str]:
    """Greedy: walk features by descending IV, keep one if not too correlated with kept ones."""
    order = sorted(woe.columns, key=lambda c: iv[c], reverse=True)
    corr = woe[order].corr().abs()
    kept: list[str] = []
    for c in order:
        if all(corr.loc[c, k] <= threshold for k in kept):
            kept.append(c)
    return kept


def _drop_wrong_sign(woe: pd.DataFrame, y: pd.Series, iv: dict[str, float]) -> list[str]:
    """Iteratively remove the lowest-IV feature whose coefficient sign disagrees with its
    univariate relationship to the target (WoE is monotone in risk by construction)."""
    kept = list(woe.columns)
    expected = {c: np.sign(np.corrcoef(woe[c], y)[0, 1]) for c in kept}
    while True:
        lr = _logreg().fit(woe[kept], y)
        wrong = [c for c, b in zip(kept, lr.coef_[0]) if np.sign(b) != expected[c]]
        if not wrong:
            return kept
        kept.remove(min(wrong, key=lambda c: iv[c]))


def select_features(X: pd.DataFrame, y: pd.Series, categorical: list[str]) -> SelectionLog:
    log = SelectionLog()
    variables = list(X.columns)
    bp = _binning_process(variables, categorical, select_iv=True).fit(X, y)
    summary = bp.summary().set_index("name")
    log.iv = summary["iv"].astype(float).round(5).to_dict()
    log.iv_selected = [v for v in variables if bool(summary.loc[v, "selected"])]

    woe = bp.transform(X, metric="woe", metric_missing="empirical", metric_special="empirical")
    woe = woe[log.iv_selected]
    uncorrelated = _drop_correlated(woe, log.iv, MAX_WOE_CORR)
    log.dropped_correlation = [v for v in log.iv_selected if v not in uncorrelated]

    signed = _drop_wrong_sign(woe[uncorrelated], y, log.iv)
    log.dropped_sign = [v for v in uncorrelated if v not in signed]
    log.final = sorted(signed, key=lambda c: log.iv[c], reverse=True)
    return log


def fit_scorecard(X: pd.DataFrame, y: pd.Series, variables: list[str], categorical: list[str]) -> Scorecard:
    scorecard = Scorecard(
        binning_process=_binning_process(variables, categorical, select_iv=False),
        estimator=_logreg(),
        scaling_method="pdo_odds",
        scaling_method_params={"pdo": PDO, "odds": BASE_ODDS, "scorecard_points": BASE_SCORE},
    )
    scorecard.fit(X[variables], y, metric_missing="empirical", metric_special="empirical")
    return scorecard


def predict_pd(scorecard: Scorecard, X: pd.DataFrame) -> np.ndarray:
    return scorecard.predict_proba(X)[:, 1]


def score(scorecard: Scorecard, X: pd.DataFrame) -> np.ndarray:
    return scorecard.score(X)
