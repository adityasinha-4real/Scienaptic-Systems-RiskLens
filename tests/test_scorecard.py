"""Scorecard checks: WoE monotonicity, score range and scaling sanity."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from risklens.config import BASE_ODDS, BASE_SCORE, IV_MAX, IV_MIN, PDO, SCORECARD_TABLE_PATH
from risklens.data.dataset import SplitData

N_TOP: int = 5


def _woe_by_bin(scorecard: Any, variable: str) -> np.ndarray:
    table = scorecard.binning_process_.get_binned_variable(variable).binning_table.build()
    table = table.drop(index="Totals", errors="ignore")
    regular = table[~table["Bin"].astype(str).isin(["Special", "Missing"])]
    return regular["WoE"].astype(float).to_numpy()


def _top_features(bundle: dict[str, Any]) -> list[str]:
    bp = bundle["scorecard"].binning_process_
    iv = bp.summary().set_index("name")["iv"].astype(float)
    return iv.loc[bundle["variables"]].sort_values(ascending=False).index[:N_TOP].tolist()


def test_selected_features_respect_iv_window(scorecard_bundle: dict[str, Any]) -> None:
    iv = scorecard_bundle["scorecard"].binning_process_.summary().set_index("name")["iv"]
    selected = iv.loc[scorecard_bundle["variables"]].astype(float)
    assert len(selected) >= N_TOP
    assert selected.between(IV_MIN, IV_MAX).all(), selected[~selected.between(IV_MIN, IV_MAX)]


@pytest.mark.parametrize("rank", range(N_TOP))
def test_woe_monotonic_for_top_features(scorecard_bundle: dict[str, Any], rank: int) -> None:
    variable = _top_features(scorecard_bundle)[rank]
    woe = _woe_by_bin(scorecard_bundle["scorecard"], variable)
    assert len(woe) >= 2, f"{variable} has a single bin"
    diffs = np.diff(woe)
    assert (diffs >= -1e-9).all() or (diffs <= 1e-9).all(), f"{variable} WoE not monotonic: {woe}"


@pytest.fixture(scope="module")
def test_scores(scorecard_bundle: dict[str, Any], splits: dict[str, SplitData]) -> pd.DataFrame:
    sc, variables = scorecard_bundle["scorecard"], scorecard_bundle["variables"]
    X = splits["test"].X[variables]
    return pd.DataFrame(
        {"score": sc.score(X), "pd": sc.predict_proba(X)[:, 1], "y": splits["test"].y.to_numpy()}
    )


def test_score_range_sanity(test_scores: pd.DataFrame) -> None:
    s = test_scores["score"]
    assert np.isfinite(s).all()
    assert s.between(200, 1000).all(), (s.min(), s.max())
    assert s.std() > 10  # the scorecard actually separates applicants


def test_score_scaling_matches_pdo_and_base_odds(test_scores: pd.DataFrame) -> None:
    """score = BASE + PDO/ln2 * (ln(good:bad odds) - ln(BASE_ODDS))."""
    p = test_scores["pd"].to_numpy()
    expected = BASE_SCORE + PDO / np.log(2) * (np.log((1 - p) / p) - np.log(BASE_ODDS))
    np.testing.assert_allclose(test_scores["score"], expected, atol=1e-6)


def test_higher_score_means_lower_bad_rate(test_scores: pd.DataFrame) -> None:
    deciles = pd.qcut(test_scores["score"], 10, labels=False, duplicates="drop")
    bad_rate = test_scores.groupby(deciles)["y"].mean()
    assert bad_rate.iloc[0] > 3 * bad_rate.iloc[-1]
    assert (np.diff(bad_rate.to_numpy()) < 0.01).all(), bad_rate.round(4).tolist()


def test_scorecard_table_written() -> None:
    table = pd.read_csv(SCORECARD_TABLE_PATH)
    assert {"Variable", "Bin", "WoE", "Points"}.issubset(table.columns)
