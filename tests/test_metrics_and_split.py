from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risklens.config import ID_COL, SPLIT_FRACTIONS, TARGET_COL
from risklens.data.split import make_split
from risklens.metrics import gini, ks, psi


def test_psi_zero_for_identical_and_large_for_shift() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=10_000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)
    assert psi(x, x + 1.0) > 0.25


def test_gini_and_ks_perfect_and_random() -> None:
    y = np.array([0, 0, 1, 1])
    assert gini(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert ks(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert gini(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)


def test_make_split_is_stratified_and_deterministic() -> None:
    rng = np.random.default_rng(0)
    labels = pd.DataFrame({ID_COL: np.arange(20_000), TARGET_COL: rng.binomial(1, 0.08, 20_000)})
    a, b = make_split(labels), make_split(labels)
    pd.testing.assert_frame_equal(a, b)
    shares = a["split"].value_counts(normalize=True)
    for name, frac in zip(("train", "val", "test"), SPLIT_FRACTIONS):
        assert shares[name] == pytest.approx(frac, abs=0.002)
    rates = a.groupby("split")[TARGET_COL].mean()
    assert rates.max() - rates.min() < 0.002


def test_saved_split_matches_spec(split: pd.DataFrame) -> None:
    assert split[ID_COL].is_unique
    shares = split["split"].value_counts(normalize=True)
    for name, frac in zip(("train", "val", "test"), SPLIT_FRACTIONS):
        assert shares[name] == pytest.approx(frac, abs=0.001)
    rates = split.groupby("split")[TARGET_COL].mean()
    assert rates.max() - rates.min() < 0.001
