"""Feature leakage: no TARGET, no post-application bureau information."""

from __future__ import annotations

import numpy as np
import pandas as pd

from risklens.config import ID_COL, TARGET_COL
from risklens.features.application import build_application_features
from risklens.features.bureau import (
    aggregate_bureau_balance,
    filter_pre_application_bureau,
)


def test_no_target_column_in_features(features: pd.DataFrame) -> None:
    assert TARGET_COL not in features.columns
    assert not any("TARGET" in c.upper() for c in features.columns)


def test_features_one_row_per_applicant(features: pd.DataFrame, split: pd.DataFrame) -> None:
    assert features[ID_COL].is_unique
    assert set(features[ID_COL]) == set(split[ID_COL])


def test_no_feature_is_a_proxy_for_target(features: pd.DataFrame, split: pd.DataFrame) -> None:
    """A leaked label shows up as a near-perfect single-feature correlation."""
    df = features.merge(split[[ID_COL, TARGET_COL]], on=ID_COL)
    numeric = df.select_dtypes(include="number").drop(columns=[ID_COL, TARGET_COL])
    corr = numeric.corrwith(df[TARGET_COL]).abs()
    assert corr.max() < 0.5, corr.sort_values(ascending=False).head()


def test_application_builder_drops_target() -> None:
    app = pd.DataFrame(
        {
            ID_COL: [1, 2],
            TARGET_COL: [0, 1],
            "DAYS_EMPLOYED": [-100, 365243],
            "DAYS_BIRTH": [-10000, -20000],
            "AMT_CREDIT": [1000.0, 2000.0],
            "AMT_INCOME_TOTAL": [500.0, 0.0],
            "AMT_ANNUITY": [100.0, 200.0],
            "AMT_GOODS_PRICE": [900.0, 1800.0],
            "CNT_FAM_MEMBERS": [2.0, 1.0],
            "EXT_SOURCE_1": [0.5, np.nan],
            "EXT_SOURCE_2": [0.4, 0.6],
            "EXT_SOURCE_3": [np.nan, 0.2],
            "FLAG_DOCUMENT_3": [1, 0],
            "NAME_CONTRACT_TYPE": ["Cash loans", None],
        }
    )
    out = build_application_features(app)
    assert TARGET_COL not in out.columns
    assert out["DAYS_EMPLOYED_ANOM"].tolist() == [0, 1]
    assert np.isnan(out.loc[2, "DAYS_EMPLOYED"])
    assert np.isnan(out.loc[2, "CREDIT_TO_INCOME"])  # zero income -> undefined, not inf
    assert out.loc[2, "NAME_CONTRACT_TYPE"] == "MISSING"


def test_bureau_records_after_application_are_dropped() -> None:
    bureau = pd.DataFrame(
        {
            "DAYS_CREDIT": [-500, 10, -300],
            "DAYS_CREDIT_UPDATE": [-10, -5, 3],
            "DAYS_ENDDATE_FACT": [5.0, -1.0, np.nan],
        }
    )
    out = filter_pre_application_bureau(bureau)
    assert len(out) == 1
    assert (out["DAYS_CREDIT"] <= 0).all()
    assert out["DAYS_ENDDATE_FACT"].isna().all()  # future close date is post-application


def test_bureau_balance_ignores_future_months() -> None:
    bb = pd.DataFrame(
        {
            "SK_ID_BUREAU": [1, 1, 1],
            "MONTHS_BALANCE": [-2, -1, 1],
            "STATUS": ["0", "1", "5"],
        }
    )
    out = aggregate_bureau_balance(bb)
    assert out.loc[1, "BB_MONTHS"] == 2
    assert out.loc[1, "BB_MAX_DPD_BUCKET"] == 1
