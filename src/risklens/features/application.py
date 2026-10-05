"""Application-level features from application_train.

Everything here is known at the time of application. TARGET is dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from risklens.config import ID_COL, TARGET_COL

# Sentinel used by Home Credit for "not employed / pensioner".
DAYS_EMPLOYED_SENTINEL: int = 365243


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def build_application_features(app: pd.DataFrame) -> pd.DataFrame:
    df = app.drop(columns=[TARGET_COL], errors="ignore").copy()

    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).astype(int)
    df.loc[df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL, "DAYS_EMPLOYED"] = np.nan

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["EMPLOYED_TO_AGE"] = _safe_div(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])

    df["CREDIT_TO_INCOME"] = _safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_TO_INCOME"] = _safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_TO_CREDIT"] = _safe_div(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    df["CREDIT_TO_GOODS"] = _safe_div(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])
    df["INCOME_PER_PERSON"] = _safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])

    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_MIN"] = ext.min(axis=1)
    df["EXT_SOURCE_MAX"] = ext.max(axis=1)
    df["EXT_SOURCE_NAN_COUNT"] = ext.isna().sum(axis=1)

    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)

    # Object columns stay as strings; downstream models decide how to encode them.
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].astype("string").fillna("MISSING").astype(object)

    return df.set_index(ID_COL)
