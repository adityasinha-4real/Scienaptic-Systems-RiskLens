"""Credit-bureau aggregations per applicant (SK_ID_CURR).

Days/months columns in bureau tables are relative to the application date (negative =
before). Any record dated after the application is dropped so no post-application
information can leak into features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from risklens.config import ID_COL

# bureau_balance STATUS -> DPD bucket. 0: current, 1: 1-30, 2: 31-60, 3: 61-90,
# 4: 91-120, 5: 120+ or written off. C (closed) and X (unknown) carry no DPD info.
DPD_BUCKET: dict[str, float] = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "C": np.nan, "X": np.nan
}
# Upper bound of days past due per bucket, for an approximate max-DPD-in-days feature.
DPD_BUCKET_DAYS: dict[int, int] = {0: 0, 1: 30, 2: 60, 3: 90, 4: 120, 5: 150}

CREDIT_TYPES: dict[str, str] = {
    "Consumer credit": "CONSUMER",
    "Credit card": "CARD",
    "Car loan": "CAR",
    "Mortgage": "MORTGAGE",
    "Microloan": "MICROLOAN",
}


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def filter_pre_application_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Keep bureau records that were known before the application date."""
    keep = (bureau["DAYS_CREDIT"] <= 0) & ~(bureau["DAYS_CREDIT_UPDATE"] > 0)
    out = bureau.loc[keep].copy()
    out.loc[out["DAYS_ENDDATE_FACT"] > 0, "DAYS_ENDDATE_FACT"] = np.nan
    return out


def aggregate_bureau_balance(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """Per-SK_ID_BUREAU delinquency history from monthly statuses."""
    bb = bureau_balance.loc[bureau_balance["MONTHS_BALANCE"] <= 0].copy()
    bb["DPD_BUCKET"] = bb["STATUS"].astype(str).map(DPD_BUCKET)
    bb["IS_LATE"] = (bb["DPD_BUCKET"] >= 1).astype(float)
    bb["IS_LATE_60"] = (bb["DPD_BUCKET"] >= 2).astype(float)
    recent = bb["MONTHS_BALANCE"] >= -12
    bb["DPD_BUCKET_12M"] = bb["DPD_BUCKET"].where(recent)
    bb["IS_LATE_12M"] = bb["IS_LATE"].where(recent)

    return bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS=("MONTHS_BALANCE", "size"),
        BB_MAX_DPD_BUCKET=("DPD_BUCKET", "max"),
        BB_MAX_DPD_BUCKET_12M=("DPD_BUCKET_12M", "max"),
        BB_LATE_MONTHS=("IS_LATE", "sum"),
        BB_LATE60_MONTHS=("IS_LATE_60", "sum"),
        BB_LATE_MONTHS_12M=("IS_LATE_12M", "sum"),
    )


def build_bureau_features(
    bureau: pd.DataFrame, bureau_balance: pd.DataFrame
) -> pd.DataFrame:
    b = filter_pre_application_bureau(bureau)
    b = b.join(aggregate_bureau_balance(bureau_balance), on="SK_ID_BUREAU")

    is_active = b["CREDIT_ACTIVE"] == "Active"
    b["IS_ACTIVE"] = is_active.astype(int)
    b["IS_CLOSED"] = (b["CREDIT_ACTIVE"] == "Closed").astype(int)
    b["IS_BAD_DEBT"] = b["CREDIT_ACTIVE"].isin(["Bad debt", "Sold"]).astype(int)
    b["IS_OVERDUE"] = (b["CREDIT_DAY_OVERDUE"] > 0).astype(int)
    b["HAS_BB"] = b["BB_MONTHS"].notna().astype(int)
    b["ACTIVE_CREDIT_SUM"] = b["AMT_CREDIT_SUM"].where(is_active)
    b["ACTIVE_DEBT_SUM"] = b["AMT_CREDIT_SUM_DEBT"].where(is_active)
    b["ACTIVE_OVERDUE_SUM"] = b["AMT_CREDIT_SUM_OVERDUE"].where(is_active)
    b["ACTIVE_LIMIT_SUM"] = b["AMT_CREDIT_SUM_LIMIT"].where(is_active)
    b["ACTIVE_ANNUITY"] = b["AMT_ANNUITY"].where(is_active)
    b["DAYS_CREDIT_ACTIVE"] = b["DAYS_CREDIT"].where(is_active)
    for raw, short in CREDIT_TYPES.items():
        b[f"TYPE_{short}"] = (b["CREDIT_TYPE"] == raw).astype(int)

    g = b.groupby(ID_COL)
    agg = g.agg(
        BUREAU_CREDIT_COUNT=("SK_ID_BUREAU", "size"),
        BUREAU_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        BUREAU_CLOSED_COUNT=("IS_CLOSED", "sum"),
        BUREAU_BAD_DEBT_COUNT=("IS_BAD_DEBT", "sum"),
        BUREAU_OVERDUE_COUNT=("IS_OVERDUE", "sum"),
        BUREAU_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
        BUREAU_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_MAX_OVERDUE_AMT=("AMT_CREDIT_MAX_OVERDUE", "max"),
        BUREAU_MAX_DAY_OVERDUE=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_PROLONG_SUM=("CNT_CREDIT_PROLONG", "sum"),
        BUREAU_ACTIVE_CREDIT_SUM=("ACTIVE_CREDIT_SUM", "sum"),
        BUREAU_ACTIVE_DEBT_SUM=("ACTIVE_DEBT_SUM", "sum"),
        BUREAU_ACTIVE_OVERDUE_SUM=("ACTIVE_OVERDUE_SUM", "sum"),
        BUREAU_ACTIVE_LIMIT_SUM=("ACTIVE_LIMIT_SUM", "sum"),
        BUREAU_ACTIVE_ANNUITY_SUM=("ACTIVE_ANNUITY", "sum"),
        BUREAU_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
        BUREAU_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BUREAU_DAYS_CREDIT_ACTIVE_MAX=("DAYS_CREDIT_ACTIVE", "max"),
        BUREAU_DAYS_ENDDATE_MAX=("DAYS_CREDIT_ENDDATE", "max"),
        BUREAU_DAYS_ENDDATE_FACT_MAX=("DAYS_ENDDATE_FACT", "max"),
        BUREAU_DAYS_UPDATE_MAX=("DAYS_CREDIT_UPDATE", "max"),
        BUREAU_HAS_BB_COUNT=("HAS_BB", "sum"),
        BB_MAX_DPD_BUCKET=("BB_MAX_DPD_BUCKET", "max"),
        BB_MAX_DPD_BUCKET_12M=("BB_MAX_DPD_BUCKET_12M", "max"),
        BB_LATE_MONTHS_SUM=("BB_LATE_MONTHS", "sum"),
        BB_LATE60_MONTHS_SUM=("BB_LATE60_MONTHS", "sum"),
        BB_LATE_MONTHS_12M_SUM=("BB_LATE_MONTHS_12M", "sum"),
        BB_MONTHS_SUM=("BB_MONTHS", "sum"),
        **{f"BUREAU_{s}_COUNT": (f"TYPE_{s}", "sum") for s in CREDIT_TYPES.values()},
    )

    agg["BUREAU_ACTIVE_RATIO"] = _safe_div(agg["BUREAU_ACTIVE_COUNT"], agg["BUREAU_CREDIT_COUNT"])
    agg["BUREAU_DEBT_TO_CREDIT"] = _safe_div(agg["BUREAU_DEBT_SUM"], agg["BUREAU_CREDIT_SUM"])
    agg["BUREAU_ACTIVE_DEBT_TO_CREDIT"] = _safe_div(
        agg["BUREAU_ACTIVE_DEBT_SUM"], agg["BUREAU_ACTIVE_CREDIT_SUM"]
    )
    agg["BUREAU_OVERDUE_TO_DEBT"] = _safe_div(agg["BUREAU_OVERDUE_SUM"], agg["BUREAU_DEBT_SUM"])
    agg["BUREAU_ACTIVE_UTILISATION"] = _safe_div(
        agg["BUREAU_ACTIVE_DEBT_SUM"],
        agg["BUREAU_ACTIVE_DEBT_SUM"] + agg["BUREAU_ACTIVE_LIMIT_SUM"],
    )
    agg["BB_LATE_MONTH_RATIO"] = _safe_div(agg["BB_LATE_MONTHS_SUM"], agg["BB_MONTHS_SUM"])
    agg["BB_MAX_DPD_DAYS"] = agg["BB_MAX_DPD_BUCKET"].map(DPD_BUCKET_DAYS)

    # Bureau-history aggregates are undefined when the applicant had no bureau-balance rows.
    no_bb = agg["BUREAU_HAS_BB_COUNT"] == 0
    bb_cols = [c for c in agg.columns if c.startswith("BB_")]
    agg.loc[no_bb, bb_cols] = np.nan
    return agg


def add_bureau_to_applications(
    app_features: pd.DataFrame, bureau_features: pd.DataFrame
) -> pd.DataFrame:
    """Left-join bureau aggregates; applicants without bureau records get count 0."""
    out = app_features.join(bureau_features, how="left")
    out["BUREAU_CREDIT_COUNT"] = out["BUREAU_CREDIT_COUNT"].fillna(0)
    out["BUREAU_HAS_RECORD"] = (out["BUREAU_CREDIT_COUNT"] > 0).astype(int)
    return out
