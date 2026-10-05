"""Build data/processed/features.parquet: application + bureau features per SK_ID_CURR."""

from __future__ import annotations

import json

import pandas as pd

from risklens.config import FEATURES_PATH, ID_COL, PROCESSED_DIR, TARGET_COL
from risklens.data.load import check_raw, load_table
from risklens.features.application import build_application_features
from risklens.features.bureau import add_bureau_to_applications, build_bureau_features

BUREAU_COLS: list[str] = [
    "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE",
    "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT", "AMT_CREDIT_MAX_OVERDUE",
    "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT",
    "AMT_CREDIT_SUM_OVERDUE", "CREDIT_TYPE", "DAYS_CREDIT_UPDATE", "AMT_ANNUITY",
]


def build_features() -> pd.DataFrame:
    check_raw()
    app = load_table("application_train")
    bureau = load_table("bureau", usecols=BUREAU_COLS)
    bureau_balance = load_table("bureau_balance")

    app_feats = build_application_features(app)
    bureau_feats = build_bureau_features(bureau, bureau_balance)
    features = add_bureau_to_applications(app_feats, bureau_feats)

    if TARGET_COL in features.columns:
        raise RuntimeError("TARGET leaked into the feature matrix")
    return features.reset_index()


def main() -> None:
    features = build_features()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PATH, index=False)
    bureau_cols = [c for c in features.columns if c.startswith(("BUREAU_", "BB_"))]
    print(
        json.dumps(
            {
                "features_path": str(FEATURES_PATH),
                "rows": len(features),
                "columns": features.shape[1] - 1,
                "bureau_feature_columns": len(bureau_cols),
                "unique_ids": int(features[ID_COL].nunique()),
                "has_target": TARGET_COL in features.columns,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
