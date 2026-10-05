"""Join the feature matrix with split labels for modelling."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from risklens.config import FEATURES_PATH, ID_COL, SPLIT_PATH, TARGET_COL


@dataclass(frozen=True)
class SplitData:
    X: pd.DataFrame
    y: pd.Series


def load_splits() -> dict[str, SplitData]:
    if not FEATURES_PATH.exists() or not SPLIT_PATH.exists():
        raise FileNotFoundError("Run `make data` and `make features` first.")
    features = pd.read_parquet(FEATURES_PATH).set_index(ID_COL)
    split = pd.read_parquet(SPLIT_PATH).set_index(ID_COL)
    if TARGET_COL in features.columns:
        raise RuntimeError("TARGET found in features.parquet")
    features = features.loc[split.index]
    return {
        name: SplitData(
            X=features.loc[split["split"] == name],
            y=split.loc[split["split"] == name, TARGET_COL],
        )
        for name in ("train", "val", "test")
    }


def categorical_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "string"]
