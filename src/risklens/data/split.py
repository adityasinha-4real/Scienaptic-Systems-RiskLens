"""Stratified 70/15/15 train/val/test split, persisted to data/processed/."""

from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from risklens.config import (
    ID_COL,
    PROCESSED_DIR,
    RANDOM_STATE,
    SPLIT_FRACTIONS,
    SPLIT_PATH,
    TARGET_COL,
)
from risklens.data.load import check_raw, load_table


def make_split(labels: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Assign each application to train/val/test, stratified on TARGET."""
    train_frac, val_frac, test_frac = SPLIT_FRACTIONS
    train, rest = train_test_split(
        labels,
        train_size=train_frac,
        stratify=labels[TARGET_COL],
        random_state=random_state,
    )
    val, test = train_test_split(
        rest,
        test_size=test_frac / (val_frac + test_frac),
        stratify=rest[TARGET_COL],
        random_state=random_state,
    )
    out = pd.concat(
        [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")]
    )
    return out.sort_values(ID_COL).reset_index(drop=True)


def main() -> None:
    check_raw()
    labels = load_table("application_train", usecols=[ID_COL, TARGET_COL])
    split = make_split(labels)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    split.to_parquet(SPLIT_PATH, index=False)
    summary = (
        split.groupby("split")[TARGET_COL]
        .agg(n="size", bad_rate="mean")
        .round(5)
        .to_dict(orient="index")
    )
    print(json.dumps({"split_path": str(SPLIT_PATH), "splits": summary}, indent=2))


if __name__ == "__main__":
    main()
