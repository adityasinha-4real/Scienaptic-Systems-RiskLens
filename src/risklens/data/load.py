"""Read-only loaders for the raw Home Credit tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from risklens.config import RAW_DIR

RAW_TABLES: tuple[str, ...] = (
    "application_train",
    "bureau",
    "bureau_balance",
)


def raw_path(table: str, raw_dir: Path = RAW_DIR) -> Path:
    path = raw_dir / f"{table}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw table {path}. Download the Home Credit Default Risk data "
            "from Kaggle into data/raw/ (see README)."
        )
    return path


def load_table(
    table: str,
    usecols: list[str] | None = None,
    raw_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    """Load a raw CSV. Raw files are never written to."""
    return pd.read_csv(raw_path(table, raw_dir), usecols=usecols, engine="pyarrow")


def check_raw(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Return the paths of the required raw tables, raising if any is missing."""
    return [raw_path(t, raw_dir) for t in RAW_TABLES]
