from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pytest

from risklens.config import FEATURES_PATH, SCORECARD_MODEL_PATH, SPLIT_PATH
from risklens.data.dataset import SplitData, load_splits


def _require(path: Path, make_target: str) -> None:
    if not path.exists():
        pytest.fail(f"{path} missing; run `make {make_target}` first")


@pytest.fixture(scope="session")
def features() -> pd.DataFrame:
    _require(FEATURES_PATH, "features")
    return pd.read_parquet(FEATURES_PATH)


@pytest.fixture(scope="session")
def split() -> pd.DataFrame:
    _require(SPLIT_PATH, "data")
    return pd.read_parquet(SPLIT_PATH)


@pytest.fixture(scope="session")
def scorecard_bundle() -> dict[str, Any]:
    _require(SCORECARD_MODEL_PATH, "train")
    return joblib.load(SCORECARD_MODEL_PATH)


@pytest.fixture(scope="session")
def splits() -> dict[str, SplitData]:
    _require(FEATURES_PATH, "features")
    _require(SPLIT_PATH, "data")
    return load_splits()
