"""Project-wide paths and constants."""

from __future__ import annotations

from pathlib import Path

RANDOM_STATE: int = 42

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
MODELS_DIR: Path = PROJECT_ROOT / "models"

ID_COL: str = "SK_ID_CURR"
TARGET_COL: str = "TARGET"

SPLIT_PATH: Path = PROCESSED_DIR / "split.parquet"
FEATURES_PATH: Path = PROCESSED_DIR / "features.parquet"

SCORECARD_MODEL_PATH: Path = MODELS_DIR / "scorecard.joblib"
CHALLENGER_MODEL_PATH: Path = MODELS_DIR / "challenger_lgbm.joblib"

PHASE1_METRICS_PATH: Path = REPORTS_DIR / "phase1_metrics.json"
SCORECARD_TABLE_PATH: Path = REPORTS_DIR / "scorecard_table.csv"
SHAP_SUMMARY_PATH: Path = REPORTS_DIR / "shap_summary.png"

# Split proportions: train / val / test.
SPLIT_FRACTIONS: tuple[float, float, float] = (0.70, 0.15, 0.15)

# Scorecard scaling: base score 600 at good:bad odds of 50:1, 20 points to double the odds.
PDO: float = 20.0
BASE_SCORE: float = 600.0
BASE_ODDS: float = 50.0

# IV window for feature selection.
IV_MIN: float = 0.02
IV_MAX: float = 0.5
