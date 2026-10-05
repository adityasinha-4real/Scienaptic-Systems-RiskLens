"""Train the application scorecard and write reports/scorecard_metrics.json."""

from __future__ import annotations

import json
from typing import Any

import joblib

from risklens.config import (
    BASE_ODDS,
    BASE_SCORE,
    IV_MAX,
    IV_MIN,
    MODELS_DIR,
    PDO,
    REPORTS_DIR,
    SCORECARD_MODEL_PATH,
    SCORECARD_TABLE_PATH,
)
from risklens.data.dataset import SplitData, categorical_columns, load_splits
from risklens.metrics import discrimination, psi
from risklens.scorecard.model import fit_scorecard, predict_pd, score, select_features

METRICS_PATH = REPORTS_DIR / "scorecard_metrics.json"


def train_scorecard(splits: dict[str, SplitData]) -> dict[str, Any]:
    train = splits["train"]
    categorical = categorical_columns(train.X)

    log = select_features(train.X, train.y, categorical)
    scorecard = fit_scorecard(train.X, train.y, log.final, categorical)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scorecard": scorecard, "variables": log.final}, SCORECARD_MODEL_PATH)
    scorecard.table(style="detailed").to_csv(SCORECARD_TABLE_PATH, index=False)

    scores = {name: score(scorecard, s.X[log.final]) for name, s in splits.items()}
    metrics: dict[str, Any] = {
        name: discrimination(s.y.to_numpy(), predict_pd(scorecard, s.X[log.final]))
        for name, s in splits.items()
    }
    metrics["psi_train_test"] = round(psi(scores["train"], scores["test"]), 5)
    metrics["psi_train_val"] = round(psi(scores["train"], scores["val"]), 5)
    metrics["score_summary_test"] = {
        "min": round(float(scores["test"].min()), 2),
        "p50": round(float(sorted(scores["test"])[len(scores["test"]) // 2]), 2),
        "max": round(float(scores["test"].max()), 2),
    }
    metrics["scaling"] = {"pdo": PDO, "base_score": BASE_SCORE, "base_odds_good_bad": BASE_ODDS}
    metrics["selection"] = {
        "iv_window": [IV_MIN, IV_MAX],
        "n_candidates": len(log.iv),
        "n_iv_selected": len(log.iv_selected),
        "dropped_correlation": log.dropped_correlation,
        "dropped_sign": log.dropped_sign,
        "final_features": {v: log.iv[v] for v in log.final},
    }
    return metrics


def main() -> None:
    metrics = train_scorecard(load_splits())
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
