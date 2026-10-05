"""LightGBM challenger: trains on the same feature matrix, writes metrics and SHAP summary."""

from __future__ import annotations

import json
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from risklens.config import (  # noqa: E402
    CHALLENGER_MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SHAP_SUMMARY_PATH,
)
from risklens.data.dataset import SplitData, categorical_columns, load_splits  # noqa: E402
from risklens.metrics import discrimination, psi  # noqa: E402

METRICS_PATH = REPORTS_DIR / "challenger_metrics.json"
SHAP_SAMPLE: int = 5000

PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.5,
    "reg_lambda": 1.0,
    "n_estimators": 3000,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}


def to_lgb_frame(X: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    out = X.copy()
    for c, cats in categories.items():
        out[c] = pd.Categorical(out[c], categories=cats)
    return out


def train_challenger(splits: dict[str, SplitData]) -> dict[str, Any]:
    train, val = splits["train"], splits["val"]
    cat_cols = categorical_columns(train.X)
    categories = {c: sorted(train.X[c].dropna().unique().tolist()) for c in cat_cols}
    frames = {name: to_lgb_frame(s.X, categories) for name, s in splits.items()}

    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(
        frames["train"],
        train.y,
        eval_set=[(frames["val"], val.y)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "categories": categories}, CHALLENGER_MODEL_PATH)

    pds = {name: model.predict_proba(f)[:, 1] for name, f in frames.items()}
    metrics: dict[str, Any] = {
        name: discrimination(splits[name].y.to_numpy(), p) for name, p in pds.items()
    }
    metrics["psi_train_test"] = round(psi(pds["train"], pds["test"]), 5)
    metrics["best_iteration"] = int(model.best_iteration_)

    sample = frames["test"].sample(min(SHAP_SAMPLE, len(frames["test"])), random_state=RANDOM_STATE)
    shap_values = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap.summary_plot(shap_values, sample, max_display=20, show=False)
    plt.title("LightGBM challenger: SHAP summary (test sample)")
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PATH, dpi=120)
    plt.close("all")

    mean_abs = np.abs(shap_values).mean(axis=0)
    top = pd.Series(mean_abs, index=sample.columns).sort_values(ascending=False).head(10)
    metrics["top_shap_features"] = top.round(5).to_dict()
    return metrics


def main() -> None:
    metrics = train_challenger(load_splits())
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
