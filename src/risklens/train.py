"""Phase 1 training: scorecard + LightGBM challenger -> reports/phase1_metrics.json."""

from __future__ import annotations

import json
from typing import Any

from risklens.challenger.train import METRICS_PATH as CHALLENGER_METRICS_PATH
from risklens.challenger.train import train_challenger
from risklens.config import PHASE1_METRICS_PATH, REPORTS_DIR
from risklens.data.dataset import load_splits
from risklens.scorecard.train import METRICS_PATH as SCORECARD_METRICS_PATH
from risklens.scorecard.train import train_scorecard


def _headline(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "test_auc": m["test"]["auc"],
        "test_gini": m["test"]["gini"],
        "test_ks": m["test"]["ks"],
        "psi_train_test": m["psi_train_test"],
    }


def main() -> None:
    splits = load_splits()
    scorecard = train_scorecard(splits)
    SCORECARD_METRICS_PATH.write_text(json.dumps(scorecard, indent=2))
    challenger = train_challenger(splits)
    CHALLENGER_METRICS_PATH.write_text(json.dumps(challenger, indent=2))

    metrics = {
        "scorecard": {**_headline(scorecard), "n_features": len(scorecard["selection"]["final_features"])},
        "challenger": _headline(challenger),
        "psi_score_basis": "scorecard points (scorecard); predicted PD (challenger)",
        "split_sizes": {name: len(s.y) for name, s in splits.items()},
        "details": {
            "scorecard": str(SCORECARD_METRICS_PATH.relative_to(REPORTS_DIR.parent)),
            "challenger": str(CHALLENGER_METRICS_PATH.relative_to(REPORTS_DIR.parent)),
        },
    }
    PHASE1_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
