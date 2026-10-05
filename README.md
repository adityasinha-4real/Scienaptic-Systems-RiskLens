# RiskLens

Credit underwriting engine on the Home Credit Default Risk data.
Phase 1: application scorecard (optbinning WoE logistic regression) with a LightGBM challenger.

## Setup (Windows)

```sh
py install 3.11
py -V:3.11 -m venv .venv
mingw32-make install        # or `make install` if GNU make is on PATH
```

Data: put a Kaggle API token at `~/.kaggle/kaggle.json`, accept the competition rules at
https://www.kaggle.com/competitions/home-credit-default-risk/rules, then `make download`.
Alternatively unzip the CSVs into `data/raw/` by hand. Raw files are never modified.

## Pipeline

| Target          | Output                                                                  |
|-----------------|-------------------------------------------------------------------------|
| `make data`     | `data/processed/split.parquet`: stratified 70/15/15 split, seed 42       |
| `make features` | `data/processed/features.parquet`: application + bureau aggregates      |
| `make train`    | `reports/phase1_metrics.json`, `scorecard_table.csv`, `shap_summary.png` |
| `make test`     | pytest: leakage, WoE monotonicity, score scaling and range              |

Scorecard scaling: 600 points at good:bad odds of 50:1, PDO 20. Features are kept when
IV is in [0.02, 0.5], then pruned for WoE correlation (>0.7) and coefficient sign.
