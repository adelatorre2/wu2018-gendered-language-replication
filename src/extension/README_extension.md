# Extension: Alternative Models for Table 1

This extension redoes the “Table 1 idea” (most predictive words for female vs male referenced posts) using two alternative models:
- OLS / Linear Probability Model (LPM)
- Random Forest classifier

The extension reuses the **same upstream data** and **same train/test split** as the Wu (2018) Lasso scripts. The goal is to be apples-to-apples with the original pipeline.

## How to run
From the repo root:

```bash
python -m src.extension.fit_extension --model ols
python -m src.extension.fit_extension --model rf
python -m src.extension.fit_extension --model all
```

Optional RF tuning:

```bash
python -m src.extension.fit_extension --model rf --rf-max-features 1000 --rf-max-train 20000 --rf-max-test 10000
```

## Outputs
Outputs are written to:
- `output/extension/ols/`
  - `metrics.json`
  - `table1_ols.csv`
  - `table1_ols.tex`
- `output/extension/rf/`
  - `metrics.json`
  - `table1_rf.csv`
  - `table1_rf.tex`

Logs are written to:
- `output/logs/extension_ols.log`
- `output/logs/extension_rf.log`
- `output/logs/extension_all.log`

## Notes
- OLS/LPM predictions are not bounded in [0, 1]; we clip for threshold-based metrics and report raw min/max.
- Random Forests are trained on a **reduced feature set** (top-k by document frequency) and **subsampled rows** to keep the run feasible. See `NOTES_methodology.md` for justification and limitations.
