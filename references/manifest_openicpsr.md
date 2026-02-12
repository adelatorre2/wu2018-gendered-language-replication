# OpenICPSR replication package manifest (Wu 2018)

Source: OpenICPSR V1 (Wu, Alice H.), "Replication data for: Gendered Language on the Economics Job Market Rumors Forum"

Local path:
`data/raw/openicpsr_wu2018_replication-pkg/`

## Raw package contents (do not edit)
- `LICENSE.txt`: license terms (code under BSD-3-Clause; data/text under CC BY 4.0).
- `README.pdf`: upstream README + codebooks for the datasets.
- `gendered_posts.csv`: post-level dataset with gender labels and training/test splits. Large file; ignored in git.
- `X_word_count.npz`: SciPy sparse CSC matrix (rows are posts, columns are top-10K words).
- `keys_to_X.csv`: `(title_id, post_id)` keys in the exact row order of `X_word_count.npz`.
- `vocab10K.csv`: 10,000-word vocabulary list with Lasso coefficients/marginal effects.
- `trend_stats.csv`: monthly summary stats for Figure 1.
- `tables-figures.R`: upstream R script that builds Table 1, Table 2, and Figure 1.
- `lasso/lasso-logit-full-sample.py`: logistic Lasso on the full gender sample.
- `lasso/lasso-logit-pronoun-sample.py`: logistic Lasso on the pronoun sample (robustness check).
- `lasso/lasso-linear-pronoun-sample.py`: linear Lasso on the pronoun sample (appendix figures).

## Path assumptions (upstream)
- The Lasso scripts assume data are located in `../` relative to `lasso/` and write outputs to that same parent directory.
- `tables-figures.R` also uses `dir_data=".."`; run it with working directory set to `data/raw/openicpsr_wu2018_replication-pkg/lasso` so `..` resolves to the raw package root.

## Script-to-output mapping (repo pipeline)

### Lasso logit: full sample
Upstream script:
`data/raw/openicpsr_wu2018_replication-pkg/lasso/lasso-logit-full-sample.py`

Generated in raw package root:
- `coef_lasso_logit_full.txt`
- `ypred_train.txt`, `ypred_test0.txt`, `ypred_test1.txt`
- `i_keep_columns.txt`

Copied into this repo:
- `output/intermediate/lasso_full/`

### Lasso logit: pronoun sample
Upstream script:
`data/raw/openicpsr_wu2018_replication-pkg/lasso/lasso-logit-pronoun-sample.py`

Generated in raw package root:
- `coef_lasso_logit_pronoun.txt`
- `ypred_pronoun_train.txt`, `ypred_pronoun_test0.txt`, `ypred_pronoun_test1.txt`
- `i_keep_columns.txt`

Copied into this repo:
- `output/intermediate/lasso_pronoun/`

### Tables and Figure 1
Upstream script:
`data/raw/openicpsr_wu2018_replication-pkg/tables-figures.R`

Inputs:
- `vocab10K.csv`
- `trend_stats.csv`

Saved into this repo (via wrapper in `src/run_all.py`):
- `output/tables/table1.csv`
- `output/tables/table2.csv`
- `output/figures/figure1.pdf`

### Note on the linear Lasso script
`lasso/lasso-linear-pronoun-sample.py` is included in the raw package but is not run in the default pipeline because it uses a linear model for appendix figures and requires small fixes to run in modern scikit-learn.

## Repo output layout
- `output/intermediate/`: copied Lasso intermediates
- `output/tables/`: Table 1 and Table 2
- `output/figures/`: Figure 1
- `output/logs/`: logs from each pipeline step

## Local compatibility patch
- Patched deprecated pandas `as_matrix()` calls to `to_numpy()` in the upstream Lasso scripts for compatibility only; analysis logic is unchanged.
