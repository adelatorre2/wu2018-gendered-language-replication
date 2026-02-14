# Wu (2018) Gendered Language Replication

## How to reproduce (full pipeline)
1. Create the conda environment.

```bash
conda env create -f environment/environment.yml
conda activate wu2018
```

2. Install R and required R packages (ggplot2).

```bash
conda install -c conda-forge r-base r-ggplot2
```

3. Download the OpenICPSR replication package and unzip it into:
`data/raw/openicpsr_wu2018_replication-pkg/`

The folder should contain `gendered_posts.csv`, `X_word_count.npz`, `vocab10K.csv`, `trend_stats.csv`, `tables-figures.R`, and `lasso/`.

4. Run the full replication pipeline (Tables 1–2 and Figure 1).

```bash
python src/run_all.py
```

Outputs are written to:
- `output/intermediate/`
- `output/tables/`
- `output/figures/`
- `output/logs/`

## Extension (alternative models)
The extension re-does the “Table 1” idea using two alternative models:
- OLS / Linear Probability Model (LPM)
- Random Forest classifier (with feature reduction + subsampling for feasibility)

Run both:

```bash
python -m src.extension.fit_extension --model all
```

Or run them separately:

```bash
python -m src.extension.fit_extension --model ols
python -m src.extension.fit_extension --model rf
```

Outputs are written to:
- `output/extension/ols/` (metrics + `table1_ols.csv` / `table1_ols.tex`)
- `output/extension/rf/` (metrics + `table1_rf.csv` / `table1_rf.tex`)

Notes:
- The extension **reuses the same data, feature construction, and split logic** as the replication package.
- RF is trained on the top-K most frequent words and subsamples rows; adjust via `--rf-max-features`, `--rf-max-train`, `--rf-max-test`.

See `src/extension/README_extension.md` and `src/extension/NOTES_methodology.md` for details.

## Compatibility patches
Upstream scripts required minor compatibility fixes for modern pandas and NumPy. These do not change analysis logic:
- `as_matrix()` \u2192 `to_numpy()`
- `np.load(..., allow_pickle=True)` for legacy `.npz` sparse matrices

See `references/manifest_openicpsr.md` for details.


## Citation

If you use this code or refer to this replication, please cite the original paper:

**Economics-style (APA):**  
Wu, A. H. (2018). *Gendered language on the Economics Job Market Rumors forum*. AEA Papers and Proceedings, 108, 175–179. https://doi.org/10.1257/pandp.20181101

**BibTeX:**
```bibtex
@article{wuGenderedLanguageEconomics2018,
  title = {Gendered Language on the Economics Job Market Rumors Forum},
  author = {Wu, Alice H.},
  journal = {AEA Papers and Proceedings},
  volume = {108},
  pages = {175--179},
  year = {2018},
  month = {may},
  doi = {10.1257/pandp.20181101},
  url = {https://doi.org/10.1257/pandp.20181101}
}
```
