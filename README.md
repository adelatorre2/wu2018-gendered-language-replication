# Wu (2018) Gendered Language Replication

## How to run
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

4. Run the full replication pipeline.

```bash
python src/run_all.py
```

Outputs are written to:
- `output/intermediate/`
- `output/tables/`
- `output/figures/`
- `output/logs/`

## Extension (alternative models)
Run the extension models that re-create the \"Table 1\" idea with OLS/LPM and Random Forest:

```bash
python -m src.extension.fit_extension --model all
```

Outputs are written to `output/extension/ols/` and `output/extension/rf/`.
See `src/extension/README_extension.md` for details and options.

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
