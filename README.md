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

The folder should contain `gendered_posts.csv`, `X_word_count.npz`, `vocab10K.csv`, `trend_stats.csv`, and `lasso/`.

4. Run the full replication pipeline.

```bash
python src/run_all.py
```

Outputs are written to `output/intermediate/`, `output/tables/`, `output/figures/`, and logs to `output/logs/`.
