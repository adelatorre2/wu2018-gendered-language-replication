from __future__ import annotations

"""
Shared utilities for the replication extension.

This module focuses on *reusing* the upstream replication data and
splits so the extension is apples-to-apples with the Wu (2018) pipeline.
The code is intentionally verbose and heavily commented for clarity.
"""

from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd
from scipy import sparse


def repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def raw_pkg_dir() -> Path:
    """Return the raw OpenICPSR package directory."""
    return repo_root() / "data" / "raw" / "openicpsr_wu2018_replication-pkg"


def extension_output_dir() -> Path:
    """Return the output directory for extension artifacts."""
    return repo_root() / "output" / "extension"


def ensure_extension_output_dirs() -> None:
    """Create output directories for the extension (OLS and RF)."""
    base = extension_output_dir()
    (base / "ols").mkdir(parents=True, exist_ok=True)
    (base / "rf").mkdir(parents=True, exist_ok=True)


def _find_keep_columns_path(raw_dir: Path) -> Path | None:
    """
    Locate i_keep_columns.txt produced by upstream scripts.

    We prefer the file copied by our pipeline into output/intermediate,
    because it reflects the exact (potentially unsorted) column order
    that the upstream script used when it ran.
    """
    candidate_paths = [
        repo_root() / "output" / "intermediate" / "lasso_full" / "i_keep_columns.txt",
        raw_dir / "i_keep_columns.txt",
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    return None


def _load_keep_columns(vocab_path: Path, raw_dir: Path) -> np.ndarray:
    """
    Load the column indices that remain after excluding gender classifiers.

    Priority:
    1) use i_keep_columns.txt if it exists (to match upstream ordering)
    2) otherwise compute from vocab10K.csv (deterministic ascending order)
    """
    keep_path = _find_keep_columns_path(raw_dir)
    if keep_path is not None:
        keep_idx = np.loadtxt(keep_path).astype(int)
        return keep_idx

    vocab = pd.read_csv(vocab_path)
    # 'exclude' == 1 means the word is *not* used as a predictor
    keep_idx = vocab.loc[vocab["exclude"] != 1, "index"].to_numpy() - 1
    # Ensure deterministic order when recomputing from vocab
    keep_idx = np.sort(keep_idx.astype(int))
    return keep_idx


def load_design_matrix_and_labels(
    raw_dir: Path | None = None,
) -> tuple[sparse.spmatrix, np.ndarray, list[str], np.ndarray, np.ndarray]:
    """
    Load X, y, vocab, and upstream train/test indices.

    Returns:
        X: sparse matrix (n_posts x n_features) after upstream filtering
        y: binary array aligned with X rows (Female=1, Male=0)
        vocab: list of tokens aligned with X columns
        train_idx: indices where training == 1 (upstream logic)
        test_idx: indices where training == 0 (upstream logic)

    Notes:
    - We match the upstream Lasso scripts:
        * keys_to_X.csv defines row order
        * gendered_posts.csv supplies labels and training split
        * vocab10K.csv + i_keep_columns.txt define predictor columns
    - We *exclude* duplicate posts (training is NA) from train/test.
    """
    raw_dir = raw_dir or raw_pkg_dir()

    keys_path = raw_dir / "keys_to_X.csv"
    posts_path = raw_dir / "gendered_posts.csv"
    vocab_path = raw_dir / "vocab10K.csv"
    npz_path = raw_dir / "X_word_count.npz"

    if not keys_path.exists():
        raise FileNotFoundError(f"Missing keys file: {keys_path}")
    if not posts_path.exists():
        raise FileNotFoundError(f"Missing posts file: {posts_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocab file: {vocab_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing word-count matrix: {npz_path}")

    # Load keys and the minimal columns needed from the large posts file
    keys = pd.read_csv(keys_path, usecols=["title_id", "post_id"])
    posts = pd.read_csv(
        posts_path,
        usecols=["title_id", "post_id", "training", "female"],
    )

    # Upstream logic merges on (title_id, post_id) to align with X rows
    keys_merged = pd.merge(keys, posts, on=["title_id", "post_id"], how="left")

    # Upstream train/test split definitions
    train_idx = np.where(keys_merged["training"] == 1)[0]
    test_idx = np.where(keys_merged["training"] == 0)[0]

    # Labels (Female=1, Male=0) aligned with full row order
    y = keys_merged["female"].to_numpy()

    # Basic sanity checks to prevent silent misalignment
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Train/test indices are empty; check training column.")
    if np.intersect1d(train_idx, test_idx).size > 0:
        raise ValueError("Train/test indices overlap; upstream split logic broke.")

    y_train = y[train_idx]
    y_test = y[test_idx]
    if np.isnan(y_train).any() or np.isnan(y_test).any():
        raise ValueError("Found NA labels in train/test sets; check source data.")

    # Ensure labels are binary
    unique_labels = set(np.unique(np.concatenate([y_train, y_test])))
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Unexpected labels in y: {sorted(unique_labels)}")

    # Load sparse word-count matrix (stored as a pickled SciPy sparse matrix)
    word_counts = np.load(npz_path, allow_pickle=True, encoding="latin1")
    X = word_counts["X"][()]

    if X.shape[0] != len(keys_merged):
        raise ValueError(
            "Row mismatch between X_word_count.npz and keys_to_X.csv: "
            f"X has {X.shape[0]} rows, keys has {len(keys_merged)}."
        )

    # Apply the same predictor filtering as upstream
    keep_idx = _load_keep_columns(vocab_path, raw_dir)
    X = X[:, keep_idx]

    # Load vocabulary and align to filtered columns
    vocab = pd.read_csv(vocab_path)["word"].tolist()
    vocab = [vocab[i] for i in keep_idx]

    if X.shape[1] != len(vocab):
        raise ValueError(
            "Column mismatch after filtering: "
            f"X has {X.shape[1]} columns, vocab has {len(vocab)}."
        )

    return X, y, vocab, train_idx, test_idx


def select_top_k_by_doc_frequency(
    X_train: sparse.spmatrix, vocab: list[str], k: int
) -> tuple[np.ndarray, list[str]]:
    """
    Keep the top-k features by document frequency on the training set.

    This is a pragmatic reduction for Random Forests; it avoids costly
    densification of a 10k-dimensional sparse matrix.
    """
    if k >= X_train.shape[1]:
        idx = np.arange(X_train.shape[1])
        return idx, vocab

    # getnnz(axis=0) counts non-zero entries per column (document frequency)
    df = np.asarray(X_train.getnnz(axis=0)).ravel()
    idx = np.argsort(df)[::-1][:k]
    return idx, [vocab[i] for i in idx]


def sample_indices(idx: np.ndarray, max_n: int | None, rng: np.random.Generator) -> np.ndarray:
    """Randomly subsample indices without replacement if max_n is set."""
    if max_n is None or len(idx) <= max_n:
        return idx
    return rng.choice(idx, size=max_n, replace=False)


def mean_difference_sign(X: sparse.spmatrix, y: np.ndarray) -> np.ndarray:
    """
    Compute the sign of the mean difference for each feature:
    sign(mean(X|y=1) - mean(X|y=0)).

    This provides a simple directional label (female vs male) to pair
    with unsigned RF feature importances.
    """
    y = y.astype(int)
    if not sparse.issparse(X):
        raise ValueError("Expected sparse X for mean_difference_sign.")

    mean_pos = X[y == 1].mean(axis=0)
    mean_neg = X[y == 0].mean(axis=0)
    diff = np.asarray(mean_pos - mean_neg).ravel()
    return np.sign(diff)


def write_word_table(
    female_words: Iterable[str],
    female_scores: Iterable[float],
    male_words: Iterable[str],
    male_scores: Iterable[float],
    csv_path: Path,
    tex_path: Path,
    female_label: str,
    male_label: str,
) -> None:
    """
    Write a side-by-side word table in CSV and LaTeX.

    The structure mirrors Wu (2018) Table 1: two columns of words
    with their scores (coefficients or importances).
    """
    def _latex_escape(text: str) -> str:
        replacements = {
            "\\\\": r"\\textbackslash{}",
            "&": r"\\&",
            "%": r"\\%",
            "$": r"\\$",
            "#": r"\\#",
            "_": r"\\_",
            "{": r"\\{",
            "}": r"\\}",
            "~": r"\\textasciitilde{}",
            "^": r"\\textasciicircum{}",
        }
        return "".join(replacements.get(ch, ch) for ch in text)

    # Ensure we always write 10 rows; pad if needed
    female_words = list(female_words)
    female_scores = list(female_scores)
    male_words = list(male_words)
    male_scores = list(male_scores)

    while len(female_words) < 10:
        female_words.append("NA")
        female_scores.append(0.0)
    while len(male_words) < 10:
        male_words.append("NA")
        male_scores.append(0.0)

    rows = []
    for fw, fs, mw, ms in zip(female_words, female_scores, male_words, male_scores):
        rows.append({
            "female_word": fw,
            "female_score": fs,
            "male_word": mw,
            "male_score": ms,
        })

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV output
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # LaTeX output
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l r l r}\n")
        f.write("\\toprule\n")
        f.write(f"{female_label} & Score & {male_label} & Score\\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(
                f"{_latex_escape(str(row['female_word']))} & {row['female_score']:.4f} "
                f"& {_latex_escape(str(row['male_word']))} & {row['male_score']:.4f}\\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def write_metrics_json(metrics: dict, path: Path) -> None:
    """Write metrics to JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
