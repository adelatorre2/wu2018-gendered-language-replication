from __future__ import annotations

"""
Fit extension models for Wu (2018):
- OLS / Linear Probability Model (LPM)
- Random Forest classifier

The goal is to reproduce the "Table 1" idea (most predictive words
for female vs male referenced posts) using alternative models while
preserving the upstream data construction and train/test split.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from scipy import sparse

try:
    # When running as a module: python -m src.extension.fit_extension
    from .utils_features import (
        ensure_extension_output_dirs,
        extension_output_dir,
        load_design_matrix_and_labels,
        mean_difference_sign,
        sample_indices,
        select_top_k_by_doc_frequency,
        write_metrics_json,
        write_word_table,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from utils_features import (  # type: ignore
        ensure_extension_output_dirs,
        extension_output_dir,
        load_design_matrix_and_labels,
        mean_difference_sign,
        sample_indices,
        select_top_k_by_doc_frequency,
        write_metrics_json,
        write_word_table,
    )


def setup_logging(log_path: Path | None) -> None:
    """Configure logging to stdout and (optionally) a log file."""
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=handlers,
    )


def evaluate_classification(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """Compute standard classification metrics at a fixed threshold."""
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_ols(
    X: sparse.spmatrix,
    y: np.ndarray,
    vocab: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Fit an OLS / LPM baseline and write outputs."""
    logging.info("[OLS] Preparing train/test splits")
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Use CSR for efficient row slicing in scikit-learn
    if sparse.issparse(X_train):
        X_train = X_train.tocsr()
        X_test = X_test.tocsr()

    logging.info("[OLS] Fitting LinearRegression (LPM)")
    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train)

    # Raw predictions are not bounded; clip for probability metrics
    y_pred_raw = model.predict(X_test)
    y_pred_clipped = np.clip(y_pred_raw, 0.0, 1.0)

    # Classification metrics at threshold 0.5
    threshold = 0.5
    metrics = evaluate_classification(y_test, y_pred_clipped, threshold)

    # AUC uses the *raw* scores to preserve ranking
    auc = roc_auc_score(y_test, y_pred_raw)

    metrics_out = {
        "model_name": "ols_lpm",
        "split_description": "Upstream training==1 vs training==0 (non-duplicate posts only)",
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features": int(X.shape[1]),
        "threshold": threshold,
        "roc_auc": float(auc),
        "raw_pred_min": float(y_pred_raw.min()),
        "raw_pred_max": float(y_pred_raw.max()),
        "notes": (
            "LPM predictions are unbounded; we clip to [0,1] for "
            "threshold-based metrics. AUC uses raw scores for ranking."
        ),
    }
    metrics_out.update({k: float(v) for k, v in metrics.items()})

    # Extract coefficients for "most predictive" words
    coef = model.coef_.ravel()
    top_pos_idx = np.argsort(coef)[-10:][::-1]
    top_neg_idx = np.argsort(coef)[:10]

    female_words = [vocab[i] for i in top_pos_idx]
    female_scores = [float(coef[i]) for i in top_pos_idx]
    male_words = [vocab[i] for i in top_neg_idx]
    male_scores = [float(coef[i]) for i in top_neg_idx]

    out_dir = extension_output_dir() / "ols"
    write_metrics_json(metrics_out, out_dir / "metrics.json")
    write_word_table(
        female_words,
        female_scores,
        male_words,
        male_scores,
        out_dir / "table1_ols.csv",
        out_dir / "table1_ols.tex",
        female_label="Most female (OLS)",
        male_label="Most male (OLS)",
    )

    logging.info("[OLS] Done. Outputs written to %s", out_dir)


def run_rf(
    X: sparse.spmatrix,
    y: np.ndarray,
    vocab: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    max_features: int,
    max_train: int,
    max_test: int,
    n_estimators: int,
    random_state: int,
) -> None:
    """Fit a Random Forest classifier with a pragmatic feature reduction."""
    logging.info("[RF] Preparing train/test splits")
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Feature reduction: keep top-k features by document frequency
    logging.info("[RF] Selecting top-%d features by document frequency", max_features)
    keep_idx, vocab_reduced = select_top_k_by_doc_frequency(X_train, vocab, max_features)
    X_train = X_train[:, keep_idx]
    X_test = X_test[:, keep_idx]

    # Subsample rows to keep the RF step tractable on large sparse data
    rng = np.random.default_rng(random_state)
    train_rows = sample_indices(np.arange(X_train.shape[0]), max_train, rng)
    test_rows = sample_indices(np.arange(X_test.shape[0]), max_test, rng)

    X_train = X_train[train_rows]
    y_train = y_train[train_rows]
    X_test = X_test[test_rows]
    y_test = y_test[test_rows]

    # RandomForest in sklearn expects dense input; convert after reduction
    # (kept small by max_features and max_train/max_test)
    if sparse.issparse(X_train):
        X_train = X_train.toarray()
    if sparse.issparse(X_test):
        X_test = X_test.toarray()

    logging.info(
        "[RF] Fitting RandomForestClassifier (n_estimators=%d, n_train=%d, n_test=%d, n_features=%d)",
        n_estimators,
        X_train.shape[0],
        X_test.shape[0],
        X_train.shape[1],
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]
    threshold = 0.5
    metrics = evaluate_classification(y_test, y_score, threshold)
    auc = roc_auc_score(y_test, y_score)

    metrics_out = {
        "model_name": "random_forest",
        "split_description": "Upstream training==1 vs training==0 (non-duplicate posts only)",
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "threshold": threshold,
        "roc_auc": float(auc),
        "feature_reduction": {
            "method": "top_k_by_document_frequency",
            "k": int(max_features),
        },
        "row_subsampling": {
            "max_train": int(max_train),
            "max_test": int(max_test),
            "random_state": int(random_state),
        },
        "notes": (
            "Random Forests are computationally heavy on large sparse text data. "
            "We reduce features and subsample rows to keep the extension feasible. "
            "Feature importances are Gini-based and can be biased toward frequent terms."
        ),
    }
    metrics_out.update({k: float(v) for k, v in metrics.items()})

    # Feature importance (unsigned) + directional sign from mean differences
    importances = model.feature_importances_.ravel()

    # Recompute sign using the *sparse* training data before densification
    # (we can approximate using the dense subset we actually trained on)
    # For interpretability, use means on the same sampled training data.
    X_train_sparse = sparse.csr_matrix(X_train)
    direction = mean_difference_sign(X_train_sparse, y_train)

    # Build ranked lists by importance, then split by direction
    ranked_idx = np.argsort(importances)[::-1]
    female_words = []
    female_scores = []
    male_words = []
    male_scores = []

    for i in ranked_idx:
        if direction[i] > 0 and len(female_words) < 10:
            female_words.append(vocab_reduced[i])
            female_scores.append(float(importances[i]))
        elif direction[i] < 0 and len(male_words) < 10:
            male_words.append(vocab_reduced[i])
            male_scores.append(float(importances[i]))
        if len(female_words) == 10 and len(male_words) == 10:
            break

    out_dir = extension_output_dir() / "rf"
    write_metrics_json(metrics_out, out_dir / "metrics.json")
    write_word_table(
        female_words,
        female_scores,
        male_words,
        male_scores,
        out_dir / "table1_rf.csv",
        out_dir / "table1_rf.tex",
        female_label="Most female (RF)",
        male_label="Most male (RF)",
    )

    logging.info("[RF] Done. Outputs written to %s", out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wu (2018) extension models")
    parser.add_argument(
        "--model",
        choices=["ols", "rf", "all"],
        default="all",
        help="Which model(s) to run",
    )
    parser.add_argument("--rf-max-features", type=int, default=1000)
    parser.add_argument("--rf-max-train", type=int, default=20000)
    parser.add_argument("--rf-max-test", type=int, default=10000)
    parser.add_argument("--rf-n-estimators", type=int, default=200)
    parser.add_argument("--rf-random-state", type=int, default=123)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("output/logs"),
        help="Directory for log files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ensure_extension_output_dirs()

    log_path = None
    if args.model == "ols":
        log_path = args.log_dir / "extension_ols.log"
    elif args.model == "rf":
        log_path = args.log_dir / "extension_rf.log"
    else:
        log_path = args.log_dir / "extension_all.log"

    setup_logging(log_path)

    logging.info("Loading design matrix and labels (upstream split)")
    X, y, vocab, train_idx, test_idx = load_design_matrix_and_labels()

    # Run requested models
    if args.model in {"ols", "all"}:
        run_ols(X, y, vocab, train_idx, test_idx)
    if args.model in {"rf", "all"}:
        run_rf(
            X,
            y,
            vocab,
            train_idx,
            test_idx,
            max_features=args.rf_max_features,
            max_train=args.rf_max_train,
            max_test=args.rf_max_test,
            n_estimators=args.rf_n_estimators,
            random_state=args.rf_random_state,
        )

    logging.info("Extension run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
