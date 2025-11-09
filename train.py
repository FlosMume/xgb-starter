#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py
--------
Beginner-friendly training script for XGBoost (GPU-ready).
- Two modes:
    1) "es" (early stopping): quick loop with a train/validation split
       Reports AUC and F1 on the validation set. Saves final model.
    2) "cv" (cross validation): 5-fold CV with AUC & F1 across folds,
       then fits on full data and saves final model.

Dataset: sklearn's Breast Cancer (tabular, small, great for practice)
GPU: set device="cuda" (falls back to CPU if CUDA is unavailable)

Run:
  # Early stopping (default)
  python train.py

  # Cross-validation
  python train.py --mode cv
"""

import argparse
import os
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_validate
)
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    make_scorer
)

from xgboost import XGBClassifier


# ---------- 0) Reproducibility helpers ----------
def set_seed(seed: int = 42):
    """Set all the common random seeds for reproducible-ish results."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------- 1) Data loading ----------
def load_data():
    """
    Loads a small, clean classification dataset from sklearn.
    Returns feature matrix X (DataFrame) and labels y (Series).
    """
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    return X, y


# ---------- 2) Model factories ----------
def make_xgb_params_es():
    """
    Reasonable defaults for early-stopping training on GPU.
    Feel free to tweak n_estimators / learning_rate / depth later.
    """
    return dict(
        n_estimators=2000,        # big cap; early stopping will stop earlier
        learning_rate=0.03,       # slower learning, generally safer
        max_depth=6,              # typical depth for tabular
        subsample=0.9,            # row subsampling to generalize better
        colsample_bytree=0.9,     # feature subsampling
        tree_method="hist",       # fast histogram algorithm
        device="cuda",            # GPU; change to "cpu" if needed
        eval_metric="auc",        # we monitor AUC during early stopping
        early_stopping_rounds=200,# stop if no AUC improvement for 200 rounds
        # regularization knobs worth trying later:
        # reg_lambda=1.0,
        # reg_alpha=0.0,
    )


def make_xgb_params_cv():
    """
    Reasonable defaults for CV; often fewer trees than ES.
    """
    return dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        device="cuda",
        eval_metric="auc",
    )


# ---------- 3) Training modes ----------
def train_early_stop(X: pd.DataFrame, y: pd.Series):
    """
    Train with a single train/validation split and early stopping.
    Reports AUC and F1 on the validation set.
    Saves the model to xgb_model.joblib.
    """
    # Split once (stratify keeps class balance in both splits)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build the classifier with GPU settings
    clf = XGBClassifier(**make_xgb_params_es())

    # Fit with early stopping:
    # - eval_set: monitor performance on the validation split
    # - early_stopping_rounds: stop if no AUC improvement for N rounds
    # - verbose: print progress every 50 trees
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=50
    )

    # Evaluate on the validation split
    proba_va = clf.predict_proba(X_va)[:, 1]          # predicted probabilities
    preds_va = (proba_va >= 0.5).astype(int)          # 0.5 threshold (adjust as needed)

    val_auc = roc_auc_score(y_va, proba_va)
    val_f1 = f1_score(y_va, preds_va)

    print(f"[ES] Best iteration: {clf.best_iteration}")
    print(f"[ES] Validation AUC: {val_auc:.4f}")
    print(f"[ES] Validation F1 : {val_f1:.4f} (threshold=0.50)")

    # Save the trained model and feature names for predict.py
    bundle = {"model": clf, "feature_names": X.columns.tolist()}
    joblib.dump(bundle, "xgb_model.joblib")
    print("✅ Saved model to xgb_model.joblib")


def train_cv(X: pd.DataFrame, y: pd.Series):
    """
    5-fold cross-validation for more reliable metrics estimates.
    Prints mean±std of AUC and F1 across folds, then fits on full data.
    Saves the model to xgb_model.joblib.
    """
    params = make_xgb_params_cv()
    clf = XGBClassifier(**params)

    # 5-fold stratified split keeps class balance per fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # We define scoring functions for AUC and F1.
    # AUC needs probabilities (so needs_threshold=True).
    scoring = {
        "auc": make_scorer(roc_auc_score, needs_threshold=True),
        "f1": make_scorer(f1_score),
    }

    # cross_validate returns scores per fold; return_estimator=True is handy,
    # but we will re-fit on full data anyway for the final model.
    scores = cross_validate(
        clf, X, y, cv=cv, scoring=scoring, return_estimator=False, n_jobs=-1
    )

    auc_mean, auc_std = scores["test_auc"].mean(), scores["test_auc"].std()
    f1_mean,  f1_std  = scores["test_f1"].mean(),  scores["test_f1"].std()

    print("---------- 5-Fold CV Results ----------")
    print(f"AUC mean±std: {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"F1  mean±std: {f1_mean:.4f} ± {f1_std:.4f}")

    # Fit a final model on ALL data (so you can save/deploy one artifact)
    final = XGBClassifier(**params).fit(X, y)

    bundle = {"model": final, "feature_names": X.columns.tolist()}
    joblib.dump(bundle, "xgb_model.joblib")
    print("✅ Saved final model (fit on full data) to xgb_model.joblib")


# ---------- 4) Main ----------
def main():
    set_seed(42)
    X, y = load_data()

    parser = argparse.ArgumentParser(description="Train XGBoost with ES or CV.")
    parser.add_argument(
        "--mode",
        choices=["es", "cv"],
        default="es",
        help="Training mode: 'es' = early stopping, 'cv' = 5-fold cross-validation"
    )
    args = parser.parse_args()

    if args.mode == "es":
        train_early_stop(X, y)
    else:
        train_cv(X, y)


if __name__ == "__main__":
    main()
