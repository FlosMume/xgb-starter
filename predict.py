#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict.py
----------
Loads a saved model (xgb_model.joblib) and runs predictions on a CSV.

Input CSV must contain the same columns used at training time.
The script:
  1) loads the trained model bundle (model + feature_names)
  2) reads your input CSV
  3) aligns columns (order & names)
  4) outputs predictions.csv with:
       - prediction (0/1 using default threshold 0.50)
       - proba      (predicted probability for the positive class)

Run:
  python predict.py path/to/your_input.csv
  # Optional: change threshold
  python predict.py path/to/your_input.csv --threshold 0.35
"""

import argparse
import sys
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier   # imported so the unpickler knows the class


def load_bundle(model_path: str = "xgb_model.joblib"):
    """
    Load the persisted model bundle (dict with 'model' and 'feature_names').
    """
    try:
        bundle = joblib.load(model_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: Cannot find '{model_path}'. Train a model first (run train.py).")
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_names" not in bundle:
        sys.exit("ERROR: Bad model bundle format. Expected keys: 'model' and 'feature_names'.")
    return bundle


def read_input_csv(csv_path: str) -> pd.DataFrame:
    """
    Read input CSV into a DataFrame.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        sys.exit(f"ERROR: Failed to read CSV '{csv_path}': {e}")
    if df.empty:
        sys.exit("ERROR: Input CSV is empty.")
    return df


def align_columns(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Ensure input DataFrame has exactly the columns used during training.
    - Checks for missing columns
    - Orders columns to match the training schema
    """
    df_cols = set(df.columns)
    feat_cols = set(feature_names)

    missing = feat_cols - df_cols
    extra = df_cols - feat_cols

    if missing:
        # For beginners, fail fast with a clear message.
        # Advanced users could impute or drop features instead.
        sys.exit(f"ERROR: Missing required columns: {sorted(missing)}")

    if extra:
        # Extra columns won't break XGBoost if we drop them explicitly.
        print(f"NOTE: Dropping unused columns: {sorted(extra)}")
        df = df.drop(columns=list(extra))

    # Reorder to match the training order
    df = df[feature_names]
    return df


def main():
    parser = argparse.ArgumentParser(description="Run predictions with a saved XGBoost model.")
    parser.add_argument("csv", help="Path to input CSV.")
    parser.add_argument(
        "--threshold", type=float, default=0.50,
        help="Decision threshold for class 1 (default 0.50)."
    )
    args = parser.parse_args()

    # 1) Load model bundle
    bundle = load_bundle("xgb_model.joblib")
    model: XGBClassifier = bundle["model"]
    feature_names = bundle["feature_names"]

    # 2) Load and align input
    df_raw = read_input_csv(args.csv)
    df = align_columns(df_raw, feature_names)

    # 3) Predict probabilities and classes
    proba = model.predict_proba(df)[:, 1]
    preds = (proba >= args.threshold).astype(int)

    # 4) Save outputs
    out = pd.DataFrame({
        "prediction": preds,
        "proba": proba
    })
    out_path = "predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path} (threshold={args.threshold:.2f})")
    print(out.head())


if __name__ == "__main__":
    main()
