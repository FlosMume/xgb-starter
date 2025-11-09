import joblib, numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score, f1_score
from xgboost import XGBClassifier

# Data
Xy = load_breast_cancer(as_frame=True)
X, y = Xy.data, Xy.target

# Model (GPU if available): device='cuda' uses your RTX 4070
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    tree_method="hist",
    device="cuda"  # change to "cpu" if no GPU
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(
    model, X, y, cv=cv,
    scoring={"auc": make_scorer(roc_auc_score, needs_threshold=True),
             "f1": make_scorer(f1_score)},
    return_estimator=True, n_jobs=-1
)

print("AUC (mean±std): %.4f ± %.4f" % (scores["test_auc"].mean(), scores["test_auc"].std()))
print("F1  (mean±std): %.4f ± %.4f" % (scores["test_f1"].mean(), scores["test_f1"].std()))

# Fit on full data and persist
final_model = XGBClassifier(
    n_estimators=round(np.mean([est.get_params()["n_estimators"] for est in scores["estimator"]])),
    max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
    reg_lambda=1.0, tree_method="hist", device="cuda"
)
final_model.fit(X, y)
joblib.dump({"model": final_model, "feature_names": X.columns.tolist()}, "xgb_model.joblib")
print("Saved model to xgb_model.joblib")

# Simple feature importance dump
imp = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)
imp.to_csv("feature_importance.csv")
print("Top 5 features:\n", imp.head(5))
