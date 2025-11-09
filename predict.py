import sys, joblib, pandas as pd
from xgboost import XGBClassifier

bundle = joblib.load("xgb_model.joblib")
model: XGBClassifier = bundle["model"]
feat_names = bundle["feature_names"]

# Expect a CSV with columns matching feat_names
df = pd.read_csv(sys.argv[1])
proba = model.predict_proba(df[feat_names])[:,1]
out = pd.DataFrame({"prediction": (proba >= 0.5).astype(int), "proba": proba})
out.to_csv("predictions.csv", index=False)
print("Wrote predictions.csv")
