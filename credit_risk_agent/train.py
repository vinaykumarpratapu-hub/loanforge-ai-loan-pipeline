"""
Trains the LoanForge credit risk agent: a gradient-boosted classifier that
predicts probability of default, with SHAP for per-decision explainability
(required for lending reasoning, per regulatory/compliance norms).
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/credit_risk_training.csv")

FEATURES = [
    "age", "monthly_income", "employment_years", "existing_emi",
    "credit_history_years", "num_existing_loans", "savings_balance",
    "past_defaults", "loan_amount", "loan_term_months", "purpose",
    "debt_to_income",
]
TARGET = "defaulted"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

categorical = ["purpose"]
numeric = [c for c in FEATURES if c not in categorical]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
], remainder="passthrough")

clf = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9,
    eval_metric="logloss", random_state=42,
)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", clf),
])

pipeline.fit(X_train, y_train)

y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_proba)
print(f"Test ROC-AUC: {auc:.3f}")
print(classification_report(y_test, y_pred, target_names=["repaid", "defaulted"]))

joblib.dump(pipeline, "models/credit_risk_model.joblib")
joblib.dump(FEATURES, "models/credit_risk_features.joblib")

# Build a SHAP explainer on the transformed (post-preprocessing) feature space
X_train_transformed = pipeline.named_steps["preprocess"].transform(X_train)
feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
explainer = shap.TreeExplainer(pipeline.named_steps["model"])
joblib.dump(explainer, "models/credit_risk_explainer.joblib")
joblib.dump(list(feature_names), "models/credit_risk_feature_names.joblib")

print("Saved model, explainer, and feature metadata to models/")
