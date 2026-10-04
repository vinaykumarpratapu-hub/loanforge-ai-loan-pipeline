"""
LoanForge Credit Risk Agent

Takes a structured loan application, returns a decision (approve / reject /
manual_review) with a default-risk probability and plain-language reasoning
(via SHAP feature attributions) — the explainability lending requires.
"""
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/credit_risk_model.joblib"
FEATURES_PATH = "models/credit_risk_features.joblib"
EXPLAINER_PATH = "models/credit_risk_explainer.joblib"
FEATURE_NAMES_PATH = "models/credit_risk_feature_names.joblib"

# Decision thresholds — tuned for a conservative lender posture:
# clearly safe -> approve, clearly risky -> reject, everything in between
# (or any individually borderline signal) -> human review.
APPROVE_BELOW = 0.20
REJECT_ABOVE = 0.65

_FRIENDLY_NAMES = {
    "debt_to_income": "debt-to-income ratio",
    "past_defaults": "history of past default",
    "credit_history_years": "length of credit history",
    "existing_emi": "existing monthly EMI obligations",
    "num_existing_loans": "number of existing loans",
    "monthly_income": "monthly income",
    "savings_balance": "savings balance",
    "employment_years": "years of employment",
    "loan_amount": "requested loan amount",
    "loan_term_months": "loan term",
    "age": "age",
}


class CreditRiskAgent:
    def __init__(self):
        self.pipeline = joblib.load(MODEL_PATH)
        self.features = joblib.load(FEATURES_PATH)
        self.explainer = joblib.load(EXPLAINER_PATH)
        self.feature_names = joblib.load(FEATURE_NAMES_PATH)

    def _reasoning(self, applicant_row: pd.DataFrame, top_n: int = 3):
        transformed = self.pipeline.named_steps["preprocess"].transform(applicant_row)
        shap_values = self.explainer.shap_values(transformed)
        row_shap = shap_values[0] if shap_values.ndim == 2 else shap_values

        contributions = list(zip(self.feature_names, row_shap))
        # Collapse one-hot "purpose" columns into a single readable entry
        collapsed = {}
        for name, val in contributions:
            key = name.split("__")[-1]
            if key.startswith("purpose_"):
                key = "purpose"
            collapsed[key] = collapsed.get(key, 0) + val

        ranked = sorted(collapsed.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        reasons = []
        for key, val in ranked:
            label = _FRIENDLY_NAMES.get(key, key)
            direction = "increased" if val > 0 else "reduced"
            reasons.append(f"{label} {direction} the risk estimate")
        return reasons

    def evaluate(self, application: dict) -> dict:
        """
        application: dict with keys matching the training features:
          age, monthly_income, employment_years, existing_emi,
          credit_history_years, num_existing_loans, savings_balance,
          past_defaults, loan_amount, loan_term_months, purpose,
          debt_to_income
        """
        row = pd.DataFrame([application])[self.features]
        proba = float(self.pipeline.predict_proba(row)[0, 1])

        if proba < APPROVE_BELOW:
            decision = "approve"
        elif proba > REJECT_ABOVE:
            decision = "reject"
        else:
            decision = "manual_review"

        return {
            "decision": decision,
            "default_risk_probability": round(proba, 3),
            "reasoning": self._reasoning(row),
            "thresholds": {"approve_below": APPROVE_BELOW, "reject_above": REJECT_ABOVE},
        }
