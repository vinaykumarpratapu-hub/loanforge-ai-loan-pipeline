"""
Generates a synthetic but realistic credit-risk training dataset for LoanForge.
Built from first principles (not a copied dataset) so the project has no
external data dependency, with a default-risk formula that mirrors how real
underwriting reasons about repayment capability.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 8000

age = np.random.randint(21, 65, N)
monthly_income = np.round(np.random.lognormal(mean=10.6, sigma=0.45, size=N), -2)  # INR
monthly_income = np.clip(monthly_income, 12000, 500000)

employment_years = np.clip(np.random.exponential(scale=5, size=N), 0, 35)
existing_emi = np.round(monthly_income * np.random.beta(2, 6, N), -2)  # existing monthly debt obligations
credit_history_years = np.clip(np.random.exponential(scale=4, size=N), 0, 30)
num_existing_loans = np.random.poisson(1.1, N)
savings_balance = np.round(np.random.lognormal(mean=9.5, sigma=1.2, size=N), -2)
past_defaults = np.random.binomial(1, 0.08, N)  # 8% have a past default on record

loan_amount = np.round(np.random.lognormal(mean=12.0, sigma=0.6, size=N), -3)
loan_amount = np.clip(loan_amount, 25000, 2000000)
loan_term_months = np.random.choice([12, 24, 36, 48, 60], N, p=[0.15, 0.25, 0.3, 0.2, 0.1])

purpose = np.random.choice(
    ["personal", "home_improvement", "vehicle", "education", "medical", "business"],
    N, p=[0.3, 0.15, 0.2, 0.1, 0.1, 0.15]
)

# Derived underwriting ratios (what a real credit officer looks at)
proposed_emi = loan_amount / loan_term_months * 1.08  # rough EMI incl. interest
debt_to_income = (existing_emi + proposed_emi) / (monthly_income + 1)
disposable_income_ratio = 1 - debt_to_income

# Risk score built from a weighted logic, plus noise -> probability of default
risk_logit = (
    -2.2
    + 3.2 * debt_to_income
    - 0.015 * credit_history_years
    - 0.01 * employment_years
    + 1.6 * past_defaults
    + 0.12 * num_existing_loans
    - 0.08 * np.log1p(savings_balance / 1000)
    - 0.15 * np.log1p(monthly_income / 10000)
    + np.random.normal(0, 0.35, N)
)
default_prob = 1 / (1 + np.exp(-risk_logit))
defaulted = np.random.binomial(1, default_prob)

df = pd.DataFrame({
    "age": age,
    "monthly_income": monthly_income,
    "employment_years": np.round(employment_years, 1),
    "existing_emi": existing_emi,
    "credit_history_years": np.round(credit_history_years, 1),
    "num_existing_loans": num_existing_loans,
    "savings_balance": savings_balance,
    "past_defaults": past_defaults,
    "loan_amount": loan_amount,
    "loan_term_months": loan_term_months,
    "purpose": purpose,
    "debt_to_income": np.round(debt_to_income, 3),
    "defaulted": defaulted,
})

df.to_csv("data/credit_risk_training.csv", index=False)
print(f"Generated {len(df)} rows. Default rate: {df['defaulted'].mean():.1%}")
print(df.head(3).to_string())
