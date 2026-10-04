"""
LoanForge Backend API

Single endpoint that a simple web form posts to: applicant details as form
fields, ID + income proof as file uploads. Saves uploads, derives the
underwriting fields the credit risk agent needs, runs the orchestrator,
and returns the full decision + trace as JSON for the frontend to render.
"""
import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from orchestrator.agent import LoanForgeOrchestrator

app = FastAPI(title="LoanForge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your deployed frontend's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = LoanForgeOrchestrator()

UPLOAD_DIR = "backend/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_upload(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename)[1] or ".png"
    dest = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/apply")
def apply_for_loan(
    name: str = Form(...),
    age: int = Form(...),
    monthly_income: float = Form(...),
    employment_years: float = Form(...),
    existing_emi: float = Form(0),
    credit_history_years: float = Form(...),
    num_existing_loans: int = Form(0),
    savings_balance: float = Form(0),
    past_defaults: int = Form(0),
    loan_amount: float = Form(...),
    loan_term_months: int = Form(...),
    purpose: str = Form(...),
    id_document: UploadFile = File(...),
    income_document: UploadFile = File(...),
):
    id_path = save_upload(id_document)
    income_path = save_upload(income_document)

    proposed_emi = loan_amount / loan_term_months * 1.08
    debt_to_income = round((existing_emi + proposed_emi) / (monthly_income + 1), 3)

    application = {
        "name": name,
        "age": age,
        "monthly_income": monthly_income,
        "employment_years": employment_years,
        "existing_emi": existing_emi,
        "credit_history_years": credit_history_years,
        "num_existing_loans": num_existing_loans,
        "savings_balance": savings_balance,
        "past_defaults": past_defaults,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "purpose": purpose,
        "debt_to_income": debt_to_income,
    }

    documents = [
        {"path": id_path, "doc_type": "id"},
        {"path": income_path, "doc_type": "income"},
    ]

    result = orchestrator.process_application(application, documents)
    return result
