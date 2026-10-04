# LoanForge — Agentic Loan Processing Pipeline (Portfolio Project)

An AI/ML portfolio project built toward Senior PM roles in banking/fintech, sitting alongside
the iPhone Review Sentiment Analysis (v1/v2), Claude-powered Fraud Detection, and Credit Risk +
Fairness Audit projects.

**Live demo:** _add your deployed frontend URL here after deploying_
**API:** _add your deployed backend URL here after deploying_

LoanForge is **not** connected to any employer's product or real customer data. It's an
independent, fictional demo — a synthetic applicant, synthetic documents, and a synthetic
credit-risk dataset — built to show how a computer-vision agent, an ML risk model, and a thin
orchestration layer compose into one pipeline, which is the shape most real loan-origination
systems actually take.

## What this project demonstrates

Most of this portfolio's earlier projects are a single model behind a single UI. LoanForge is
different: it's an **agentic pipeline** — three specialized agents coordinated by an orchestrator
that decides, at each step, whether to proceed, short-circuit to a rejection, or route to a
human, and that produces a full trace of its own reasoning for transparency. It's modeled loosely
on a real NBFC loan origination flow: intake → document/KYC verification → underwriting →
disbursement.

```
Customer fills form + uploads documents (ID proof, income proof)
        |
        v
Document Verification Agent (computer vision / OCR)
  - OCR via Tesseract, extracts fields, checks required fields are present
  - Tamper detection: edge-density heuristic flags pasted/re-rendered regions
  - Cross-checks the name on the document vs. the name typed on the form
        |
   invalid? ---> REJECT (credit risk agent is never called)
        |
       valid
        v
Credit Risk Agent (ML — XGBoost classifier + SHAP explainability)
  - approve / reject / manual_review + probability of default
  - plain-language reasoning via top SHAP feature attributions
        |
        v
Orchestrator Agent
  - coordinates the sequence above (goal -> choose tool -> take action -> observe result)
  - decides whether to proceed, short-circuit to reject, or route to manual review
  - produces a full agent_trace for transparency
        |
        v
Disbursement queue / summary report (final JSON response)
```

## Design decisions

**Why there's no separate fraud-check agent.** The project started as a 4-agent chain (intake →
credit risk → document verification → fraud check). In real loan origination systems, fraud
detection isn't usually its own pipeline stage — it's embedded either in KYC/document
verification (forged documents, identity mismatch) or in underwriting (suspicious financial
patterns). So fraud signal is folded into the Document Verification Agent (document-level fraud:
tampering, mismatched identity) and into the Credit Risk Agent (pattern-level fraud via its risk
score), rather than bolted on as a fourth stage that duplicates what the other two already catch.

**Why intake is a form, not a chatbot.** A conversational intake agent was considered and
dropped in favor of a straightforward web form (name, loan amount, purpose, basic financials)
plus a document upload button — more realistic for this use case, and it keeps the complexity
budget on the two agents that actually need to be smart (vision and risk), not on the UI.

**Why credit risk runs after document verification, not before or in parallel.** There's no
point scoring creditworthiness for an applicant whose identity hasn't been verified yet. The
orchestrator short-circuits to reject at the document stage without ever invoking the credit risk
agent if documents are invalid — visible directly in the `agent_trace` and in `credit_result:
null` in the API response. This is a deliberate design point, not a missing feature, and it's
easy to demo live.

**Why the dataset is synthetic.** There's no access to real bank data for a personal portfolio
project, and this was built in an environment with no route to public dataset sources either. So
the credit-risk training data is generated from first principles: realistic feature distributions
(income, EMI, credit history, savings, etc.) combined through a logistic risk formula with noise
— not scraped or copied from an existing dataset. This is disclosed honestly rather than implied
otherwise; it's a normal, accepted approach for a portfolio project, and the formula itself
(weighted by debt-to-income, past defaults, credit history length, etc.) is written to mirror how
a real credit officer actually reasons about repayment risk.

**Why the documents are synthetic too.** Same reason. The demo applicant "Rahul Sharma" has a
generated ID card and salary slip, clearly watermarked "SAMPLE" and not based on any real
document template, used to exercise the pipeline end-to-end. There's also a deliberately
tampered version of the ID (a pasted-in name edit) used to demonstrate the tamper-detection check
actually catching something, rather than just asserting that it would.

**Why explainability matters here specifically.** This is a lending use case, so the credit risk
agent has to justify its decisions in plain language rather than return a bare probability — both
a regulatory/ethical expectation in real underwriting and a stronger interview talking point than
"the model said no." SHAP (TreeExplainer) supplies the per-decision reasoning.

## Tech stack

- **Credit risk agent:** Python, scikit-learn pipeline (`OneHotEncoder` + `ColumnTransformer`),
  XGBoost classifier, SHAP (`TreeExplainer`) for per-decision reasoning, joblib for persistence
- **Document verification agent:** Python, OpenCV (`cv2`) for image processing + an edge-density
  tamper heuristic, Tesseract OCR via `pytesseract`, regex-based field extraction
- **Orchestrator:** a plain Python class coordinating the two agents above — no external
  agent framework, kept simple and transparent for interview walk-throughs (LangChain/LangGraph
  is a stated future improvement, not a missing requirement)
- **Backend:** FastAPI, `python-multipart` for file uploads, CORS enabled
- **Frontend:** a single self-contained HTML/CSS/JS file (no build step), styled like a loan
  officer's ledger/passbook rather than a generic SaaS dashboard — paper background, serif
  headings, hairline dividers — submitting via `fetch()` with `FormData`

## Results

5-fold held-out test set, 8,000 synthetic applicants, 29.5% default rate:

| Metric | Value |
|---|---|
| ROC-AUC | 0.733 |
| Precision (defaulted) | 0.63 |
| Recall (defaulted) | 0.32 |
| Precision (repaid) | 0.77 |
| Recall (repaid) | 0.92 |

The model is deliberately conservative about calling a default (low recall on the minority
class) relative to its precision — consistent with the three-way decision design, where anything
it isn't confident about gets routed to `manual_review` rather than forced into a binary
approve/reject.

**Example pipeline runs** (from local testing before deployment):

- **Approve** — strong applicant, clean documents: both documents pass verification, credit
  risk agent returns `approve` at a 7.5% default probability, with SHAP citing a healthy
  debt-to-income ratio, income, and savings balance as the main factors reducing risk.
- **Reject at document stage** — tampered ID uploaded: the document verification agent catches
  the mismatch between the (tampered) name on the ID and the name typed on the form and rejects
  immediately; `credit_result` is `null` and the trace shows the credit risk agent was never
  invoked.
- **Reject at credit-risk stage** — clean documents, weak financial profile (low income, short
  credit history, an existing default, high requested loan amount): documents pass, but the
  credit risk agent returns `reject` at a 98.3% default probability, citing debt-to-income, past
  default history, and loan amount as the main risk drivers.

## Repo structure

```
loanforge/
├── credit_risk_agent/
│   ├── generate_data.py     # synthetic training data generator
│   ├── train.py              # trains XGBoost + builds SHAP explainer
│   └── agent.py               # CreditRiskAgent: evaluate() -> decision + reasoning
├── document_verification_agent/
│   ├── generate_sample_docs.py  # synthetic demo ID / income proof images
│   └── agent.py               # DocumentVerificationAgent: verify() -> valid + issues
├── orchestrator/
│   └── agent.py               # LoanForgeOrchestrator: coordinates the two agents above
├── backend/
│   └── main.py                 # FastAPI app, single POST /apply endpoint
├── frontend/
│   └── index.html               # self-contained form + results UI
├── data/                        # generated training data + sample documents
├── models/                      # trained model, SHAP explainer, feature metadata
└── requirements.txt
```

## Running locally

```bash
pip install -r requirements.txt
# Tesseract OCR is a system binary, not a pip package:
#   Debian/Ubuntu: sudo apt-get install tesseract-ocr
#   macOS: brew install tesseract

python credit_risk_agent/generate_data.py
python credit_risk_agent/train.py
python document_verification_agent/generate_sample_docs.py

uvicorn backend.main:app --reload --port 8000
# in a separate terminal/tab, just open frontend/index.html in a browser
```

The frontend's `API_URL` constant defaults to `http://127.0.0.1:8000/apply` for local dev —
update it to the deployed backend URL before deploying the frontend (see below).

## Deploying

1. **Push to GitHub** — `git init && git add . && git commit -m "..." && git remote add origin
   ... && git push`.
2. **Backend (FastAPI + Tesseract):** deploy to a host that supports installing a system
   package, not a serverless platform that can't — **Render** or **Railway** both work well.
   - Add an `apt.txt` (Render) or equivalent build step containing `tesseract-ocr` so the system
     binary is installed before the app starts.
   - Set the start command to `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
   - Once live, note the backend's public URL.
3. **Frontend (static file):** deploy `frontend/index.html` to GitHub Pages, Netlify, or Vercel
   — any static host works since there's no build step.
   - Before deploying, update the `API_URL` constant near the bottom of `index.html`'s `<script>`
     to the backend's deployed URL from step 2.
4. Update the live demo links at the top of this README once both are live.

## Interview talking points

- **Why this is an agentic pipeline, not just a model behind a form** — the orchestrator makes a
  real routing decision at each step (proceed / short-circuit / route to human) and exposes its
  own reasoning as a trace, which is the part of "agentic" that actually matters for a lending
  use case: auditability.
- **Why there's no separate fraud-check agent** — fraud signal is embedded in document
  verification (identity-level) and credit risk (pattern-level) rather than duplicated as a
  fourth stage, mirroring how real underwriting teams actually structure this.
- **Why credit risk only runs after documents clear** — scoring an unverified identity is
  wasted work and a bad practice to normalize, so the orchestrator enforces the ordering in code,
  not just in a diagram.
- **Honest handling of synthetic data** — both the credit-risk dataset and the demo documents are
  synthetic by necessity (no access to real bank data, no reachable public dataset source in the
  build environment), and that's disclosed plainly rather than glossed over. The generation logic
  itself mirrors real underwriting reasoning (debt-to-income, credit history, past defaults)
  rather than being arbitrary.
- **Why SHAP, not just a probability score** — lending decisions carry a real expectation of
  explainability; "high risk" alone isn't a defensible reason to decline someone, but "high
  debt-to-income ratio and a recorded past default" is.
- **What's a stated next step, not a gap** — swapping the hand-rolled orchestrator for
  LangChain/LangGraph is a reasonable evolution once the pipeline needs more than two
  specialist agents; it was kept simple here deliberately, for transparency during a live
  interview walkthrough.
