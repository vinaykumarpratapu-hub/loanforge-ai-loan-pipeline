"""
LoanForge Orchestrator Agent

This is the "agentic" piece of the pipeline: it doesn't score risk or read
documents itself, it decides the sequence, calls each specialist agent,
inspects each result, and decides whether to proceed, stop, or route to a
human — the goal -> choose tool -> take action -> observe result loop.

Flow:
  1. Document verification agent checks all uploaded documents.
     -> any invalid document = immediate reject, credit risk agent is
        never called (no point scoring risk for an unverified identity).
  2. Credit risk agent evaluates the application.
     -> approve / reject / manual_review, with reasoning.
  3. Final routing decision + a human-readable summary for the
     disbursement team / applicant.
"""
from datetime import datetime, timezone

from credit_risk_agent.agent import CreditRiskAgent
from document_verification_agent.agent import DocumentVerificationAgent


class LoanForgeOrchestrator:
    def __init__(self):
        self.doc_agent = DocumentVerificationAgent()
        self.credit_agent = CreditRiskAgent()

    def process_application(self, application: dict, documents: list) -> dict:
        """
        application: dict of applicant/loan fields (see CreditRiskAgent.evaluate)
                      plus "name" (the name typed on the form).
        documents: list of {"path": str, "doc_type": "id"|"income"}
        """
        trace = []
        applicant_name = application.get("name")

        # Step 1: document verification
        doc_results = []
        for doc in documents:
            result = self.doc_agent.verify(
                image_path=doc["path"],
                doc_type=doc["doc_type"],
                applicant_name=applicant_name,
            )
            doc_results.append(result)
            trace.append({
                "agent": "document_verification_agent",
                "action": f"verify {doc['doc_type']} document",
                "result_summary": "valid" if result["valid"] else f"invalid: {result['issues']}",
            })

        all_docs_valid = all(r["valid"] for r in doc_results)

        if not all_docs_valid:
            decision = "reject"
            reasons = [issue for r in doc_results for issue in r["issues"]]
            trace.append({
                "agent": "orchestrator",
                "action": "route decision",
                "result_summary": "rejected at document stage — credit risk agent not invoked",
            })
            return self._finalize(
                decision=decision,
                stage="document_verification",
                reasons=reasons,
                doc_results=doc_results,
                credit_result=None,
                trace=trace,
                application=application,
            )

        # Step 2: credit risk evaluation (only reached if documents are clean)
        credit_fields = {k: v for k, v in application.items()
                          if k in self.credit_agent.features}
        credit_result = self.credit_agent.evaluate(credit_fields)
        trace.append({
            "agent": "credit_risk_agent",
            "action": "evaluate repayment capability + risk signals",
            "result_summary": f"{credit_result['decision']} (risk={credit_result['default_risk_probability']})",
        })

        trace.append({
            "agent": "orchestrator",
            "action": "route decision",
            "result_summary": f"final decision: {credit_result['decision']}",
        })

        return self._finalize(
            decision=credit_result["decision"],
            stage="credit_risk",
            reasons=credit_result["reasoning"],
            doc_results=doc_results,
            credit_result=credit_result,
            trace=trace,
            application=application,
        )

    def _finalize(self, decision, stage, reasons, doc_results, credit_result, trace, application):
        summary = {
            "approve": "Application approved. Routed to disbursement queue.",
            "reject": f"Application rejected at {stage} stage.",
            "manual_review": "Application sent to manual review — credit signal is borderline.",
        }[decision]

        return {
            "applicant_name": application.get("name"),
            "decision": decision,
            "stage_decided_at": stage,
            "summary": summary,
            "reasons": reasons,
            "document_results": doc_results,
            "credit_result": credit_result,
            "agent_trace": trace,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
