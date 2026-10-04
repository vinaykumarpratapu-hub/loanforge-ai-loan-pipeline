"""
LoanForge Document Verification Agent

Computer vision / OCR agent that reads an uploaded ID or income proof image,
extracts key fields, and flags authenticity issues:
  - OCR confidence too low (poor scan / likely fake-print)
  - Required fields missing or unreadable
  - Tamper signal: localized region with a different font/edge-sharpness
    profile than the rest of the document (classic sign of a pasted edit)
  - Cross-document mismatch: does the name on this doc match the name the
    applicant gave on the form?
"""
import re
import numpy as np
import cv2
import pytesseract
from pytesseract import Output


REQUIRED_FIELDS_ID = ["name", "id_number", "address"]
REQUIRED_FIELDS_INCOME = ["name", "employer", "monthly_income"]

FIELD_PATTERNS = {
    "name": r"Name:\s*([A-Za-z .]+)",
    "employee_name": r"Employee Name:\s*([A-Za-z .]+)",
    "dob": r"Date of Birth:\s*([0-9A-Za-z-]+)",
    "id_number": r"ID Number:\s*([A-Za-z0-9-]+)",
    "address": r"Address:\s*(.+)",
    "employer": r"Employer:\s*(.+)",
    "monthly_income": r"Net Monthly Pay:\s*Rs\.?\s*([0-9,]+)",
}


class DocumentVerificationAgent:

    def _ocr_with_confidence(self, image_path: str):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(gray, output_type=Output.DICT)
        text = pytesseract.image_to_string(gray)

        confidences = [int(c) for c in data["conf"] if c not in ("-1", -1)]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return text, avg_conf, img

    def _extract_fields(self, text: str) -> dict:
        fields = {}
        for key, pattern in FIELD_PATTERNS.items():
            m = re.search(pattern, text)
            if m:
                fields[key] = m.group(1).strip()
        return fields

    def _detect_tamper_region(self, img) -> dict:
        """
        Heuristic tamper check: run edge-density analysis in horizontal bands.
        A pasted/edited region typically has a different edge-sharpness
        signature (re-rendered text, resampling artifacts) than the
        surrounding untouched print. We flag a band whose edge density is a
        statistical outlier vs the rest of the document.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        h = edges.shape[0]
        band_h = max(h // 12, 10)
        bands = [edges[i:i + band_h, :] for i in range(0, h, band_h)]
        densities = [float(np.mean(b > 0)) for b in bands if b.size > 0]

        if len(densities) < 4:
            return {"tamper_suspected": False}

        mean_d, std_d = np.mean(densities), np.std(densities)
        if std_d == 0:
            return {"tamper_suspected": False}

        outliers = [
            idx for idx, d in enumerate(densities)
            if abs(d - mean_d) > 2.2 * std_d and d > mean_d  # unusually dense = re-rendered text block
        ]
        return {
            "tamper_suspected": len(outliers) > 0,
            "suspect_bands": outliers,
        }

    def verify(self, image_path: str, doc_type: str, applicant_name: str) -> dict:
        """
        doc_type: "id" or "income"
        applicant_name: the name the applicant typed on the LoanForge form,
                         used to cross-check against the document.
        """
        text, avg_conf, img = self._ocr_with_confidence(image_path)
        fields = self._extract_fields(text)
        tamper = self._detect_tamper_region(img)

        required = REQUIRED_FIELDS_ID if doc_type == "id" else REQUIRED_FIELDS_INCOME
        name_on_doc = fields.get("name") or fields.get("employee_name")
        present = {
            "name": bool(name_on_doc),
            "id_number": bool(fields.get("id_number")),
            "address": bool(fields.get("address")),
            "employer": bool(fields.get("employer")),
            "monthly_income": bool(fields.get("monthly_income")),
        }
        missing = [f for f in required if not present.get(f, False)]

        name_match = None
        if name_on_doc and applicant_name:
            name_match = name_on_doc.strip().lower() == applicant_name.strip().lower()

        issues = []
        if avg_conf < 60:
            issues.append(f"low OCR confidence ({avg_conf:.0f}/100) — scan quality or print tampering suspected")
        if missing:
            issues.append(f"missing required field(s): {', '.join(missing)}")
        if tamper["tamper_suspected"]:
            issues.append("localized region with abnormal text/edge signature — possible pasted edit")
        if name_match is False:
            issues.append(f"name on document ('{name_on_doc}') does not match applicant name ('{applicant_name}')")

        valid = len(issues) == 0

        return {
            "doc_type": doc_type,
            "valid": valid,
            "ocr_confidence": round(avg_conf, 1),
            "extracted_fields": fields,
            "name_match": name_match,
            "issues": issues,
        }
