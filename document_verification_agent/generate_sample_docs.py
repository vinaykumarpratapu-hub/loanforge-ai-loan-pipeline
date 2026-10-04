"""
Generates synthetic, clearly-fictional demo documents (ID proof + income
proof) for the LoanForge pipeline demo — not real customer documents.
These exist purely so the document verification agent has something to
read and validate during a live run-through of the pipeline.
"""
from PIL import Image, ImageDraw, ImageFont
import os

OUT = "data/sample_docs"
os.makedirs(OUT, exist_ok=True)

# Adjust these font paths for your deployment environment, or bundle a .ttf
# file with the repo instead of relying on system fonts.
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def make_id_card(path, name, dob, id_number, address, tamper=False):
    img = Image.new("RGB", (640, 400), "white")
    d = ImageDraw.Draw(img)
    bold = ImageFont.truetype(FONT_BOLD, 22)
    reg = ImageFont.truetype(FONT_REG, 18)
    small = ImageFont.truetype(FONT_REG, 14)

    d.rectangle([0, 0, 639, 60], fill=(20, 60, 110))
    d.text((20, 15), "LOANFORGE DEMO - NATIONAL ID (SAMPLE)", font=bold, fill="white")
    d.text((20, 90), f"Name: {name}", font=reg, fill="black")
    d.text((20, 125), f"Date of Birth: {dob}", font=reg, fill="black")
    d.text((20, 160), f"ID Number: {id_number}", font=reg, fill="black")
    d.text((20, 195), f"Address: {address}", font=reg, fill="black")
    d.text((20, 360), "This is a synthetic sample document for demo purposes only.", font=small, fill=(120, 120, 120))

    if tamper:
        # Simulate a tampered document: inconsistent font size + overlapping edit on the name field
        d.rectangle([110, 85, 400, 112], fill="white")
        bigger = ImageFont.truetype(FONT_BOLD, 26)
        d.text((112, 83), "Rohit Verma", font=bigger, fill=(40, 40, 40))

    img.save(path)


def make_income_proof(path, name, employer, monthly_income, month):
    img = Image.new("RGB", (640, 420), "white")
    d = ImageDraw.Draw(img)
    bold = ImageFont.truetype(FONT_BOLD, 22)
    reg = ImageFont.truetype(FONT_REG, 18)
    small = ImageFont.truetype(FONT_REG, 14)

    d.rectangle([0, 0, 639, 60], fill=(20, 110, 70))
    d.text((20, 15), "SALARY SLIP (SAMPLE)", font=bold, fill="white")
    d.text((20, 90), f"Employee Name: {name}", font=reg, fill="black")
    d.text((20, 125), f"Employer: {employer}", font=reg, fill="black")
    d.text((20, 160), f"Pay Period: {month}", font=reg, fill="black")
    d.text((20, 195), f"Net Monthly Pay: Rs. {monthly_income:,}", font=reg, fill="black")
    d.text((20, 380), "This is a synthetic sample document for demo purposes only.", font=small, fill=(120, 120, 120))
    img.save(path)


# Rahul Sharma — our consistent demo applicant, genuine document set
make_id_card(
    f"{OUT}/rahul_id_genuine.png",
    name="Rahul Sharma", dob="14-Mar-1994", id_number="DEMO-ID-884213",
    address="221 MG Road, Hyderabad, Telangana",
)
make_income_proof(
    f"{OUT}/rahul_income_genuine.png",
    name="Rahul Sharma", employer="Acme Analytics Pvt Ltd",
    monthly_income=65000, month="August 2026",
)

# A second, deliberately tampered ID to show the verification agent catching it
make_id_card(
    f"{OUT}/sample_id_tampered.png",
    name="Rahul Sharma", dob="14-Mar-1994", id_number="DEMO-ID-884213",
    address="221 MG Road, Hyderabad, Telangana",
    tamper=True,
)

print("Generated sample docs in", OUT)
print(os.listdir(OUT))
