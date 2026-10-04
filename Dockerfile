FROM python:3.11-slim

# System dependencies:
# - tesseract-ocr: the OCR engine the document verification agent calls via pytesseract
# - libglib2.0-0, libsm6, libxext6, libxrender1: common shared-lib dependencies that
#   opencv-python-headless still pulls in at import time on a minimal base image
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render injects $PORT at runtime; shell form lets us expand it
CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT
