FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install all packages (no separate torch step — torch removed from requirements)
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Pre-export sentence-transformers model to ONNX at build time.
# This converts the model weights once during docker build so container
# startup only loads the ONNX file (~60 MB) — no torch, no download.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('paraphrase-MiniLM-L3-v2', backend='onnx', model_kwargs={'provider': 'CPUExecutionProvider'}); \
m.encode('warmup'); \
print('ONNX model ready')"

# Copy app
COPY . .

EXPOSE 8000

# --timeout 0 is correct for UvicornWorker (async workers manage their own
# per-request timeouts via asyncio.wait_for; gunicorn's timeout only applies
# to sync workers and incorrectly kills async workers mid-request).
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", \
     "app:app", "--bind", "0.0.0.0:8000", \
     "--timeout", "0", "--graceful-timeout", "30", "--keep-alive", "5"]