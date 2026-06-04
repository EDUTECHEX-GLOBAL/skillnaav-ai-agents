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

# Install onnxruntime + optimum FIRST, standalone.
# This ensures the ONNX deps are present before sentence-transformers resolves.
RUN pip install --no-cache-dir onnxruntime==1.17.3 optimum==1.19.2

# Install sentence-transformers WITHOUT torch.
# --no-deps skips auto-installing torch (which would add ~700MB).
# All real deps (numpy, transformers, tokenizers etc.) are in requirements.txt.
RUN pip install --no-cache-dir --no-deps "sentence-transformers==3.3.1"

# Install remaining packages
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Pre-export model to ONNX at build time so cold starts are instant.
# Verify torch is NOT present — fail the build if it snuck in.
RUN python -c "\
import importlib.util, sys; \
torch_found = importlib.util.find_spec('torch') is not None; \
print('[build] torch present:', torch_found); \
assert not torch_found, 'ERROR: torch was installed — build aborted to prevent OOM on Render'; \
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('paraphrase-MiniLM-L3-v2', backend='onnx', model_kwargs={'provider': 'CPUExecutionProvider'}); \
m.encode('warmup'); \
print('[build] ONNX model ready — torch-free image confirmed')"

# Copy app
COPY . .

EXPOSE 8000

CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", \
     "app:app", "--bind", "0.0.0.0:8000", \
     "--timeout", "0", "--graceful-timeout", "30", "--keep-alive", "5"]