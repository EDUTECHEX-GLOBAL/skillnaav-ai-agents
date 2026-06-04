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

# Install sentence-transformers with ONNX extras BEFORE the rest,
# so the onnxruntime/optimum deps are resolved correctly.
RUN pip install --no-cache-dir "sentence-transformers[onnx]==3.3.1"

# Install remaining packages (torch is removed from requirements.txt —
# onnxruntime replaces it, saving ~300 MB of image size and runtime RAM)
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Pre-export sentence-transformers model to ONNX format at build time.
# The container loads this cached file on startup (~2s) instead of
# downloading the model on every cold boot (10-30s).
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('paraphrase-MiniLM-L3-v2', backend='onnx', model_kwargs={'provider': 'CPUExecutionProvider'}); \
m.encode('warmup'); \
print('ONNX model ready')"

# Copy app
COPY . .

EXPOSE 8000

# --timeout 0 is correct for async UvicornWorker.
# Non-zero values cause gunicorn to kill workers mid-request on long
# Claude API calls. Per-request timeouts are handled inside the app
# via asyncio.wait_for().
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", \
     "app:app", "--bind", "0.0.0.0:8000", \
     "--timeout", "0", "--graceful-timeout", "30", "--keep-alive", "5"]