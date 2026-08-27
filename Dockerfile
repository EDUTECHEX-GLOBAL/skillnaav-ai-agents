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

COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2+cpu

# Install Sentence Transformers
RUN pip install --no-cache-dir \
    sentence-transformers==3.3.1

# Install application dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Warm up embedding model during build
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('paraphrase-MiniLM-L3-v2'); \
m.encode('warmup'); \
print('[build] model ready')"

# Copy application code
COPY . .

EXPOSE 8000

CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:8000", "--timeout", "0", "--graceful-timeout", "30", "--keep-alive", "5"]
