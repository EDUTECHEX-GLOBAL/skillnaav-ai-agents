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

# Install torch CPU only
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Install remaining packages
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Pre-download the sentence-transformers model into the image layer.
# Without this, every cold container start triggers a ~200 MB network download
# before the first request can be served (10-30s on Render free tier).
# Baking it in means startup only loads the model into RAM (~2s), not download it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L3-v2')"

# Copy app
COPY . .

EXPOSE 8000

# Gunicorn flags:
#   -w 1               single worker keeps RAM under 512 MB and makes _inflight
#                      dedup work correctly (dict is per-process)
#   --timeout 120      first recommendation request can take 40-60s on cold CPU;
#                      120s gives safe headroom (old value 300 is too generous)
#   --graceful-timeout 30  let in-flight requests finish on deploy/restart
#   --keep-alive 5     matches Render load-balancer keep-alive window
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:8000", "--timeout", "120", "--graceful-timeout", "30", "--keep-alive", "5"]