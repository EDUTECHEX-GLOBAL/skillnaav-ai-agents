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

# Pre-download sentence-transformers model into the image so cold starts
# don't pay a 200MB download penalty on every container boot.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L3-v2')"

# Copy app
COPY . .

EXPOSE 8000

# --timeout 0 is correct for UvicornWorker (async).
# The gunicorn timeout is designed for SYNC workers — it kills a worker that
# hasn't responded to the master heartbeat within N seconds. With an async
# worker the event loop IS the heartbeat; a long-running coroutine does not
# block it and should never trigger a kill. Setting any non-zero value causes
# gunicorn to kill the worker mid-request whenever a single request (e.g. a
# Claude API call) takes longer than the timeout — exactly the restart loop
# seen in production. Per-request timeouts are enforced inside the app with
# asyncio.wait_for(), not at the gunicorn level.
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:8000", "--timeout", "0", "--graceful-timeout", "30", "--keep-alive", "5"]