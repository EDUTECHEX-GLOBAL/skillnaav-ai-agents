FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential gcc g++ libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

# Install onnxruntime + optimum first
RUN pip install --no-cache-dir onnxruntime==1.17.3 optimum==1.19.2

# Install sentence-transformers with correct transformers version
# sentence-transformers 3.3.1 needs transformers>=4.41.0
RUN pip install --no-cache-dir "transformers==4.41.0" "sentence-transformers==3.3.1"

# Install remaining packages
RUN pip install --no-cache-dir -r requirements.txt

# Force-remove torch — it gets pulled in transitively by transformers/sentence-transformers
# but is NOT needed at runtime since we use the ONNX backend.
# onnxruntime handles all inference without torch.
RUN pip uninstall -y torch torchvision torchaudio || true

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Pre-export model to ONNX and confirm torch-free
RUN python -c "\
import importlib.util, sys; \
torch_found = importlib.util.find_spec('torch') is not None; \
print('[build] torch present:', torch_found); \
assert not torch_found, 'torch still present after uninstall — check deps'; \
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('paraphrase-MiniLM-L3-v2', backend='onnx', model_kwargs={'provider': 'CPUExecutionProvider'}); \
m.encode('warmup'); \
print('[build] ONNX model ready — torch-free confirmed')"

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", \
     "app:app", "--bind", "0.0.0.0:8000", \
     "--timeout", "0", "--graceful-timeout", "30", "--keep-alive", "5"]