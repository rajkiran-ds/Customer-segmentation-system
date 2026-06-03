# ============================================================
# CLUSTERSENSE AI — DOCKERFILE
# Build : docker build -t clustersense-ai .
# Run   : docker run clustersense-ai
# ============================================================

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install dist/*.whl || true

CMD ["python", "main.py"]