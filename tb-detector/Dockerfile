# ── Stage 1: base Python + system deps ────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system libraries needed by OpenCV / Pillow / TF
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: Python deps ───────────────────────────────────────────────────────
FROM base AS deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# ── Stage 3: final image ───────────────────────────────────────────────────────
FROM deps AS final
COPY . .

# Create a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

ENV PORT=8080 \
    MODEL_PATH=/tb-detector/models/best_cnn.keras \
    THRESHOLD=0.5

EXPOSE 8080

# Gunicorn: 2 workers, 120s timeout for model inference
CMD ["gunicorn", "--bind", "0.0.0.0:8080", \
     "--workers", "2", "--timeout", "120", \
     "--access-logfile", "-", "--error-logfile", "-", \
     "app:app"]
