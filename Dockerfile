# ─── Build Stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ─── Production Stage ────────────────────────────────────────────────────────
FROM python:3.11-slim AS production

LABEL maintainer="Smart Ticket Router"
LABEL description="Automated IT support ticket classification and routing"
LABEL version="1.0.0"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p data models

# Environment configuration
ENV MODEL_PATH=/app/models/classifier.pkl
ENV DB_PATH=/app/data/tickets.db
ENV DATA_PATH=/app/data/tickets.csv
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH=/root/.local/bin:$PATH

# Train the model during build (so it's baked in)
RUN python src/train_model.py

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Start API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
