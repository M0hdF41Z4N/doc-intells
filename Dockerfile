# ============================================================
# Stage 1 — Builder
# Install system deps + Python packages in an isolated layer
# so the final runtime image stays lean.
# ============================================================
FROM python:3.11-slim AS builder

# camelot-py[cv] requires Ghostscript and OpenCV native libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ghostscript \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the dependency manifest first so Docker layer-caches
# the heavy pip install step unless requirements change.
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ============================================================
# Stage 2 — Runtime
# Lean final image: copy installed packages + app source only.
# ============================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="Patronus Document Intelligence"
LABEL description="Autonomous Document Intelligence API (FastAPI + LangGraph + Qdrant)"

# Same system libs needed at runtime by camelot / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ghostscript \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages installed in the builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source
COPY agents/    ./agents/
COPY api/       ./api/
COPY etl/       ./etl/
COPY services/  ./services/
COPY .env       ./.env

# Create directories the app writes to at runtime
RUN mkdir -p /app/data /app/logs

# Expose the FastAPI port
EXPOSE 8000

# Healthcheck: poll the /health endpoint every 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run with uvicorn. Use --workers 1 in the container; scale
# horizontally at the orchestration layer (Kubernetes / ECS).
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
