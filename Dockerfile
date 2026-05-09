# Builder stage: compile wheels and dependencies
FROM python:3.11-slim as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements and install to custom prefix
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        --prefix=/install \
        -r requirements.txt

# Runtime stage: minimal image with only compiled packages
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy compiled packages from builder
COPY --from=builder /install /install

# Set Python path to use installed packages
ENV PYTHONPATH=/install/lib/python3.11/site-packages:$PYTHONPATH \
    PATH=/install/bin:$PATH

# Create non-root user
RUN useradd -u 1001 -M -s /usr/sbin/nologin appuser

# Copy application code preserving directory structure
# app.py expects BASE_DIR to be 2 levels up (../.. from api/app.py)
COPY api/app.py api/

# Create data directories (will be mounted at runtime)
RUN mkdir -p data/processed data/interim

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

USER appuser

# Configuration via environment variables
ENV APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_WORKERS=1 \
    LOG_LEVEL=info

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${APP_PORT}/health || exit 1

# Set working directory to api folder for app module discovery
WORKDIR /app/api

# Run FastAPI app with environment variable expansion
CMD sh -c "uvicorn app:app --host ${APP_HOST} --port ${APP_PORT} --workers ${APP_WORKERS} --log-level ${LOG_LEVEL}"
