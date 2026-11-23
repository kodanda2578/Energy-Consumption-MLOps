# Production-oriented Dockerfile for Energy Consumption Prediction FastAPI Service

FROM python:3.11-slim

# Prevent Python from writing bytecode files and enable unbuffered output logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MODEL_PATH=deployment_model \
    MODEL_NAME=EnergyConsumptionModel \
    MODEL_ALIAS=champion \
    PORT=8000

WORKDIR /app

# Install curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, standalone deployment model, and configuration
COPY api/ ./api/
COPY src/ ./src/
COPY configs/ ./configs/
COPY reports/ ./reports/
COPY deployment_model/ ./deployment_model/

# Create and switch to a non-root security user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose FastAPI application port
EXPOSE 8000

# Docker Healthcheck verifying FastAPI health endpoint status
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Production server execution command using environment PORT
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
