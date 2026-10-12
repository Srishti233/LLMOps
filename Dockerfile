# syntax=docker/dockerfile:1
# Stage 1: build a virtualenv with CPU-only PyTorch (no compilers needed in the final image)
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Stage 2: MLflow tracking server / model registry UI (used by docker compose as the `mlflow` service)
FROM python:3.11-slim AS mlflow
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN pip install mlflow==2.16.2
WORKDIR /workspace
EXPOSE 5000

# Stage 3 (default target): the API
FROM python:3.11-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    LLMOPS_HOME=/app HOME=/tmp TOKENIZERS_PARALLELISM=false
RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY configs ./configs
COPY prompts ./prompts
COPY src ./src
COPY scripts ./scripts
RUN mkdir -p models/hf_cache models/production data reports mlruns && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=240s --retries=20 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4)" || exit 1
CMD ["uvicorn", "src.serve.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
