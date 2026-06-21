# syntax=docker/dockerfile:1

# Multi-stage build for the claim-approval model-serving API.
# Dependencies are installed with uv from the committed uv.lock (reproducible),
# then the resolved virtualenv is copied into a slim runtime image.
#
# Build:  docker build -t bolttech-prac .
# Run:    docker run --rm -p 8000:8000 bolttech-prac
#         # -> serving API at http://localhost:8000  (docs at /docs)
#
# The image ships the trained model (models/best_model.joblib) so the API works
# out of the box. To retrain inside the container instead, override the command:
#   docker run --rm bolttech-prac python src/run_pipeline.py

# ---------------------------------------------------------------------------
# Stage 1 — builder: resolve and install dependencies with uv
# ---------------------------------------------------------------------------
# The uv image is built FROM python:3.12-slim-bookworm, so the interpreter path
# matches the runtime stage and the copied .venv stays valid.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install only the locked dependencies first (cached layer; the project itself
# is package=false, so it is not installed — only its deps).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# ---------------------------------------------------------------------------
# Stage 2 — runtime: slim image with the venv + application code
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# libgomp1 is the OpenMP runtime required by LightGBM (and XGBoost).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bring in the resolved virtual environment from the builder.
COPY --from=builder /app/.venv /app/.venv

# Application code + artifacts needed at runtime.
COPY src/ ./src/
COPY models/ ./models/
COPY reports/ ./reports/
COPY data/ ./data/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Liveness check against the API's /health endpoint (no curl in slim image).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Default: the production serving API (serve_api.py). Override the command to
# run the dashboard backend (serve:app) or the training pipeline instead.
CMD ["uvicorn", "serve_api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
