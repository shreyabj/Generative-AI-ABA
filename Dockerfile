# ============================================================================
# SETU - single-image deployment: React UI built, then served by FastAPI.
# One container, one URL, one port. Works on Render, Fly.io, Railway, Cloud Run
# and Hugging Face Spaces without modification.
# ============================================================================

# ---------------------------------------------------------------- stage 1: web
FROM node:22-alpine AS web

WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm ci --no-audit --no-fund 2>/dev/null || npm install --no-audit --no-fund

COPY web/ ./
RUN npm run build

# -------------------------------------------------------------- stage 2: server
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/server \
    HF_HOME=/app/.cache/huggingface \
    OMP_NUM_THREADS=4 \
    SETU_HOST=0.0.0.0 \
    SETU_PORT=8000

WORKDIR /app

# libgomp is required by onnxruntime; curl is used by the healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first so the layer caches independently of source changes.
COPY pyproject.toml README.md ./
COPY server/setu/__init__.py server/setu/__init__.py
RUN pip install --upgrade pip \
 && pip install ".[ingest]"

COPY server/ ./server/
COPY scripts/ ./scripts/
COPY data/sample/ ./data/sample/
# A copy outside /app/data: platforms that mount an *empty* volume at /app/data (Fly,
# Render) hide the baked files, so first boot rebuilds the sample index from here.
# The metadata DB (data/setu.db) and collection indexes (data/collections/) live on
# the volume and survive restarts and image rebuilds.
COPY data/sample/ ./seed/sample/
COPY data/sources/ ./seed/sources/
ENV SETU_SAMPLE_CORPUS_FILE=/app/seed/sample/corpus.jsonl \
    SETU_SOURCES_DIR=/app/seed/sources
COPY docs/ ./docs/
COPY --from=web /web/dist ./web/dist

# Bake the encoder into the image so a cold container does not download 120 MB
# on its first request, and build the sample index so the app is useful the
# instant it boots. Mount a volume at /app/data to persist a larger ingest.
RUN python -c "from fastembed import TextEmbedding; \
TextEmbedding(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')" \
 && python -m setu.cli build --sample

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

CMD ["python", "-m", "uvicorn", "setu.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
