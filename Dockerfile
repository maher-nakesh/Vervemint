# syntax=docker/dockerfile:1.7
#
# Vervemint: one image for the three services (API,
# web UI, Telegram bot); docker-compose.yml picks the command for each.
#
#   CPU (default): docker build -t vervemint .
#   NVIDIA GPU:    docker build -t vervemint:gpu \
#                    --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 .
#
# The embedding and reranker models are downloaded at build time, so a
# container starts fast, offline, and never depends on the Hugging Face Hub.

ARG PYTHON_VERSION=3.13

# --- Build: install dependencies and download the models -----------------
FROM python:${PYTHON_VERSION}-slim AS build

# The default PyPI torch wheel bundles ~3 GB of CUDA libraries that a CPU
# server never uses; the CPU index has a ~200 MB build instead.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt .
RUN pip install --index-url "$TORCH_INDEX_URL" \
        "torch==$(sed -n 's/^torch==//p' requirements.txt)" \
 && pip install -r requirements.txt

# Model names come from config.yaml, so the image holds exactly the
# models the app is configured to load.
COPY config.yaml .
RUN python -c "import yaml; \
from sentence_transformers import CrossEncoder, SentenceTransformer; \
c = yaml.safe_load(open('config.yaml')); \
SentenceTransformer(c['embedding_model'], device='cpu'); \
CrossEncoder(c['reranker_model'], device='cpu')"

# --- Runtime --------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Unprivileged user; it owns only the folders the app writes to.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
WORKDIR /app

COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/hf /opt/hf
COPY config.yaml ./
COPY .streamlit ./.streamlit
COPY src ./src
COPY ui ./ui
RUN mkdir -p data logs && chown app:app data logs

USER app
EXPOSE 8000 8501

# The API by default, on $PORT when the host assigns one (8000 otherwise).
# One worker: every extra worker loads its own copy of the models. Our
# middleware writes the access log, so uvicorn's is off.
# Single-container hosts that should serve the web UI as well:
#   command: python -m src.vervemint.serve
CMD ["sh", "-c", "exec uvicorn src.vervemint.api:app --host 0.0.0.0 \
--port ${PORT:-8000} --workers 1 --no-access-log \
--timeout-graceful-shutdown 30"]
