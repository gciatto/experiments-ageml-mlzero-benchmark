# MLZero (AutoGluon Assistant) runtime for ageml-bench.
#
# Build from the repository root:
#   docker build -f environments/mlzero/Dockerfile -t ageml-mlzero:f228cd7 .
#
# Every moving input is pinned: base image digest, MLZero commit, uv and
# Miniforge versions, retrieval-model revision, and UV_EXCLUDE_NEWER, which
# freezes PyPI resolution both here and in the per-node environments that
# MLZero creates at run time. CPU-only campaign: no CUDA base image.

ARG BASE_IMAGE=python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd
FROM ${BASE_IMAGE}

ARG MLZERO_REPO=https://github.com/autogluon/autogluon-assistant.git
ARG MLZERO_COMMIT=f228cd7028e197750391c02d94afcb7eea8da5ff
ARG UV_VERSION=0.12.21
ARG MINIFORGE_VERSION=26.7.2-0
ARG EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
ARG EMBEDDING_REVISION=a5beb1e3e68b9ab74eb54cfd186867f64f240e1a
ARG EXCLUDE_NEWER=2026-10-01T00:00:00Z

ENV DEBIAN_FRONTEND=noninteractive \
    UV_EXCLUDE_NEWER=${EXCLUDE_NEWER} \
    UV_LINK_MODE=hardlink \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# conda is required by MLZero's generated environment scripts
RUN curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

RUN pip install uv==${UV_VERSION}

RUN git clone ${MLZERO_REPO} /opt/mlzero && git -C /opt/mlzero checkout ${MLZERO_COMMIT}
COPY patches/ /opt/bench/patches/
RUN cd /opt/mlzero && for p in /opt/bench/patches/*.patch; do git apply "$p"; done
RUN uv pip install --system opencv-python-headless -e /opt/mlzero \
    && uv pip freeze --system > /opt/bench/base-freeze.txt

# retrieval model at a fixed revision, and prebuilt tutorial indices, so
# neither is downloaded or rebuilt inside the timed search
RUN python -c "from huggingface_hub import snapshot_download as s; s('${EMBEDDING_MODEL}', revision='${EMBEDDING_REVISION}')" \
    && python -c "from autogluon.assistant.tools_registry.indexing import TutorialIndexer as T; t = T(); t.build_indices(); t.save_indices()"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY environments/mlzero/supervise.py environments/mlzero/warm_cache.sh /opt/bench/
COPY environments/mlzero/guard/ /opt/bench/guard/
RUN printf '%s\n' \
      "mlzero_commit=${MLZERO_COMMIT}" "uv=${UV_VERSION}" "miniforge=${MINIFORGE_VERSION}" \
      "embedding_model=${EMBEDDING_MODEL}@${EMBEDDING_REVISION}" "exclude_newer=${EXCLUDE_NEWER}" \
      > /opt/bench/build-info.txt \
    && sha256sum /opt/bench/patches/*.patch >> /opt/bench/build-info.txt

WORKDIR /bench
