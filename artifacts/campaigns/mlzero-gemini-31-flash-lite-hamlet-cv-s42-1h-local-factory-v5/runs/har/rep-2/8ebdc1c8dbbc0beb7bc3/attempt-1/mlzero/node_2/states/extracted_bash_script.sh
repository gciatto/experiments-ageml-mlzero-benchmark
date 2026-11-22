#!/bin/bash
set -e

# Define paths
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/har/rep-2/8ebdc1c8dbbc0beb7bc3/attempt-1/mlzero/node_2/conda_env"
REQ_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_MM="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Install dependencies using uv
# --no-upgrade ensures existing versions are respected if already satisfied
# --prerelease=allow is used as requested
uv pip install \
    --no-upgrade \
    --prerelease=allow \
    -r "$REQ_MM" \
    -r "$REQ_COMMON" \
    numpy \
    pandas \
    scikit-learn