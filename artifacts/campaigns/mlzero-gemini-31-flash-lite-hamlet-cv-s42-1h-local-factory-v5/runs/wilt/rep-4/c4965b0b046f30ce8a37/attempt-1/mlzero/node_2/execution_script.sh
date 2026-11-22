#!/bin/bash
set -e

# Define paths
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/wilt/rep-4/c4965b0b046f30ce8a37/attempt-1/mlzero/node_2/conda_env"
REQ_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_AG="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Install core dependencies
uv pip install numpy pandas scikit-learn

# Install requirements from registry
# --no-upgrade ensures existing versions (like autogluon.multimodal) are respected if already present
uv pip install \
    --no-upgrade \
    --prerelease=allow \
    -r "$REQ_AG" \
    -r "$REQ_COMMON"