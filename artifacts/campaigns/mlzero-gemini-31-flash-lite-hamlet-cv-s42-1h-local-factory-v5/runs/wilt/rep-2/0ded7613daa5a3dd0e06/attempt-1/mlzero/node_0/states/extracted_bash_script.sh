#!/bin/bash
set -e

# Define paths
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/wilt/rep-2/0ded7613daa5a3dd0e06/attempt-1/mlzero/node_0/conda_env"
REQ_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_TABULAR="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

# Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate environment
source activate "$ENV_PATH"

# Install base dependencies
pip install numpy pandas scikit-learn uv

# Install requirements using uv
# --no-upgrade ensures existing versions are respected if satisfied
uv pip install \
    --no-upgrade \
    --prerelease=allow \
    -r "$REQ_TABULAR" \
    -r "$REQ_COMMON"