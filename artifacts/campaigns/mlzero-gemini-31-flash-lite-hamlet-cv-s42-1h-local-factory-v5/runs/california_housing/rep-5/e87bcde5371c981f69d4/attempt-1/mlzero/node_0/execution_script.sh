#!/bin/bash
set -e

# Define paths
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/california_housing/rep-5/e87bcde5371c981f69d4/attempt-1/mlzero/node_0/conda_env"
REQ_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_AG="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

# Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Install dependencies using uv
# --no-upgrade ensures existing packages (if any) are respected
# --prerelease=allow is used as requested
uv pip install \
    numpy \
    pandas \
    scikit-learn \
    -r "$REQ_AG" \
    --prerelease=allow \
    -r "$REQ_COMMON"