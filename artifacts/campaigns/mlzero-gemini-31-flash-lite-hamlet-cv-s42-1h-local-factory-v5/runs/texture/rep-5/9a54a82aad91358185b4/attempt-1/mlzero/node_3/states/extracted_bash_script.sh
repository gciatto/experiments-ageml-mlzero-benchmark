#!/bin/bash
set -e

# Define paths
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/texture/rep-5/9a54a82aad91358185b4/attempt-1/mlzero/node_3/conda_env"
REQ_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_AG="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

# Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate the environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Prepare environment with core libraries
uv pip install numpy pandas scikit-learn

# Install required packages from registry
# --no-upgrade prevents reinstalling/upgrading if already satisfied
# --prerelease=allow is required by the prompt
uv pip install \
    --no-upgrade \
    --prerelease=allow \
    -r "$REQ_AG" \
    -r "$REQ_COMMON"

echo "Environment installation complete."