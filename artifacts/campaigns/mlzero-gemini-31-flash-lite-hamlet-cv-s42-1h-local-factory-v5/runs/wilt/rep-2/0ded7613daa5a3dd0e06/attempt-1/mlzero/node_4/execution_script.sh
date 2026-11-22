#!/bin/bash
set -e

# Define environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/wilt/rep-2/0ded7613daa5a3dd0e06/attempt-1/mlzero/node_4/conda_env"

# Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Install dependencies from registry files
# Using --system to ensure packages are installed into the prefix environment
uv pip install --system \
    -r "/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt" \
    --prerelease=allow \
    -r "/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"

# Install specific requirements for the factory module
# numpy, pandas, and scikit-learn are required for the provided WiltClassifierWrapper
uv pip install --system numpy pandas scikit-learn