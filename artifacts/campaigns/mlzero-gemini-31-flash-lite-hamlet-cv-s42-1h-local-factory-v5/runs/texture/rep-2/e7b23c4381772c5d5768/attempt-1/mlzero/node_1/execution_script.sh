#!/bin/bash
set -e

# Define paths
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/texture/rep-2/e7b23c4381772c5d5768/attempt-1/mlzero/node_1/conda_env"
COMMON_REQS="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQS="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

# Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Install dependencies from registry files
uv pip install -r "$ML_REQS" --prerelease=allow -r "$COMMON_REQS"

# Install specific requirements for the factory module
# Ensuring numpy, pandas, and scikit-learn are present
uv pip install numpy pandas scikit-learn

echo "Environment successfully configured at $ENV_PATH"