#!/bin/bash
set -e

# Define environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/adult/rep-4/f3c7819dd2a2eace0c9e/attempt-1/mlzero/node_2/conda_env"
REQ_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_AG="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# 1. Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# 2. Activate environment
source activate "$ENV_PATH"

# 3. Install uv
pip install uv

# 4. Install dependencies
# Using --no-upgrade to respect existing versions if present, 
# and --prerelease=allow as requested.
uv pip install \
    --prerelease=allow \
    -r "$REQ_AG" \
    -r "$REQ_COMMON" \
    numpy pandas scikit-learn

echo "Environment successfully configured at $ENV_PATH"