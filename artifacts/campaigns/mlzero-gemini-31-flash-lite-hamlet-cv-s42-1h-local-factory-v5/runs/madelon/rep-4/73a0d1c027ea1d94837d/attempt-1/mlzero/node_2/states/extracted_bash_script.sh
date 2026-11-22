#!/bin/bash

# Define the environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/madelon/rep-4/73a0d1c027ea1d94837d/attempt-1/mlzero/node_2/conda_env"

# Create conda environment with Python 3.11
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate the environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Define requirement paths
COMMON_REQS="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AG_REQS="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# Install dependencies using uv
# --prerelease=allow is used as requested
# --no-upgrade ensures existing versions (if any) are respected
uv pip install \
    --prerelease=allow \
    -r "$AG_REQS" \
    -r "$COMMON_REQS" \
    numpy pandas scikit-learn

# Verify installation
python -c "import numpy; import pandas; import sklearn; from autogluon.multimodal import MultiModalPredictor; print('Dependencies verified successfully.')"