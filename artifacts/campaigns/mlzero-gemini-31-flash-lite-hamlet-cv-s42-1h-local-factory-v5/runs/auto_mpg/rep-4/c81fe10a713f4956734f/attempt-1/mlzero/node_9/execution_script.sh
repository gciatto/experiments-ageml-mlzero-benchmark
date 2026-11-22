#!/bin/bash

# Define the environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/auto_mpg/rep-4/c81fe10a713f4956734f/attempt-1/mlzero/node_9/conda_env"

# Create conda environment with Python 3.11
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate the environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Install dependencies from registry files
# Using --prerelease=allow as requested
uv pip install \
    -r "/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt" \
    -r "/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt" \
    --prerelease=allow

# Install specific required packages for the factory module
# Ensuring numpy, pandas, scikit-learn, torch, and autogluon are present
uv pip install numpy pandas scikit-learn torch autogluon