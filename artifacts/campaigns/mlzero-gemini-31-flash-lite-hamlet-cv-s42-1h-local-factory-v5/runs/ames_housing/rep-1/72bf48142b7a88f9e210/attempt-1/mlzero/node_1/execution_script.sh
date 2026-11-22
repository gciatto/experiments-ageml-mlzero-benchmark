#!/bin/bash

# Define the target environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-1/72bf48142b7a88f9e210/attempt-1/mlzero/node_1/conda_env"

# Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate the environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Define requirement paths
COMMON_REQS="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AG_REQS="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# Install base dependencies and ML library requirements
# Using --no-upgrade to respect existing versions if present, and --prerelease=allow as requested
uv pip install numpy pandas scikit-learn
uv pip install --prerelease=allow -r "$AG_REQS" -r "$COMMON_REQS"