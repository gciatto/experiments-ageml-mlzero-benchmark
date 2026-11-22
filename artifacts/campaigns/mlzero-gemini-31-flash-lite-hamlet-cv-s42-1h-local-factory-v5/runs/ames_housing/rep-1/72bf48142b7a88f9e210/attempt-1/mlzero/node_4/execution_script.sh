#!/bin/bash
set -e

# Define environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-1/72bf48142b7a88f9e210/attempt-1/mlzero/node_4/conda_env"

# Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Install core dependencies
uv pip install numpy pandas scikit-learn

# Install requirements from specified files
# --no-upgrade ensures we do not reinstall/upgrade if already satisfied
# --prerelease=allow is used as requested
uv pip install \
    --no-upgrade \
    --prerelease=allow \
    -r /root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt \
    -r /root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt