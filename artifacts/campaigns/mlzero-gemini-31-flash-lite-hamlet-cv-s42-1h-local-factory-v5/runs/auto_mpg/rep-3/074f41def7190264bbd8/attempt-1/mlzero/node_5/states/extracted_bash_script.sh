#!/bin/bash
set -e

# Define environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/auto_mpg/rep-3/074f41def7190264bbd8/attempt-1/mlzero/node_5/conda_env"

# Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate the environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Install required packages from registry files
# Note: Escaping spaces in paths for shell compatibility
uv pip install -r "/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt" --prerelease=allow -r "/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"

# Install specific dependencies for the factory module
uv pip install numpy pandas scikit-learn