#!/bin/bash
set -e

# Define environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-5/c75197609fe05a0e77bd/attempt-1/mlzero/node_2/conda_env"

# Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Install required packages from registry files
# Using --prerelease=allow as requested
uv pip install -r /root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine\ learning/requirements.txt \
    --prerelease=allow \
    -r /root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt

# Install specific dependencies required by the factory module
# numpy, pandas, and scikit-learn are required for the ModelWrapper
uv pip install numpy pandas scikit-learn