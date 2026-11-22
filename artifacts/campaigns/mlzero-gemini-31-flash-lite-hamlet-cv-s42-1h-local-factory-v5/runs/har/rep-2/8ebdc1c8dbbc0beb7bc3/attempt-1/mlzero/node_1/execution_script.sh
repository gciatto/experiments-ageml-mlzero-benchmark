#!/bin/bash
set -e

# Define environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/har/rep-2/8ebdc1c8dbbc0beb7bc3/attempt-1/mlzero/node_1/conda_env"

# Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Install base requirements and ML dependencies
# Using the paths provided in the prompt
COMMON_REQ="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQ="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

uv pip install -r "$ML_REQ" --prerelease=allow -r "$COMMON_REQ"

# Install specific dependencies required by the factory module
# numpy, pandas, and scikit-learn are required for the HARModelWrapper
uv pip install numpy pandas scikit-learn