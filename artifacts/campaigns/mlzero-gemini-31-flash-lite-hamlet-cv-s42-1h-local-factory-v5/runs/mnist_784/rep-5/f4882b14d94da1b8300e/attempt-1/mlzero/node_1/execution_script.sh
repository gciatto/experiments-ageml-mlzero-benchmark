#!/bin/bash

# Define the environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/mnist_784/rep-5/f4882b14d94da1b8300e/attempt-1/mlzero/node_1/conda_env"

# Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate the environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Install required packages using uv
# Note: Using the paths provided in the prompt
COMMON_REQ="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQ="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

uv pip install -r "$ML_REQ" --prerelease=allow -r "$COMMON_REQ"

# Install additional packages required by the factory module
# numpy, pandas, and scikit-learn are typically in requirements, 
# but we ensure they are present along with autogluon
uv pip install numpy pandas scikit-learn autogluon