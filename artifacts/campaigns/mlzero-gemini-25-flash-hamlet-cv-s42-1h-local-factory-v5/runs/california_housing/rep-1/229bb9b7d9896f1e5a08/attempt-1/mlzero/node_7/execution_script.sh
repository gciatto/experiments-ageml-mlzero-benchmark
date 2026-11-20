#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the target environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/california_housing/rep-1/229bb9b7d9896f1e5a08/attempt-1/mlzero/node_7/conda_env"
PYTHON_VERSION="3.11"

# Define paths to requirement files
REQUIREMENTS_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_ML="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

echo "Creating conda environment at $ENV_PATH with Python $PYTHON_VERSION..."
# Create the conda environment non-interactively
conda create -p "$ENV_PATH" python="$PYTHON_VERSION" -y

# Check if conda environment creation was successful
if [ ! -d "$ENV_PATH" ]; then
    echo "Error: Conda environment directory not found at $ENV_PATH. Conda environment creation might have failed."
    exit 1
fi

echo "Activating the conda environment: $ENV_PATH"
# Initialize conda for the current shell session if not already done.
# This is necessary for 'conda activate' to work in non-interactive scripts.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

echo "Installing uv into the environment..."
# Install uv using the environment's pip
pip install uv

echo "Installing required packages from requirements.txt files using uv..."
# Install packages from specified requirement files using uv.
# The order of requirement files is specified in the prompt.
uv pip install -r "$REQUIREMENTS_ML" --prerelease=allow -r "$REQUIREMENTS_COMMON"

echo "Installing numpy, pandas, scikit-learn, and autogluon.tabular..."
# Install additional core packages and the selected ML library (autogluon.tabular)
# These are explicitly required by the factory module and the prompt.
uv pip install numpy pandas scikit-learn autogluon.tabular

echo "Conda environment setup complete at $ENV_PATH."

# Deactivate the environment to restore the shell's original state.
# This is good practice for scripts, though not strictly required by the prompt.
conda deactivate