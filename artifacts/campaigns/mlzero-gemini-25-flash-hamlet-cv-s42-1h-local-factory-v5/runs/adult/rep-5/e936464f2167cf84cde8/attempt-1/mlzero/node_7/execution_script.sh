#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the full path for the conda environment
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/adult/rep-5/e936464f2167cf84cde8/attempt-1/mlzero/node_7/conda_env"

# Define paths to requirement files
REQUIREMENTS_ML_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"
REQUIREMENTS_COMMON_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"

echo "Creating conda environment at $CONDA_ENV_PATH with Python 3.11..."
# Create the conda environment non-interactively
conda create -p "$CONDA_ENV_PATH" python=3.11 -y

echo "Activating the environment..."
# Initialize conda for the current shell session if not already done, then activate the environment.
# This ensures 'conda activate' works in non-interactive scripts.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

echo "Installing uv..."
# Install uv using the environment's pip
pip install uv

echo "Installing required packages from requirements.txt files using uv..."
# Install packages from the specified requirements files using uv
uv pip install -r "$REQUIREMENTS_ML_PATH" --prerelease=allow -r "$REQUIREMENTS_COMMON_PATH"

echo "Installing numpy, pandas, scikit-learn, and autogluon.tabular..."
# Install additional packages explicitly mentioned or implied by the factory module
uv pip install numpy pandas scikit-learn autogluon.tabular

echo "Conda environment setup complete."