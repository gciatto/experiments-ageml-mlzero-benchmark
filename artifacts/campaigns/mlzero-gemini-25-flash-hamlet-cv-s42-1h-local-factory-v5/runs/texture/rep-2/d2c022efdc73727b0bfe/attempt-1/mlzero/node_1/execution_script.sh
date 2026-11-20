#!/bin/bash
set -e

# Define the full path for the conda environment
ENV_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/texture/rep-2/d2c022efdc73727b0bfe/attempt-1/mlzero/node_1/conda_env"

# Define paths to requirement files
COMMON_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

# Ensure the parent directory for the conda environment exists
mkdir -p "$(dirname "$ENV_DIR")"

# Create the conda environment with Python 3.11 at the specified path
echo "Creating conda environment at $ENV_DIR with Python 3.11..."
conda create -y -p "$ENV_DIR" python=3.11

# Activate the environment
echo "Activating conda environment..."
# Initialize conda for the current shell session if not already done
# This is crucial for 'conda activate' to work in non-interactive scripts
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# Install uv using the activated environment's pip
echo "Installing uv into the environment..."
pip install uv

# Install required packages from the specified requirements files and additional libraries using uv
echo "Installing packages from requirements files and additional libraries (numpy, pandas, scikit-learn, autogluon.tabular) using uv..."
uv pip install \
    -r "$ML_REQ_FILE" \
    --prerelease=allow \
    -r "$COMMON_REQ_FILE" \
    numpy pandas scikit-learn autogluon.tabular

echo "Conda environment setup complete at $ENV_DIR."