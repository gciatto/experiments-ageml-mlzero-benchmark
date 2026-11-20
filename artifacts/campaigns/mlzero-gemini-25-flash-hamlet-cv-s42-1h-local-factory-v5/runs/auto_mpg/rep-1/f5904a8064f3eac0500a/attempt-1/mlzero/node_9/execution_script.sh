#!/bin/bash
set -e

# Define the target Conda environment path
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/auto_mpg/rep-1/f5904a8064f3eac0500a/attempt-1/mlzero/node_9/conda_env"

# Define paths to the requirements files
REQ_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_AUTOGLUON_TABULAR="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

# Create a new conda environment with Python 3.11
echo "Creating conda environment at $CONDA_ENV_PATH with Python 3.11..."
conda create -p "$CONDA_ENV_PATH" python=3.11 -y

# Activate Conda environment
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

# Install uv into the environment
echo "Installing uv..."
pip install uv

# Install core libraries: numpy, pandas, scikit-learn using uv
echo "Installing numpy, pandas, scikit-learn..."
uv pip install numpy pandas scikit-learn

# Install required packages from requirements files using uv
echo "Installing packages from requirements files..."
uv pip install \
    -r "$REQ_AUTOGLUON_TABULAR" \
    --prerelease=allow \
    -r "$REQ_COMMON"

echo "Conda environment setup complete."