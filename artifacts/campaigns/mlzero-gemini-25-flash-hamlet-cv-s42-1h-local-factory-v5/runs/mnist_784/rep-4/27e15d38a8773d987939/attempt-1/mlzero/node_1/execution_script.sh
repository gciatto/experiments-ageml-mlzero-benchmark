#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error when substituting.
# Exit if any command in a pipeline fails.
set -euxo pipefail

# Define the full path for the conda environment
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/mnist_784/rep-4/27e15d38a8773d987939/attempt-1/mlzero/node_1/conda_env"

# Define paths to the requirement files
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

echo "Starting minimal BASH INSTALLATION script for the factory module..."

# 1. Create the conda environment with Python 3.11
echo "Creating conda environment at $CONDA_ENV_PATH with Python 3.11..."
conda create -p "$CONDA_ENV_PATH" python=3.11 -y

# 2. Activate the environment
echo "Activating conda environment..."
# Initialize conda for the current shell session if not already done.
# This is crucial for 'conda activate' to work in non-interactive scripts.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

# 3. Install uv
echo "Installing uv..."
pip install uv

# 4. Install required packages using uv
# This includes numpy, pandas, scikit-learn, and packages from the specified requirement files.
# The factory module imports AutoGluon, which is expected to be covered by the provided requirement files.
echo "Installing core packages (numpy, pandas, scikit-learn) and requirements from files using uv..."
uv pip install \
    numpy \
    pandas \
    scikit-learn \
    -r "$ML_REQUIREMENTS_FILE" \
    --prerelease=allow \
    -r "$COMMON_REQUIREMENTS_FILE"

echo "Conda environment setup complete at $CONDA_ENV_PATH."