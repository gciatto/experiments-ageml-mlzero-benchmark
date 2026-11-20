#!/bin/bash
set -euo pipefail

# Define environment path and requirements file paths
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/california_housing/rep-2/353a64eebb2bcfbd2485/attempt-1/mlzero/node_3/conda_env"
REQ_FILE_TABULAR="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"
REQ_FILE_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"

echo "--- Starting Conda environment setup ---"

# Create the Conda environment with Python 3.11
echo "Creating conda environment at $ENV_PATH with Python 3.11..."
conda create -p "$ENV_PATH" python=3.11 -y || { echo "ERROR: Failed to create conda environment."; exit 1; }

# Activate the environment
echo "Activating conda environment..."
# Initialize conda for the current shell session if not already done
source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "ERROR: Failed to source conda.sh. Ensure conda is installed and configured."; exit 1; }
conda activate "$ENV_PATH" || { echo "ERROR: Failed to activate conda environment."; exit 1; }

# Install uv
echo "Installing uv into the environment..."
pip install uv || { echo "ERROR: Failed to install uv."; exit 1; }

# Install numpy, pandas, scikit-learn as explicitly requested
echo "Installing numpy, pandas, scikit-learn..."
uv pip install numpy pandas scikit-learn || { echo "ERROR: Failed to install numpy, pandas, scikit-learn."; exit 1; }

# Install required packages from requirements files using uv
echo "Installing required packages from requirements files using uv..."
# The --prerelease=allow flag is important for AutoGluon beta versions.
# uv respects exact versions in requirements.txt and will not upgrade/reinstall
# if the package is already at the specified version.
uv pip install -r "$REQ_FILE_TABULAR" --prerelease=allow -r "$REQ_FILE_COMMON" || { echo "ERROR: Failed to install packages from requirements files."; exit 1; }

echo "--- Conda environment setup complete ---"