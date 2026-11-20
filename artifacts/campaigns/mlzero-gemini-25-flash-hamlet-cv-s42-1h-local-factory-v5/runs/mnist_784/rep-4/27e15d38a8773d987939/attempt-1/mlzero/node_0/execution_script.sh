#!/bin/bash
set -e

# Define paths
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/mnist_784/rep-4/27e15d38a8773d987939/attempt-1/mlzero/node_0/conda_env"
COMMON_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AUTOGLUON_TABULAR_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

# Ensure the parent directory for the conda environment exists
mkdir -p "$(dirname "$CONDA_ENV_PATH")"

# Create the conda environment with Python 3.11
echo "Creating conda environment at $CONDA_ENV_PATH with Python 3.11..."
conda create -p "$CONDA_ENV_PATH" python=3.11 -y

# Initialize conda for script usage
# This makes 'conda activate' available in non-interactive shells
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found. Ensure Conda is installed and initialized."
    exit 1
fi

# Activate the newly created environment
echo "Activating conda environment..."
conda activate "$CONDA_ENV_PATH"

# Install uv into the activated environment
echo "Installing uv into the environment..."
pip install uv

# Install required packages from requirements files using uv
echo "Installing required packages from requirements files using uv..."
uv pip install -r "$AUTOGLUON_TABULAR_REQ_PATH" --prerelease=allow -r "$COMMON_REQ_PATH"

# Explicitly install numpy, pandas, and scikit-learn as requested
# These are often dependencies of autogluon.tabular, but explicitly installing ensures compliance.
echo "Ensuring numpy, pandas, and scikit-learn are installed..."
uv pip install numpy pandas scikit-learn

echo "Conda environment setup complete."