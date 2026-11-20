#!/bin/bash
set -euo pipefail

# Define environment variables
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/mnist_784/rep-2/f4df8aae301901eae99a/attempt-1/mlzero/node_1/conda_env"
PYTHON_VERSION="3.11"
REQUIREMENTS_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_AUTOGLUON_TABULAR="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "--- Starting Conda environment setup ---"

# 1. Create the conda environment
echo "Creating conda environment at $CONDA_ENV_PATH with Python $PYTHON_VERSION..."
# Remove existing environment if it exists to ensure a clean slate.
if [ -d "$CONDA_ENV_PATH" ]; then
    echo "Existing environment found. Removing it..."
    conda env remove -p "$CONDA_ENV_PATH" -y || true
fi
conda create -p "$CONDA_ENV_PATH" python="$PYTHON_VERSION" -y

echo "Conda environment created."

# 2. Activate the environment
echo "Activating conda environment..."
# Initialize conda for the current shell session if not already done
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"
echo "Conda environment activated."

# 3. Install uv
echo "Installing uv into the environment..."
pip install uv
echo "uv installed."

# 4. Install required packages using uv
echo "Installing required packages using uv..."
# Install packages from the specified requirements files and additional libraries.
# The order of -r flags is maintained as specified in the prompt.
uv pip install \
    -r "$REQUIREMENTS_AUTOGLUON_TABULAR" \
    --prerelease=allow \
    -r "$REQUIREMENTS_COMMON" \
    numpy pandas scikit-learn

echo "Required packages installed."

echo "--- Conda environment setup complete ---"