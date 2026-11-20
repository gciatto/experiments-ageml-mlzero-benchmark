#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Define paths
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/madelon/rep-1/b94fca9f26a3228d15f1/attempt-1/mlzero/node_1/conda_env"
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# Ensure the parent directory for the conda environment exists
mkdir -p "$(dirname "$CONDA_ENV_PATH")"

# Create the conda environment with Python 3.11
# This command will fail if the directory already exists and is not empty,
# which will be caught by 'set -e' and propagate the error.
echo "Creating conda environment at $CONDA_ENV_PATH with Python 3.11..."
conda create -p "$CONDA_ENV_PATH" python=3.11 -y

# Activate the newly created environment
echo "Activating the conda environment..."
source "$CONDA_ENV_PATH/bin/activate"

# Install uv using the environment's pip
echo "Installing uv..."
pip install uv

# Install core packages using uv
echo "Installing numpy, pandas, scikit-learn..."
uv pip install numpy pandas scikit-learn

# Install required packages from the specified requirements files using uv
# The order of requirements files is as specified in the prompt.
echo "Installing required packages from requirements files using uv..."
uv pip install \
    -r "$MULTIMODAL_REQUIREMENTS_FILE" \
    --prerelease=allow \
    -r "$COMMON_REQUIREMENTS_FILE"

echo "Conda environment setup complete."