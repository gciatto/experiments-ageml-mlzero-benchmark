#!/bin/bash
set -euo pipefail # Exit immediately if a command exits with a non-zero status, exit on unset variables, propagate errors in pipelines.

# Define paths for the conda environment and requirements files
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/madelon/rep-4/3f1d9fe8ce5e73e90c19/attempt-1/mlzero/node_2/conda_env"
REQ_FILE_AUTOGLUON_MM="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"
REQ_FILE_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"

echo "Starting minimal BASH INSTALLATION script for the factory module..."

# 1. Ensure a clean conda environment by removing any existing one at the target path
if [ -d "$CONDA_ENV_PATH" ]; then
    echo "Existing conda environment found at '$CONDA_ENV_PATH'. Removing it for a clean installation..."
    rm -rf "$CONDA_ENV_PATH"
fi

# 2. Create the conda environment with Python 3.11
echo "Creating conda environment at '$CONDA_ENV_PATH' with Python 3.11..."
conda create -p "$CONDA_ENV_PATH" python=3.11 -y

# 3. Activate the newly created environment
echo "Activating conda environment..."
source "$CONDA_ENV_PATH/bin/activate"

# 4. Install uv (a fast Python package installer and resolver)
echo "Installing uv..."
pip install uv

# 5. Install core ML libraries: numpy, pandas, scikit-learn
echo "Installing numpy, pandas, scikit-learn using uv..."
uv pip install numpy pandas scikit-learn

# 6. Install required packages from the specified requirements files
# The order and --prerelease=allow flag are crucial as per instructions.
echo "Installing packages from requirements files using uv..."
uv pip install \
    -r "$REQ_FILE_AUTOGLUON_MM" \
    --prerelease=allow \
    -r "$REQ_FILE_COMMON"

echo "Conda environment setup complete at '$CONDA_ENV_PATH'."
echo "The environment is now ready for the factory module."