#!/bin/bash
set -euo pipefail

# Define the full path for the conda environment
# The target path is updated to 'node_6' as specified in the prompt.
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-2/ffb0a47cbef81e5b1c5d/attempt-1/mlzero/node_6/conda_env"

# Define paths to the requirements files
REQUIREMENTS_COMMON_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_MULTIMODAL_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

echo "--- Starting minimal BASH INSTALLATION script ---"
echo "Target Conda environment path: $CONDA_ENV_PATH"

# Ensure the parent directory for the conda environment exists
mkdir -p "$(dirname "$CONDA_ENV_PATH")"

# Create a new conda environment with Python 3.11
echo "Creating conda environment at '$CONDA_ENV_PATH' with Python 3.11..."
conda create -y -p "$CONDA_ENV_PATH" python=3.11

# Initialize conda for the current shell session if not already done
# This is crucial for 'conda activate' to work in non-interactive scripts.
# The check for conda.sh existence ensures robustness.
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Error: conda.sh not found. Ensure Conda is installed and initialized."
    exit 1
fi

# Activate the newly created environment
echo "Activating conda environment: $CONDA_ENV_PATH"
conda activate "$CONDA_ENV_PATH"

# Install uv into the activated environment
echo "Installing uv..."
pip install uv

# Install numpy, pandas, and scikit-learn using uv pip
echo "Installing numpy, pandas, scikit-learn using uv pip..."
uv pip install numpy pandas scikit-learn

# Install required packages from the specified requirements files using uv pip.
# The command uses the exact paths and flags provided in the instructions.
# 'uv' handles dependency resolution efficiently and avoids unnecessary upgrades
# if packages already satisfy the requirements.
echo "Installing packages from requirements files using uv pip..."
uv pip install \
    -r "$REQUIREMENTS_MULTIMODAL_PATH" \
    --prerelease=allow \
    -r "$REQUIREMENTS_COMMON_PATH"

echo "--- Installation complete ---"