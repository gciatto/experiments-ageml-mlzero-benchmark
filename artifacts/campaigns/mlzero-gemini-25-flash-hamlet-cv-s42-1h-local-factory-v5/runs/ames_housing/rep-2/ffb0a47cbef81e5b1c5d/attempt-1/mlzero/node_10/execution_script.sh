#!/bin/bash
set -euo pipefail

# Define the full path for the conda environment
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-2/ffb0a47cbef81e5b1c5d/attempt-1/mlzero/node_10/conda_env"

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
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    # Ensure CONDA_SHLVL is defined to prevent 'unbound variable' errors with 'set -u'
    export CONDA_SHLVL="${CONDA_SHLVL:-0}"
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Error: conda.sh not found. Ensure Conda is installed and initialized."
    exit 1
fi

# Activate the newly created environment
echo "Activating conda environment: $CONDA_ENV_PATH"
conda activate "$CONDA_ENV_PATH"

# Install uv into the activated environment using its pip
echo "Installing uv using the environment's pip..."
"$CONDA_ENV_PATH/bin/pip" install uv

# Install numpy, pandas, and scikit-learn using uv pip
echo "Installing numpy, pandas, scikit-learn using uv pip..."
"$CONDA_ENV_PATH/bin/uv" pip install numpy pandas scikit-learn

# Install required packages from the specified requirements files using uv pip
# uv pip install by default only installs specified packages and their dependencies,
# and does not upgrade existing packages unless explicitly requested, satisfying the
# "Do NOT upgrade or reinstall autogluon.multimodal" requirement.
echo "Installing packages from requirements files using uv pip..."
"$CONDA_ENV_PATH/bin/uv" pip install \
    -r "$REQUIREMENTS_MULTIMODAL_PATH" \
    --prerelease=allow \
    -r "$REQUIREMENTS_COMMON_PATH"

echo "--- Installation complete ---"