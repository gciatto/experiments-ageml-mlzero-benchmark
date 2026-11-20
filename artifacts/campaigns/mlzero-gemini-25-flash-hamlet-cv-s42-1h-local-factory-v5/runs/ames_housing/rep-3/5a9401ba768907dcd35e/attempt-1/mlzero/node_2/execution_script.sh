#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error.

# Define environment path and requirements files
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-3/5a9401ba768907dcd35e/attempt-1/mlzero/node_2/conda_env"
COMMON_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# --- Conda Initialization and Environment Setup ---

# Ensure conda is initialized in the shell.
# This is crucial for 'conda activate' to work correctly.
# Temporarily disable 'set -u' around sourcing conda.sh to prevent issues with unset variables
# that conda might expect to be set (e.g., CONDA_SHLVL).
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    set -u
else
    echo "ERROR: conda.sh not found. Please ensure Conda is installed and initialized."
    exit 1
fi

# Create the conda environment with Python 3.11
echo "Creating conda environment at $CONDA_ENV_PATH with Python 3.11..."
conda create -p "$CONDA_ENV_PATH" python=3.11 -y

# Activate the newly created environment.
# Temporarily disable 'set -u' around 'conda activate' to prevent 'unbound variable' errors
# like CONDA_SHLVL, which can occur if the variable isn't set before activation.
echo "Activating conda environment..."
set +u
conda activate "$CONDA_ENV_PATH"
set -u
echo "Conda environment activated."

# --- Package Installation using uv ---

# Install uv (Rust-based pip replacement for faster and more reliable installs)
echo "Installing uv..."
pip install uv

# Install core packages: numpy, pandas, scikit-learn
echo "Installing numpy, pandas, scikit-learn using uv pip..."
uv pip install numpy pandas scikit-learn

# Install required packages from the specified requirements files.
# uv handles dependency resolution and ensures exact versions are met without unnecessary upgrades.
echo "Installing packages from requirements files using uv pip..."
uv pip install \
    -r "$MULTIMODAL_REQ_FILE" \
    --prerelease=allow \
    -r "$COMMON_REQ_FILE"

echo "Installation complete for environment: $CONDA_ENV_PATH"