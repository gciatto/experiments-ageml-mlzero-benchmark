#!/bin/bash
set -euo pipefail

# --- Configuration ---
# Path to the base conda installation. Adjust this if your conda installation is in a different location.
# Common paths include /opt/conda, /usr/local/conda, or ~/miniconda3.
CONDA_BASE_PATH="/opt/conda"

# The full path for the new conda environment as specified in the prompt.
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-4/2638bba56a1dcdafac5f/attempt-1/mlzero/node_2/conda_env"

# Paths to the requirements files as specified in the prompt.
COMMON_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# --- Conda Initialization ---
# Ensure conda is properly initialized in the current shell session.
# This addresses the "conda activate script not found" error from previous attempts.
if [ -f "$CONDA_BASE_PATH/etc/profile.d/conda.sh" ]; then
    echo "Sourcing conda initialization script from $CONDA_BASE_PATH..."
    source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
else
    echo "Error: Conda initialization script not found at $CONDA_BASE_PATH/etc/profile.d/conda.sh" >&2
    echo "Please ensure CONDA_BASE_PATH is set correctly and conda is installed." >&2
    exit 1
fi

# --- Environment Setup ---
echo "Attempting to create or update conda environment at: $ENV_PATH"

# Remove existing environment if it exists to ensure a clean slate for installation.
# This helps prevent conflicts and ensures all packages are installed according to the requirements.
if conda env list | grep -q "$ENV_PATH"; then
    echo "Existing environment found at $ENV_PATH. Removing it to ensure a clean installation."
    conda env remove -p "$ENV_PATH" -y || { echo "Failed to remove existing conda environment." >&2; exit 1; }
fi

# Create the new conda environment with the specified Python version.
echo "Creating new conda environment with Python 3.11..."
conda create -p "$ENV_PATH" python=3.11 -y || { echo "Failed to create conda environment." >&2; exit 1; }

# Activate the newly created environment.
echo "Activating conda environment: $ENV_PATH"
conda activate "$ENV_PATH" || { echo "Failed to activate conda environment." >&2; exit 1; }

# --- Package Installation ---
echo "Installing 'uv' into the environment for efficient package management..."
pip install uv || { echo "Failed to install 'uv'." >&2; exit 1; }

echo "Installing core packages (numpy, pandas, scikit-learn) using uv..."
uv pip install numpy pandas scikit-learn || { echo "Failed to install core packages." >&2; exit 1; }

echo "Installing required packages from specified requirements files using uv..."
# The '--prerelease=allow' flag is included as per the prompt.
# 'uv' inherently respects exact package versions specified in requirements files.
# If a package is already installed at the exact required version, 'uv' will typically skip reinstallation,
# aligning with the "Do NOT upgrade or reinstall autogluon.multimodal if it's already at the correct version" requirement.
uv pip install -r "$MULTIMODAL_REQ_FILE" --prerelease=allow -r "$COMMON_REQ_FILE" || { echo "Failed to install packages from requirements files." >&2; exit 1; }

echo "Conda environment setup complete at $ENV_PATH."
echo "To activate this environment manually, run: source $CONDA_BASE_PATH/etc/profile.d/conda.sh && conda activate $ENV_PATH"