#!/bin/bash
set -euo pipefail

# --- Configuration ---
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/texture/rep-5/f597436d592bb5952d1c/attempt-1/mlzero/node_1/conda_env"
PYTHON_VERSION="3.11"
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

echo "Starting minimal BASH INSTALLATION script for factory module."

# --- 1. Conda Initialization ---
# Source the conda.sh script unconditionally to ensure conda commands are available.
# This addresses the previous error where 'conda activate' failed due to uninitialized shell.
echo "Initializing conda shell..."
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh" || { echo "ERROR: Failed to source conda.sh. Ensure Conda is installed and accessible."; exit 1; }

# --- 2. Create and Activate Conda Environment ---
echo "Checking for existing conda environment at $CONDA_ENV_PATH..."
if [ -d "$CONDA_ENV_PATH" ]; then
    echo "Existing environment found. Removing it for a clean installation."
    conda env remove -p "$CONDA_ENV_PATH" -y || { echo "ERROR: Failed to remove existing conda environment at $CONDA_ENV_PATH."; exit 1; }
fi

echo "Creating new conda environment at $CONDA_ENV_PATH with Python $PYTHON_VERSION..."
conda create -p "$CONDA_ENV_PATH" python="$PYTHON_VERSION" -y || { echo "ERROR: Failed to create conda environment at $CONDA_ENV_PATH."; exit 1; }

echo "Activating conda environment: $CONDA_ENV_PATH"
conda activate "$CONDA_ENV_PATH" || { echo "ERROR: Failed to activate conda environment $CONDA_ENV_PATH."; exit 1; }

# --- 3. Install uv ---
echo "Installing 'uv' into the activated conda environment..."
pip install uv || { echo "ERROR: Failed to install 'uv'."; exit 1; }

# --- 4. Install required packages from requirements.txt files using uv ---
echo "Installing packages from requirements files using 'uv pip install'..."
# The order of requirements files is as specified in the prompt.
uv pip install -r "$ML_REQUIREMENTS_FILE" --prerelease=allow -r "$COMMON_REQUIREMENTS_FILE" || { echo "ERROR: Failed to install packages from requirements files."; exit 1; }

# --- 5. Install additional packages (numpy, pandas, scikit-learn, autogluon.tabular) ---
echo "Installing numpy, pandas, scikit-learn, and autogluon.tabular..."
uv pip install numpy pandas scikit-learn autogluon.tabular || { echo "ERROR: Failed to install core ML libraries (numpy, pandas, scikit-learn, autogluon.tabular)."; exit 1; }

echo "Conda environment setup and package installation complete successfully."