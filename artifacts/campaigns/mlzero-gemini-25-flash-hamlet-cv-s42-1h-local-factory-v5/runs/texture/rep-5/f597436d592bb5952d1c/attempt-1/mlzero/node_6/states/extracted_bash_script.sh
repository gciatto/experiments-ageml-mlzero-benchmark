#!/bin/bash
set -euo pipefail

# --- Configuration ---
# The exact candidate environment path provided in the prompt
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/texture/rep-5/f597436d592bb5952d1c/attempt-1/mlzero/node_6/conda_env"
PYTHON_VERSION="3.11"
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

echo "Starting minimal BASH INSTALLATION script for factory module."

# --- 1. Conda Initialization ---
# Source the conda.sh script unconditionally to ensure conda commands are available.
# This addresses previous errors where 'conda activate' failed due to an uninitialized shell.
echo "Initializing conda shell..."
# Find conda base path robustly.
CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: Conda base path not found. Ensure Conda is installed and in PATH."
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh" || { echo "ERROR: Failed to source conda.sh. Ensure Conda is installed and accessible."; exit 1; }

# --- 2. Create and Activate Conda Environment ---
echo "Checking for existing conda environment at $CONDA_ENV_PATH..."
# Use 'conda env list' to check for path-based environments
if conda env list | grep -q "$CONDA_ENV_PATH"; then
    echo "Existing environment found. Removing it for a clean installation."
    conda env remove -p "$CONDA_ENV_PATH" -y || { echo "ERROR: Failed to remove existing conda environment at $CONDA_ENV_PATH."; exit 1; }
fi

echo "Creating new conda environment at $CONDA_ENV_PATH with Python $PYTHON_VERSION..."
conda create -p "$CONDA_ENV_PATH" python="$PYTHON_VERSION" -y || { echo "ERROR: Failed to create conda environment at $CONDA_ENV_PATH."; exit 1; }

echo "Activating conda environment: $CONDA_ENV_PATH"
conda activate "$CONDA_ENV_PATH" || { echo "ERROR: Failed to activate conda environment $CONDA_ENV_PATH."; exit 1; }

# --- 3. Install core ML packages with conda ---
# This addresses previous errors about missing numpy, pandas, scikit-learn
# and ensures robust installation of these core libraries via conda's dependency solver.
echo "Installing numpy, pandas, and scikit-learn using conda..."
conda install -p "$CONDA_ENV_PATH" numpy pandas scikit-learn -y || { echo "ERROR: Failed to install core ML libraries (numpy, pandas, scikit-learn) via conda."; exit 1; }

# --- 4. Install uv ---
echo "Installing 'uv' into the activated conda environment..."
# Use the pip from the activated environment to ensure it's installed correctly.
pip install uv || { echo "ERROR: Failed to install 'uv'."; exit 1; }

# --- 5. Install AutoGluon and other required packages using uv ---
# Explicitly install autogluon.tabular as it's a core dependency for the factory module
# and was previously missed (e.g., if ML_REQUIREMENTS_FILE was empty).
# Then install from requirements files as specified, using uv.
echo "Installing autogluon.tabular and packages from requirements files using 'uv pip install'..."
# The prompt specifies the order for requirements files: ML_REQUIREMENTS_FILE then COMMON_REQUIREMENTS_FILE.
uv pip install autogluon.tabular -r "$ML_REQUIREMENTS_FILE" --prerelease=allow -r "$COMMON_REQUIREMENTS_FILE" || { echo "ERROR: Failed to install autogluon.tabular or packages from requirements files."; exit 1; }

echo "Conda environment setup and package installation complete successfully."