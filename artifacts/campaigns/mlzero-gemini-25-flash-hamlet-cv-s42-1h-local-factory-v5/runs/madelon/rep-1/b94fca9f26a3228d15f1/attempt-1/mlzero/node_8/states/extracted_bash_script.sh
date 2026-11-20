#!/bin/bash
set -euo pipefail # Exit immediately if a command exits with a non-zero status.
                  # Exit if an unset variable is used.
                  # The return value of a pipeline is the value of the last command to exit with a non-zero status.

# Define paths and versions
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/madelon/rep-1/b94fca9f26a3228d15f1/attempt-1/mlzero/node_8/conda_env"
PYTHON_VERSION="3.11"
COMMON_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

echo "Starting minimal BASH INSTALLATION script for AutoGluon factory module..."
echo "Target Conda environment path: $CONDA_ENV_PATH"

# --- Conda Environment Setup ---

# 1. Ensure conda is initialized for the current shell session
# This is crucial for `conda activate` to work in non-interactive scripts.
if command -v conda &>/dev/null; then
    echo "Initializing conda shell hook..."
    eval "$(conda shell hook)"
else
    echo "Error: 'conda' command not found. Please ensure Anaconda/Miniconda is installed and in PATH."
    exit 1
fi

# 2. Remove existing environment if it exists to ensure a clean install.
if [ -d "$CONDA_ENV_PATH" ]; then
    echo "Removing existing conda environment at $CONDA_ENV_PATH..."
    # Use -y for non-interactive removal.
    conda env remove -p "$CONDA_ENV_PATH" -y || { echo "Warning: Failed to remove existing environment. Attempting to proceed with creation."; }
fi

# 3. Create the new environment with the specified Python version and pip.
echo "Creating conda environment at $CONDA_ENV_PATH with Python $PYTHON_VERSION..."
conda create -p "$CONDA_ENV_PATH" python="$PYTHON_VERSION" pip -y
if [ $? -ne 0 ]; then
    echo "Error: Conda environment creation failed."
    exit 1
fi

# 4. Activate the environment
echo "Activating conda environment: $CONDA_ENV_PATH"
conda activate "$CONDA_ENV_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment."
    exit 1
fi
echo "Conda environment activated. Python version: $(python --version)"

# --- Package Installation ---

# 5. Install uv
echo "Installing uv into the activated environment..."
pip install uv
if [ $? -ne 0 ]; then
    echo "Error: Failed to install uv."
    exit 1
fi

# 6. Install required packages from requirements files using uv
echo "Installing packages from requirements files using uv pip install..."
if [ ! -f "$ML_REQ_PATH" ]; then
    echo "Error: ML requirements file not found at $ML_REQ_PATH"
    exit 1
fi
if [ ! -f "$COMMON_REQ_PATH" ]; then
    echo "Error: Common requirements file not found at $COMMON_REQ_PATH"
    exit 1
fi

uv pip install -r "$ML_REQ_PATH" --prerelease=allow -r "$COMMON_REQ_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to install packages from requirements files."
    exit 1
fi

# 7. Install additional packages: numpy, pandas, scikit-learn, and autogluon.tabular
echo "Installing numpy, pandas, scikit-learn, and autogluon.tabular..."
uv pip install numpy pandas scikit-learn autogluon.tabular
if [ $? -ne 0 ]; then
    echo "Error: Failed to install core ML libraries (numpy, pandas, scikit-learn, autogluon.tabular)."
    exit 1
fi

echo "All required packages installed successfully."