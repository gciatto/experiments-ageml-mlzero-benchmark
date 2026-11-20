#!/bin/bash
set -euo pipefail

# Define the full path for the conda environment
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/wilt/rep-3/f9fe36638309f2eeddc8/attempt-1/mlzero/node_1/conda_env"
REQUIREMENTS_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_ML="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

# Check if conda is available in the PATH
if ! command -v conda &> /dev/null
then
    echo "Error: 'conda' command not found. Please ensure Conda is installed and in your system's PATH."
    exit 1
fi

echo "Creating conda environment at $CONDA_ENV_PATH with Python 3.11..."
# Create the conda environment with the specified Python version
conda create -y --prefix "$CONDA_ENV_PATH" python=3.11

# Define paths to the new environment's executables
PYTHON_BIN="$CONDA_ENV_PATH/bin/python"
PIP_BIN="$CONDA_ENV_PATH/bin/pip"
UV_BIN="$CONDA_ENV_PATH/bin/uv"

# Verify that the Python executable was created successfully
if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: Python executable not found at $PYTHON_ENV_PATH/bin/python after environment creation."
    exit 1
fi

echo "Installing uv into the new environment..."
# Install uv using the pip from the newly created environment
"$PIP_BIN" install uv

# Verify that uv executable is available
if [ ! -f "$UV_BIN" ]; then
    echo "Error: 'uv' executable not found at $UV_BIN after installation."
    exit 1
fi

echo "Installing required packages using uv..."
# Install packages from the specified requirements files and additional libraries
# `autogluon.tabular` will pull `numpy` and `pandas` as dependencies.
"$UV_BIN" pip install \
    --prerelease=allow \
    -r "$REQUIREMENTS_ML" \
    -r "$REQUIREMENTS_COMMON" \
    scikit-learn \
    autogluon.tabular

echo "Conda environment setup complete at $CONDA_ENV_PATH."
echo "To activate this environment for interactive use, run: conda activate $CONDA_ENV_PATH"