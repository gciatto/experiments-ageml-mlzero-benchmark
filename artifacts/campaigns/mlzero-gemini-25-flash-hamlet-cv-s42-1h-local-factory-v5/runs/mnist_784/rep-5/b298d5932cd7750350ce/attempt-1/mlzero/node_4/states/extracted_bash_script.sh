#!/bin/bash
set -euo pipefail

# Define environment and file paths
CONDA_ENV_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/mnist_784/rep-5/b298d5932cd7750350ce/attempt-1/mlzero/node_4/conda_env"
PYTHON_VERSION="3.11"
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AUTOGLUON_TABULAR_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "Starting Conda environment setup for AutoGluon Tabular module..."

# 1. Create the Conda environment
echo "Creating conda environment at $CONDA_ENV_DIR with Python $PYTHON_VERSION..."
conda create --prefix "$CONDA_ENV_DIR" python="$PYTHON_VERSION" --yes || { echo "ERROR: Failed to create conda environment at $CONDA_ENV_DIR." >&2; exit 1; }
echo "Conda environment created successfully."

# 2. Install uv into the newly created environment using 'conda run'
echo "Installing uv into the environment..."
conda run -p "$CONDA_ENV_DIR" pip install uv || { echo "ERROR: Failed to install uv into $CONDA_ENV_DIR." >&2; exit 1; }
echo "uv installed successfully."

# 3. Install required packages from requirements files using uv within the environment
echo "Installing packages from requirements files using uv..."
conda run -p "$CONDA_ENV_DIR" uv pip install \
    -r "$AUTOGLUON_TABULAR_REQUIREMENTS_FILE" \
    --prerelease=allow \
    -r "$COMMON_REQUIREMENTS_FILE" \
    || { echo "ERROR: Failed to install packages from requirements files using uv." >&2; exit 1; }
echo "Packages from requirements files installed successfully."

# 4. Ensure numpy, pandas, scikit-learn are installed using uv within the environment
echo "Ensuring numpy, pandas, scikit-learn are installed..."
conda run -p "$CONDA_ENV_DIR" uv pip install numpy pandas scikit-learn || { echo "ERROR: Failed to install numpy, pandas, scikit-learn using uv." >&2; exit 1; }
echo "numpy, pandas, scikit-learn ensured."

echo "Conda environment setup complete for AutoGluon Tabular module."