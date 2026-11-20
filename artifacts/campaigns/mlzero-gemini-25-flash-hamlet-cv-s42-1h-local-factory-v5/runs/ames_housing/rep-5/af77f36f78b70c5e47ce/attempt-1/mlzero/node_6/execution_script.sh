#!/bin/bash
set -euo pipefail

# Define the full path for the conda environment
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-5/af77f36f78b70c5e47ce/attempt-1/mlzero/node_6/conda_env"

# Define paths to the requirements files
REQUIREMENTS_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_TABULAR="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "--- Starting Conda Environment Setup ---"
echo "Target conda environment path: ${CONDA_ENV_PATH}"
echo "Python version: 3.11"

# 1. Remove existing environment if it exists to ensure a clean setup.
# This step is included for robustness, ensuring a clean slate for installation.
# Using '|| true' to prevent script failure if the environment does not exist.
if [ -d "$CONDA_ENV_PATH" ]; then
    echo "Removing existing conda environment at ${CONDA_ENV_PATH}..."
    conda env remove --prefix "$CONDA_ENV_PATH" --yes || true
fi

# 2. Create the conda environment with Python 3.11.
echo "Creating new conda environment with Python 3.11..."
conda create --prefix "$CONDA_ENV_PATH" python=3.11 --yes

echo "Conda environment created successfully."

# 3. Install uv into the newly created environment.
# As per previous error summary's suggestion, use 'conda run --prefix' for robustness.
# This ensures commands are executed within the correct environment without relying on
# explicit 'conda activate' and its shell function limitations.
echo "Installing uv into the environment using 'conda run --prefix'..."
conda run --prefix "$CONDA_ENV_PATH" pip install uv

echo "uv installed successfully."

# 4. Install core packages (numpy, pandas, scikit-learn) and required packages from files using uv.
echo "Installing core packages (numpy, pandas, scikit-learn) and requirements from files using uv..."
echo "Requirements files: ${REQUIREMENTS_TABULAR}, ${REQUIREMENTS_COMMON}"

# Install numpy, pandas, scikit-learn explicitly, then the requirements files.
# uv will resolve dependencies and ensure correct versions, respecting the requirements files.
# The order of -r flags and --prerelease=allow is maintained as specified in the prompt.
conda run --prefix "$CONDA_ENV_PATH" uv pip install \
    numpy pandas scikit-learn \
    -r "$REQUIREMENTS_TABULAR" \
    --prerelease=allow \
    -r "$REQUIREMENTS_COMMON"

echo "Required packages installed successfully."
echo "--- Conda Environment Setup Complete ---"