#!/bin/bash
set -euo pipefail

# Define the exact path for the Conda environment
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/california_housing/rep-3/1bf45bafd12f0834c802/attempt-1/mlzero/node_0/conda_env"

# Define paths to the requirements files
REQUIREMENTS_COMMON_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_TABULAR_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "--- Starting Conda environment setup for AutoGluon Tabular ---"

# 1. Create the Conda environment with Python 3.11
echo "Creating conda environment at '${CONDA_ENV_PATH}' with Python 3.11..."
conda create -p "${CONDA_ENV_PATH}" python=3.11 -y

# 2. Activate the newly created environment
echo "Activating conda environment..."
# Ensure conda's shell functions are available
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PATH}"

# 3. Install uv using pip within the activated environment
echo "Installing uv..."
pip install uv

# 4. Install required packages using uv pip install
# This command installs packages from both requirements files, allowing prereleases.
# uv's resolver will ensure only necessary packages and their dependencies are installed,
# and will not unnecessarily upgrade or reinstall if versions already satisfy requirements.
echo "Installing required packages from '${REQUIREMENTS_TABULAR_PATH}' and '${REQUIREMENTS_COMMON_PATH}' using uv..."
uv pip install \
    -r "${REQUIREMENTS_TABULAR_PATH}" \
    --prerelease=allow \
    -r "${REQUIREMENTS_COMMON_PATH}"

echo "--- Conda environment setup complete ---"