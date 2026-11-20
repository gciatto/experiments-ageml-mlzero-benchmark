#!/bin/bash
set -e

# Define the full path for the conda environment
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/har/rep-3/053ecb30e95bae230b78/attempt-1/mlzero/node_0/conda_env"

# Define paths to the requirements files
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AUTOGLUON_TABULAR_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "--- Starting Conda environment setup ---"

# 1. Create a new conda environment with Python 3.11
echo "Creating conda environment at '$CONDA_ENV_PATH' with Python 3.11..."
conda create --prefix "$CONDA_ENV_PATH" python=3.11 --yes

# 2. Activate the newly created environment
echo "Activating conda environment..."
# Ensure conda is initialized for the current shell session
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

# 3. Install uv (a fast Python package installer)
echo "Installing uv into the environment..."
pip install uv

# 4. Install required packages using uv
# This command installs packages from both requirements files, allowing prereleases.
# uv will ensure only specified packages and their dependencies are installed,
# and will not upgrade/reinstall if already at the correct version.
echo "Installing packages from requirements files using uv pip install..."
uv pip install \
    -r "$AUTOGLUON_TABULAR_REQUIREMENTS_FILE" \
    --prerelease=allow \
    -r "$COMMON_REQUIREMENTS_FILE"

echo "--- Conda environment setup complete ---"