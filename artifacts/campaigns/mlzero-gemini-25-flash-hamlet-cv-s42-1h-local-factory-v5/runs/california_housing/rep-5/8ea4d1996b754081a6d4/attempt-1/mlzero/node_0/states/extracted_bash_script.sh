#!/bin/bash
set -euo pipefail

# Define the target conda environment directory
ENV_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/california_housing/rep-5/8ea4d1996b754081a6d4/attempt-1/mlzero/node_0/conda_env"

# Define paths to the requirements files
REQ_FILE_TABULAR="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"
REQ_FILE_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"

echo "--- Starting Conda Environment Setup ---"

# 1. Create the conda environment with Python 3.11
echo "Creating conda environment at '$ENV_DIR' with Python 3.11..."
conda create -p "$ENV_DIR" python=3.11 -y

# 2. Activate the newly created environment
echo "Activating conda environment..."
# Ensure conda is initialized in the current shell for `conda activate` to work
# This is a robust way to activate in non-interactive scripts.
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv using the environment's pip
echo "Installing uv into the environment..."
pip install uv

# 4. Install required packages from the specified requirements files using uv
# This command will handle AutoGluon, numpy, pandas, scikit-learn, and their dependencies.
echo "Installing packages from requirements files using uv pip install..."
uv pip install -r "$REQ_FILE_TABULAR" --prerelease=allow -r "$REQ_FILE_COMMON"

echo "--- Conda Environment Setup Complete ---"

# Deactivate the environment (optional, but good practice to return to base)
# conda deactivate