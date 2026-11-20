#!/bin/bash
set -euo pipefail

# Define the target conda environment directory for node_7
ENV_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/california_housing/rep-5/8ea4d1996b754081a6d4/attempt-1/mlzero/node_7/conda_env"

# Define paths to the requirements files
REQ_FILE_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_FILE_TABULAR="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "--- Starting Conda Environment Setup for node_7 ---"

# 1. Create the conda environment with Python 3.11
echo "Creating conda environment at '$ENV_DIR' with Python 3.11..."
conda create --prefix "$ENV_DIR" python=3.11 -y || { echo "ERROR: Failed to create conda environment at '$ENV_DIR'."; exit 1; }

# 2. Activate the newly created environment
echo "Activating conda environment..."
# Ensure conda is initialized in the current shell for `conda activate` to work.
# This is a robust way to activate in non-interactive scripts.
# Fix for "unbound variable" error with CONDA_SHLVL by using parameter expansion default value.
if [ -z "${CONDA_SHLVL:-}" ]; then
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "/opt/conda") # Fallback if conda info fails
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "ERROR: conda.sh not found. Ensure Conda is installed and initialized correctly."
        exit 1
    fi
fi
conda activate "$ENV_DIR" || { echo "ERROR: Failed to activate conda environment at '$ENV_DIR'."; exit 1; }

# 3. Install uv using the environment's pip
echo "Installing uv into the environment..."
pip install uv || { echo "ERROR: Failed to install uv into the environment."; exit 1; }

# 4. Install required packages from the specified requirements files using uv.
# This command will install packages from both requirements files, including
# autogluon.tabular, numpy, pandas, scikit-learn, and their dependencies.
# uv respects specified versions and avoids unnecessary upgrades/reinstalls.
echo "Installing packages from requirements files using uv pip install..."
# The order of requirements files is specified in the prompt.
uv pip install -r "$REQ_FILE_TABULAR" --prerelease=allow -r "$REQ_FILE_COMMON" || { echo "ERROR: Failed to install required packages using uv."; exit 1; }

echo "--- Conda Environment Setup Complete for node_7 ---"