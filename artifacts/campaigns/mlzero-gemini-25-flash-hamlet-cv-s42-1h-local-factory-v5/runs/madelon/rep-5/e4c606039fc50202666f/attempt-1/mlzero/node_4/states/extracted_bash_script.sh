#!/bin/bash
set -e

# Define the exact path for the conda environment
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/madelon/rep-5/e4c606039fc50202666f/attempt-1/mlzero/node_4/conda_env"

# Define paths to the requirement files
COMMON_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

echo "--- Starting minimal BASH INSTALLATION script ---"

# Remove existing environment directory if it exists to ensure a clean slate
if [ -d "$CONDA_ENV_PATH" ]; then
    echo "Removing existing environment directory: $CONDA_ENV_PATH"
    rm -rf "$CONDA_ENV_PATH"
fi

echo "Creating conda environment at $CONDA_ENV_PATH with Python 3.11..."
conda create -p "$CONDA_ENV_PATH" python=3.11 -y

echo "Activating the conda environment..."
# Robustly initialize conda for the current shell session if not already done
# This addresses previous CondaError related to 'conda activate' not working.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

echo "Installing uv into the environment..."
pip install uv

echo "Installing core packages (numpy, pandas, scikit-learn, autogluon.tabular) and requirements from files using uv..."
# Install explicitly requested packages and then packages from the specified requirement files.
# The order of requirement files and the --prerelease=allow flag are as specified.
# numpy, pandas, scikit-learn are explicitly requested.
# autogluon.tabular is the selected ML library based on the factory module imports.
uv pip install \
    numpy pandas scikit-learn autogluon.tabular \
    -r "$ML_REQ_FILE" \
    --prerelease=allow \
    -r "$COMMON_REQ_FILE"

echo "--- Installation complete for conda_env ---"