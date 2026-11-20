#!/bin/bash
set -euo pipefail

# Define the target conda environment directory
ENV_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/adult/rep-3/614ff58cfd5d961d5dfb/attempt-1/mlzero/node_2/conda_env"

# Define paths to the requirements files
REQ_FILE_ML="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"
REQ_FILE_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"

echo "--- Starting Conda Environment Setup ---"

# Ensure conda is initialized for the current shell session
# This is crucial for 'conda activate' to work within a script.
# It assumes 'conda' command is available in PATH.
if ! command -v conda &> /dev/null; then
    echo "Error: 'conda' command not found. Please ensure Conda is installed and initialized."
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# Remove existing environment if it exists to ensure a clean install
if [ -d "$ENV_DIR" ]; then
    echo "Removing existing conda environment at $ENV_DIR..."
    conda env remove -p "$ENV_DIR" -y || true # '|| true' prevents script from exiting if removal fails for non-critical reasons
fi

echo "Creating conda environment at $ENV_DIR with Python 3.11..."
conda create -p "$ENV_DIR" python=3.11 -y

echo "Activating conda environment..."
conda activate "$ENV_DIR"

echo "Installing uv..."
pip install uv

echo "Installing required packages from requirements.txt files using uv..."
uv pip install -r "$REQ_FILE_ML" --prerelease=allow -r "$REQ_FILE_COMMON"

echo "Installing additional packages: numpy, pandas, scikit-learn, autogluon.tabular using uv..."
uv pip install numpy pandas scikit-learn autogluon.tabular

echo "--- Conda Environment Setup Complete ---"
echo "Environment is ready at $ENV_DIR"