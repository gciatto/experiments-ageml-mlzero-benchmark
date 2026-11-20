#!/bin/bash
set -euo pipefail

# Define the full path for the conda environment
CONDA_ENV_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/adult/rep-4/4c241e27bfec28ef4c09/attempt-1/mlzero/node_1/conda_env"

# Define paths to the requirements files
ML_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"

echo "--- Starting Conda environment setup ---"

# 1. Create the conda environment with Python 3.11
echo "Creating conda environment at ${CONDA_ENV_DIR} with Python 3.11..."
conda create -p "${CONDA_ENV_DIR}" python=3.11 -y

# 2. Activate the environment
echo "Activating conda environment..."
# Ensure conda is initialized in the current shell session
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_DIR}"

# 3. Install uv
echo "Installing uv into the environment..."
pip install uv

# 4. Install required packages from specified requirements files using uv
echo "Installing packages from requirements files using uv..."
uv pip install -r "${ML_REQUIREMENTS_FILE}" --prerelease=allow -r "${COMMON_REQUIREMENTS_FILE}"

# 5. Install additional packages: numpy, pandas, scikit-learn, and autogluon.tabular
echo "Installing numpy, pandas, scikit-learn, and autogluon.tabular..."
uv pip install numpy pandas scikit-learn autogluon.tabular

echo "--- Conda environment setup complete at ${CONDA_ENV_DIR} ---"

# Deactivate the environment to return to the base shell state
conda deactivate