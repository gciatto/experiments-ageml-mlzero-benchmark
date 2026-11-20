#!/bin/bash
set -euo pipefail

# Define the full path for the conda environment
CONDA_ENV_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/adult/rep-4/4c241e27bfec28ef4c09/attempt-1/mlzero/node_4/conda_env"

# Define paths to the requirements files
ML_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"

echo "--- Starting Conda environment setup for node_4 ---"

# 1. Remove existing environment if it exists to ensure a clean install
if [ -d "${CONDA_ENV_DIR}" ]; then
    echo "Removing existing environment at ${CONDA_ENV_DIR}..."
    rm -rf "${CONDA_ENV_DIR}"
fi

# 2. Create the conda environment with Python 3.11
echo "Creating conda environment at ${CONDA_ENV_DIR} with Python 3.11..."
conda create -p "${CONDA_ENV_DIR}" python=3.11 -y

# 3. Activate the environment
echo "Activating conda environment..."
# Ensure conda is initialized in the current shell session for 'conda activate'
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_DIR}"

# 4. Install uv
echo "Installing uv into the environment..."
pip install uv

# 5. Install required packages from specified requirements files using uv
echo "Installing packages from requirements files using uv..."
uv pip install -r "${ML_REQUIREMENTS_FILE}" --prerelease=allow -r "${COMMON_REQUIREMENTS_FILE}"

# 6. Install additional packages: numpy, pandas, scikit-learn, and autogluon.tabular
# These are explicitly mentioned as needed for the node's conda_env and the factory module.
echo "Installing numpy, pandas, scikit-learn, and autogluon.tabular..."
uv pip install numpy pandas scikit-learn autogluon.tabular

echo "--- Conda environment setup complete at ${CONDA_ENV_DIR} ---"