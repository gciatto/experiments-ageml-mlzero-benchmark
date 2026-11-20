#!/bin/bash
set -euo pipefail

# Define the target Conda environment path
ENV_BASE_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/texture/rep-4/79235980f285c36790bd/attempt-1/mlzero/node_1"
CONDA_ENV_PATH="${ENV_BASE_PATH}/conda_env"

# Define paths to the requirements files
REQ_COMMON_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_MULTIMODAL_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

echo "Starting Conda environment setup for AutoGluon MultiModal at ${CONDA_ENV_PATH}..."

# 1. Remove existing environment if it exists to ensure a clean state
if [ -d "${CONDA_ENV_PATH}" ]; then
    echo "Existing Conda environment found at ${CONDA_ENV_PATH}. Removing it."
    rm -rf "${CONDA_ENV_PATH}"
fi

# Ensure the parent directory exists for the conda environment
mkdir -p "${ENV_BASE_PATH}"

# 2. Create a new Conda environment with Python 3.11
echo "Creating Conda environment with Python 3.11..."
conda create -p "${CONDA_ENV_PATH}" python=3.11 -y

# 3. Activate the newly created environment
# Source conda.sh to make 'conda activate' available in non-interactive scripts
# This assumes conda is installed and its initialization script is available.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PATH}"

echo "Conda environment activated: $(which python)"

# 4. Install uv using the environment's pip
echo "Installing uv into the environment..."
python -m pip install uv

# 5. Install required packages using uv
# The --prerelease=allow flag is included as specified.
# uv will handle dependencies and avoid unnecessary upgrades/reinstalls if versions are satisfied.
echo "Installing core packages and AutoGluon MultiModal dependencies using uv..."
uv pip install -r "${REQ_MULTIMODAL_PATH}" --prerelease=allow -r "${REQ_COMMON_PATH}"

# 6. Ensure numpy, pandas, and scikit-learn are explicitly installed
# uv will only install these if they are not already satisfied by the requirements files.
echo "Ensuring numpy, pandas, and scikit-learn are installed..."
uv pip install numpy pandas scikit-learn

echo "Conda environment setup complete at ${CONDA_ENV_PATH}"
echo "Installation successful."