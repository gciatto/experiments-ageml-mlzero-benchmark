#!/bin/bash
set -euxo pipefail

# Define the base path for the experiment
BASE_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/madelon/rep-4/3f1d9fe8ce5e73e90c19/attempt-1/mlzero/node_1"

# Define the full path for the conda environment
ENV_PATH="${BASE_PATH}/conda_env"

# Define paths to the requirements files
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

# 1. Create the conda environment with Python 3.11
echo "Creating conda environment at ${ENV_PATH} with Python 3.11..."
conda create -p "${ENV_PATH}" python=3.11 -y

# 2. Activate the environment
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

# 3. Install uv
echo "Installing uv into the environment..."
pip install uv

# 4. Install required packages from the specified requirements files using uv
echo "Installing packages from requirements files using uv..."
uv pip install -r "${ML_REQUIREMENTS_FILE}" --prerelease=allow -r "${COMMON_REQUIREMENTS_FILE}"

# 5. Install additional packages: numpy, pandas, scikit-learn, and autogluon.tabular
echo "Installing numpy, pandas, scikit-learn, and autogluon.tabular..."
uv pip install numpy pandas scikit-learn autogluon.tabular

echo "Conda environment setup complete."