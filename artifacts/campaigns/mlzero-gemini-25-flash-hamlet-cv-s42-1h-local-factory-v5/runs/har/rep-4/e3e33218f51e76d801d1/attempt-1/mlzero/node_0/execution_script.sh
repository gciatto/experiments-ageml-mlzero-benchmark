#!/bin/bash
set -euo pipefail

# Define the base directory for the environment
BASE_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/har/rep-4/e3e33218f51e76d801d1/attempt-1/mlzero/node_0"
CONDA_ENV_PATH="${BASE_DIR}/conda_env"

# Define paths to requirement files
COMMON_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AUTOGLUON_TABULAR_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "--- Starting Conda Environment Setup ---"
echo "Conda environment path: ${CONDA_ENV_PATH}"

# Ensure conda is available in the PATH
if ! command -v conda &> /dev/null; then
    echo "Error: 'conda' command not found. Please ensure Anaconda/Miniconda is installed and in your PATH."
    exit 1
fi

# Remove existing environment to ensure a clean and consistent installation
if conda env list | grep -q "${CONDA_ENV_PATH}" ; then
    echo "Removing existing conda environment at ${CONDA_ENV_PATH}..."
    conda env remove --prefix "${CONDA_ENV_PATH}" --yes
fi

# Create the conda environment with Python 3.11
echo "Creating conda environment with Python 3.11 at ${CONDA_ENV_PATH}..."
conda create --prefix "${CONDA_ENV_PATH}" python=3.11 --yes

# Activate the conda environment
# This is crucial for subsequent pip commands to install into the correct environment.
echo "Activating the conda environment..."
# Source the conda.sh script to properly activate the environment in the current shell context
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PATH}"

# Install uv into the activated environment
echo "Installing uv into the environment..."
pip install uv

# Install required packages using uv
# This includes packages from the specified requirements files and explicitly requested libraries.
# uv ensures exact package versions and minimal changes as per requirements.
echo "Installing required packages using uv..."
uv pip install \
    -r "${AUTOGLUON_TABULAR_REQ_FILE}" \
    --prerelease=allow \
    -r "${COMMON_REQ_FILE}" \
    numpy pandas scikit-learn

echo "--- Conda Environment Setup Complete ---"

# The environment remains active for subsequent operations by the harness.