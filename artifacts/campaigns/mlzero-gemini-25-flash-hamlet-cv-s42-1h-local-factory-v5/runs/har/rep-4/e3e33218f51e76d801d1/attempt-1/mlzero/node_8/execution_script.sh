#!/bin/bash
set -euo pipefail

# Define the base directory for the environment
# The prompt specifies 'node_8' for the conda_env folder.
BASE_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/har/rep-4/e3e33218f51e76d801d1/attempt-1/mlzero/node_8"
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

# Remove existing environment directory to ensure a clean and consistent installation.
# This addresses potential issues from previous failed attempts where the environment
# might have been incompletely created or corrupted.
if [ -d "${CONDA_ENV_PATH}" ]; then
    echo "Removing existing environment directory at ${CONDA_ENV_PATH}..."
    rm -rf "${CONDA_ENV_PATH}" || { echo "Error: Failed to remove directory ${CONDA_ENV_PATH}."; exit 1; }
fi

# Create the conda environment with Python 3.11
echo "Creating conda environment with Python 3.11 at ${CONDA_ENV_PATH}..."
conda create --prefix "${CONDA_ENV_PATH}" python=3.11 --yes || { echo "Error: Conda environment creation failed."; exit 1; }

# Install uv into the newly created environment using 'conda run'.
# 'conda run -p' ensures commands are executed within the specified environment
# without needing to explicitly activate it, addressing previous activation issues.
echo "Installing uv into the environment using 'conda run'..."
conda run -p "${CONDA_ENV_PATH}" python -m pip install uv || { echo "Error: Failed to install uv."; exit 1; }

# Install required packages using uv from within the environment via 'conda run'.
# This includes packages from the specified requirements files and explicitly requested libraries
# (numpy, pandas, scikit-learn). 'uv' handles exact package versions and dependencies.
echo "Installing required packages using uv via 'conda run'..."
conda run -p "${CONDA_ENV_PATH}" uv pip install \
    -r "${AUTOGLUON_TABULAR_REQ_FILE}" \
    --prerelease=allow \
    -r "${COMMON_REQ_FILE}" \
    numpy pandas scikit-learn || { echo "Error: Failed to install required packages with uv."; exit 1; }

echo "--- Conda Environment Setup Complete ---"

# The environment is now prepared. The harness will activate it as needed for execution.