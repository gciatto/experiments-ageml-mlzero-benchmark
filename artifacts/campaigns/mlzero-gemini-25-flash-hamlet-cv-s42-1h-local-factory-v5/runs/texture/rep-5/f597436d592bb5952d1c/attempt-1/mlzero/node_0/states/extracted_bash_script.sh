#!/bin/bash
set -euo pipefail

# Define the base directory for the environment
BASE_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/texture/rep-5/f597436d592bb5952d1c/attempt-1/mlzero/node_0"
CONDA_ENV_PATH="${BASE_DIR}/conda_env"

# Define requirements file paths
REQUIREMENTS_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_TABULAR="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "--- Starting Conda Environment Setup ---"

# Ensure the base directory exists
mkdir -p "${BASE_DIR}"
echo "Target environment path: ${CONDA_ENV_PATH}"

# Initialize Conda for the script session
# This assumes 'conda' command is available in the PATH.
# If conda is not initialized, this will attempt to source the necessary script.
if ! type "conda" > /dev/null 2>&1; then
    echo "Conda command not found. Attempting to initialize Conda."
    # Find conda.sh in common installation paths
    CONDA_SH_PATH=$(find /opt/conda /usr/local/conda /home/*/miniconda* /home/*/anaconda* -name "conda.sh" 2>/dev/null | head -n 1)
    if [ -f "${CONDA_SH_PATH}" ]; then
        source "$(dirname "${CONDA_SH_PATH}")/../etc/profile.d/conda.sh"
        echo "Conda initialized from ${CONDA_SH_PATH}"
    else
        echo "Error: conda.sh not found. Please ensure Conda is installed and accessible in the PATH."
        exit 1
    fi
else
    echo "Conda command found, assuming it's initialized."
    # Deactivate any active environment to ensure a clean state before creating a new one
    conda deactivate 2>/dev/null || true
fi

# Remove existing environment if it exists to ensure a clean slate
if [ -d "${CONDA_ENV_PATH}" ]; then
    echo "Removing existing conda environment at ${CONDA_ENV_PATH}..."
    conda env remove -p "${CONDA_ENV_PATH}" -y
fi

echo "Creating conda environment at ${CONDA_ENV_PATH} with Python 3.11..."
conda create -p "${CONDA_ENV_PATH}" python=3.11 -y

echo "Activating conda environment..."
conda activate "${CONDA_ENV_PATH}"

echo "Installing uv..."
pip install uv

echo "Installing required packages from requirements files using uv..."
# Check if requirements files exist
if [ ! -f "${REQUIREMENTS_COMMON}" ]; then
    echo "Error: Common requirements file not found at ${REQUIREMENTS_COMMON}"
    exit 1
fi
if [ ! -f "${REQUIREMENTS_TABULAR}" ]; then
    echo "Error: Tabular requirements file not found at ${REQUIREMENTS_TABULAR}"
    exit 1
fi

# Install packages. uv will handle dependencies and avoid re-installing if already satisfied.
# The --prerelease=allow flag is included as specified.
uv pip install -r "${REQUIREMENTS_TABULAR}" --prerelease=allow -r "${REQUIREMENTS_COMMON}"

echo "Installing numpy, pandas, and scikit-learn using uv..."
# These packages are explicitly requested. uv will skip them if already satisfied by the requirements files.
uv pip install numpy pandas scikit-learn

echo "--- Conda Environment Setup Complete ---"