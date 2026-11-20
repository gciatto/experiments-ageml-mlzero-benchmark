#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the target Conda environment path
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/auto_mpg/rep-2/36b7694a16463792f153/attempt-1/mlzero/node_1/conda_env"

# Define paths to the requirements files
REQUIREMENTS_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_MULTIMODAL="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# --- Conda Initialization ---
# This section addresses the previous 'invalid choice: 'shell'' error by explicitly sourcing conda.sh.
# It attempts to find conda.sh using `conda info --base` first, then a common default path.
CONDA_BASE_PATH=$(conda info --base 2>/dev/null || echo "/opt/conda") # Fallback to /opt/conda if info fails
if [ -f "${CONDA_BASE_PATH}/etc/profile.d/conda.sh" ]; then
    echo "Sourcing conda.sh from ${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
    source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    echo "Sourcing conda.sh from /opt/conda/etc/profile.d/conda.sh"
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "Error: conda.sh not found. Please ensure Conda is installed and accessible in the PATH."
    exit 1
fi

# --- Create and Activate Conda Environment ---
echo "Creating or updating conda environment at ${CONDA_ENV_PATH} with Python 3.11..."

# Remove existing environment if it exists to ensure a clean slate for installation
if [ -d "${CONDA_ENV_PATH}" ]; then
    echo "Removing existing environment at ${CONDA_ENV_PATH}..."
    conda env remove -p "${CONDA_ENV_PATH}" -y
fi

# Create the new conda environment with Python 3.11
conda create -p "${CONDA_ENV_PATH}" python=3.11 -y

# Activate the newly created environment
echo "Activating conda environment: ${CONDA_ENV_PATH}"
conda activate "${CONDA_ENV_PATH}"

# --- Install uv ---
echo "Installing uv into the environment..."
pip install uv

# --- Install all required packages using uv ---
echo "Installing numpy, pandas, scikit-learn, and packages from requirements files using uv..."
# Install core packages and then the requirements from files.
# The order of requirements files and the --prerelease=allow flag are preserved as specified.
uv pip install \
    numpy pandas scikit-learn \
    -r "${REQUIREMENTS_MULTIMODAL}" \
    --prerelease=allow \
    -r "${REQUIREMENTS_COMMON}"

echo "Conda environment setup complete at ${CONDA_ENV_PATH}"
echo "Installed packages in ${CONDA_ENV_PATH}:"
conda list