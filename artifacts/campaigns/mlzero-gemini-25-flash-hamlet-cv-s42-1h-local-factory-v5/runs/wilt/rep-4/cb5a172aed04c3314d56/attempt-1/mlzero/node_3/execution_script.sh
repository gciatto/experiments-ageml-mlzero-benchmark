#!/bin/bash
set -euo pipefail

# Define environment and file paths
# The conda environment will be created in the 'conda_env' folder under node_3.
ENV_BASE_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/wilt/rep-4/cb5a172aed04c3314d56/attempt-1/mlzero/node_3"
CONDA_ENV_PATH="${ENV_BASE_DIR}/conda_env"
COMMON_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AUTOGLUON_TABULAR_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "Starting conda environment setup for AutoGluonTabularModel..."
echo "Target environment path: ${CONDA_ENV_PATH}"

# Ensure conda is available and initialized for the current shell session.
if ! command -v conda &> /dev/null; then
    echo "Error: 'conda' command not found. Please ensure Conda is installed and in your PATH."
    exit 1
fi

# Initialize conda for the current shell session to make 'conda activate' available.
# This addresses previous errors related to conda initialization.
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Error: conda initialization script not found at expected location: $(conda info --base)/etc/profile.d/conda.sh"
    echo "Cannot activate environment. Please check your conda installation."
    exit 1
fi

# Remove existing environment if it exists to ensure a clean slate for installation.
if conda env list | grep -q "${CONDA_ENV_PATH}"; then
    echo "Removing existing conda environment at ${CONDA_ENV_PATH}..."
    conda env remove -p "${CONDA_ENV_PATH}" -y
fi

echo "Creating conda environment with Python 3.11 at ${CONDA_ENV_PATH}..."
conda create -p "${CONDA_ENV_PATH}" python=3.11 -y

echo "Activating conda environment: ${CONDA_ENV_PATH}"
conda activate "${CONDA_ENV_PATH}"

echo "Installing uv..."
# Use python -m pip to ensure pip from the activated environment is used.
python -m pip install uv

echo "Installing required packages from requirement files using uv..."
# Install packages from the specified requirement files.
# The order is important as specified: autogluon.tabular requirements first, then common.
# uv will install exact packages and their dependencies, including numpy, pandas, scikit-learn,
# and autogluon.tabular, without unnecessary upgrades or reinstallation.
uv pip install -r "${AUTOGLUON_TABULAR_REQ_FILE}" --prerelease=allow -r "${COMMON_REQ_FILE}"

echo "Conda environment setup complete."