#!/bin/bash
set -euo pipefail # Exit immediately if a command exits with a non-zero status, exit on unset variables, pipefail

# Define the full path for the conda environment
ENV_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/texture/rep-4/79235980f285c36790bd/attempt-1/mlzero/node_2/conda_env"

# Define paths to the requirements files
COMMON_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

# --- Conda Initialization ---
# Check if conda is available and initialize it for the current shell session.
# This is crucial for 'conda activate' to work in a non-interactive script.
if command -v conda &> /dev/null; then
    if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    else
        echo "Warning: conda.sh not found. Conda might not be fully initialized. Attempting to proceed." >&2
    fi
else
    echo "Error: 'conda' command not found. Please ensure Conda is installed and in your PATH." >&2
    exit 1
fi

# --- Environment Setup ---
# Remove existing environment if it exists to ensure a clean installation.
# Using -p for path-based environments.
if conda env list | grep -q "${ENV_DIR}"; then
    echo "Removing existing conda environment at ${ENV_DIR}..."
    conda env remove -p "${ENV_DIR}" -y
fi

# Create the conda environment with Python 3.11 at the specified path.
echo "Creating conda environment at ${ENV_DIR} with Python 3.11..."
conda create -p "${ENV_DIR}" python=3.11 -y

# Activate the newly created environment.
echo "Activating conda environment: ${ENV_DIR}"
conda activate "${ENV_DIR}"

# --- Package Installation ---
# Install uv using the environment's pip.
echo "Installing uv..."
pip install uv

# Install core packages: numpy, pandas, scikit-learn, and autogluon.
# This ensures the explicitly requested libraries are present.
echo "Installing numpy, pandas, scikit-learn, and autogluon using uv..."
uv pip install numpy pandas scikit-learn autogluon

# Install additional packages from the specified requirements files using uv.
# uv will handle dependencies and potential conflicts with already installed packages.
echo "Installing additional packages from requirements files using uv..."
uv pip install -r "${ML_REQ_PATH}" --prerelease=allow -r "${COMMON_REQ_PATH}"

echo "Conda environment setup complete at ${ENV_DIR}"