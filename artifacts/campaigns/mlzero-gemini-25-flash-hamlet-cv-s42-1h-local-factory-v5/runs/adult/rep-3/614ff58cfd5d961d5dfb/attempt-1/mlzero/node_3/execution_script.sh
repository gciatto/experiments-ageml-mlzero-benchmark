#!/bin/bash
# Minimal BASH INSTALLATION script for the factory module

# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error when substituting.
# The return value of a pipeline is the status of the last command to exit with a non-zero status,
# or zero if all commands in the pipeline exit successfully.
set -euo pipefail

# --- Configuration ---
# Corrected CONDA_ENV_FULL_PATH to point to node_3 as specified in the prompt.
CONDA_ENV_FULL_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/adult/rep-3/614ff58cfd5d961d5dfb/attempt-1/mlzero/node_3/conda_env"
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AUTOGLUON_TABULAR_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"
PYTHON_VERSION="3.11"

# --- Conda Initialization ---
# Ensure conda is initialized for the script.
# This is crucial for 'conda activate' to work correctly in non-interactive shells.
if command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base)
    if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        . "${CONDA_BASE}/etc/profile.d/conda.sh"
    else
        echo "ERROR: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh. Please ensure Conda is installed and initialized in your shell environment."
        exit 1
    fi
else
    echo "ERROR: 'conda' command not found. Please ensure Conda is installed and available in your PATH."
    exit 1
fi

# --- Environment Creation ---
echo "Creating conda environment at ${CONDA_ENV_FULL_PATH} with Python ${PYTHON_VERSION}..."
# Use --prefix to specify the exact path for the environment.
conda create -y --prefix "${CONDA_ENV_FULL_PATH}" python="${PYTHON_VERSION}"

# --- Environment Activation ---
echo "Activating environment: ${CONDA_ENV_FULL_PATH}"
conda activate "${CONDA_ENV_FULL_PATH}"

# Check if activation was successful (though set -e should catch it)
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment. Exiting."
    exit 1
fi

# --- Install uv ---
echo "Installing uv into the activated environment..."
pip install uv

# --- Install Required Packages using uv ---
echo "Installing required packages from requirements files using uv..."
# The prompt specifies the exact command structure.
# uv handles dependencies and avoids unnecessary upgrades by default,
# respecting existing installations if they match the requirements.
uv pip install \
    -r "${AUTOGLUON_TABULAR_REQUIREMENTS_FILE}" \
    --prerelease=allow \
    -r "${COMMON_REQUIREMENTS_FILE}"

# --- Install Core ML Libraries ---
# Explicitly install numpy, pandas, and scikit-learn as requested.
# These are fundamental for the factory module and often implicitly expected.
echo "Installing numpy, pandas, and scikit-learn into the activated environment using uv..."
uv pip install numpy pandas scikit-learn

echo "Conda environment setup complete at ${CONDA_ENV_FULL_PATH}"