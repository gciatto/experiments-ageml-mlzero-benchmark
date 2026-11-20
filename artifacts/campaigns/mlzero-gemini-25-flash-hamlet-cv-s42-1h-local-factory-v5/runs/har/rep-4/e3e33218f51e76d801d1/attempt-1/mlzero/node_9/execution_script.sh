#!/bin/bash
set -euo pipefail

# Define the base directory for the environment as specified in the prompt
# The path is specifically for 'node_9' as requested.
BASE_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/har/rep-4/e3e33218f51e76d801d1/attempt-1/mlzero/node_9"
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

# Remove existing environment to ensure a clean and consistent installation.
# This helps prevent conflicts from previous failed attempts or partial installations.
if [ -d "${CONDA_ENV_PATH}" ] && [ -f "${CONDA_ENV_PATH}/bin/python" ]; then
    echo "Removing existing conda environment at ${CONDA_ENV_PATH}..."
    # Use --prefix for robustness with environments not in the default list.
    conda env remove --prefix "${CONDA_ENV_PATH}" --yes || {
        echo "Warning: Failed to remove existing conda environment via 'conda env remove'. Attempting to remove directory directly."
        rm -rf "${CONDA_ENV_PATH}"
    }
fi

# Create the conda environment with Python 3.11 at the specified path.
echo "Creating conda environment with Python 3.11 at ${CONDA_ENV_PATH}..."
conda create --prefix "${CONDA_ENV_PATH}" python=3.11 --yes || {
    echo "Error: Failed to create conda environment at ${CONDA_ENV_PATH}."
    echo "Please check disk space, permissions, and conda installation."
    exit 1
}

# Verify environment creation by checking for the activation script.
# This addresses previous errors where the environment was incomplete.
if [ ! -f "${CONDA_ENV_PATH}/bin/activate" ]; then
    echo "Error: Conda environment activation script not found at ${CONDA_ENV_PATH}/bin/activate."
    echo "This indicates an incomplete or corrupted environment setup. Please check disk space and permissions."
    exit 1
fi

# Activate the conda environment.
echo "Activating the conda environment..."
# Source the conda.sh script to properly initialize conda in the current shell context.
# This is essential for 'conda activate' to work reliably in non-interactive scripts.
source "$(conda info --base)/etc/profile.d/conda.sh" || {
    echo "Error: Failed to source conda.sh. Ensure conda is properly installed and initialized."
    exit 1
}
conda activate "${CONDA_ENV_PATH}" || {
    echo "Error: Failed to activate conda environment at ${CONDA_ENV_PATH}."
    exit 1
}

# Install core scientific packages (numpy, pandas, scikit-learn) using conda.
# This ensures these fundamental packages are robustly installed and managed by conda,
# addressing previous ModuleNotFoundError issues.
echo "Installing numpy, pandas, scikit-learn using conda..."
conda install numpy pandas scikit-learn --yes || {
    echo "Error: Failed to install core packages via conda."
    exit 1
}

# Install uv into the activated environment using its pip.
echo "Installing uv into the environment..."
pip install uv || {
    echo "Error: Failed to install uv."
    exit 1
}

# Install required packages from the specified requirements files using uv.
# uv ensures exact package versions and minimal changes, respecting existing installations.
# --prerelease=allow is included as it's often necessary for AutoGluon dependencies.
# The order of requirement files is as specified in the prompt.
echo "Installing required packages from requirements files using uv..."
uv pip install \
    -r "${AUTOGLUON_TABULAR_REQ_FILE}" \
    --prerelease=allow \
    -r "${COMMON_REQ_FILE}" \
    || {
    echo "Error: Failed to install packages via uv."
    exit 1
}

echo "--- Conda Environment Setup Complete ---"

# The environment remains active for subsequent operations by the harness.
# No training, predictions, holdout evaluation, file-generation scripts,
# or the factory module itself are executed by this script, as per instructions.