#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the target conda environment path
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/mnist_784/rep-1/ba3cf320b37c1083b0e8/attempt-1/mlzero/node_1/conda_env"

# Define paths to the requirements files
MULTIMODAL_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"

echo "--- Starting Conda environment setup ---"
echo "Target Conda environment path: ${CONDA_ENV_PATH}"

# --- Conda Initialization ---
# Check if conda is already initialized in the current shell.
# If not, attempt to find and source the conda.sh script.
if ! type "_conda_activate" &> /dev/null; then
    echo "Conda not initialized. Attempting to initialize..."
    CONDA_SH_PATH=""
    # Try common Miniconda/Anaconda installation paths
    if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        CONDA_SH_PATH="${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        CONDA_SH_PATH="/opt/conda/etc/profile.d/conda.sh"
    elif command -v conda &> /dev/null; then
        # If 'conda' command is available but not sourced, try to find its base path
        CONDA_BASE=$(conda info --base)
        if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
            CONDA_SH_PATH="${CONDA_BASE}/etc/profile.d/conda.sh"
        fi
    fi

    if [ -n "${CONDA_SH_PATH}" ]; then
        source "${CONDA_SH_PATH}"
        echo "Conda initialized from ${CONDA_SH_PATH}"
    else
        echo "Error: Could not find conda initialization script. Please ensure conda is installed and accessible in your PATH."
        exit 1
    fi
fi
# --- End Conda Initialization ---

# Remove existing environment if it exists to ensure a clean slate
if [ -d "${CONDA_ENV_PATH}" ]; then
    echo "Removing existing conda environment at ${CONDA_ENV_PATH}..."
    conda env remove --prefix "${CONDA_ENV_PATH}" -y
fi

# Create the conda environment with Python 3.11
echo "Creating conda environment with Python 3.11 at ${CONDA_ENV_PATH}..."
conda create --prefix "${CONDA_ENV_PATH}" python=3.11 -y

# Activate the newly created environment
echo "Activating conda environment: ${CONDA_ENV_PATH}"
conda activate "${CONDA_ENV_PATH}"

# Verify activation
if [[ "${CONDA_PREFIX}" != "${CONDA_ENV_PATH}" ]]; then
    echo "Error: Conda environment activation failed. CONDA_PREFIX is '${CONDA_PREFIX}', expected '${CONDA_ENV_PATH}'"
    exit 1
fi
echo "Conda environment activated successfully. Python interpreter: $(which python)"

# Install uv
echo "Installing uv into the environment..."
pip install uv

# Install numpy, pandas, scikit-learn using uv
echo "Installing numpy, pandas, scikit-learn using uv..."
uv pip install numpy pandas scikit-learn

# Install required packages from the specified requirements files using uv
# The command is provided explicitly in the prompt, ensuring correct order and flags.
echo "Installing packages from requirements files using uv..."
uv pip install -r "${MULTIMODAL_REQUIREMENTS_FILE}" --prerelease=allow -r "${COMMON_REQUIREMENTS_FILE}"

echo "--- Conda environment setup complete ---"

# Deactivate the environment (optional, but good practice for scripts)
# conda deactivate