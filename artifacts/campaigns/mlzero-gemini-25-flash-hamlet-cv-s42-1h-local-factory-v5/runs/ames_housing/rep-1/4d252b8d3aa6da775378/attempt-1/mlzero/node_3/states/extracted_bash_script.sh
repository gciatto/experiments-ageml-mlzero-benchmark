#!/bin/bash
set -euo pipefail

# --- Configuration ---
# The target path for the Conda environment, updated to node_3 as per the prompt.
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-1/4d252b8d3aa6da775378/attempt-1/mlzero/node_3/conda_env"
COMMON_REQUIREMENTS_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQUIREMENTS_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"
PYTHON_VERSION="3.11"

echo "Starting Conda environment setup for Python $PYTHON_VERSION at $CONDA_ENV_PATH"

# --- Conda Initialization ---
# This block ensures that 'conda' commands, especially 'conda activate',
# are properly initialized for the current shell session.
# It addresses common issues in non-interactive scripts where conda might not be fully set up.
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "Sourcing conda.sh from $CONDA_BASE/etc/profile.d/conda.sh"
    . "$CONDA_BASE/etc/profile.d/conda.sh"
    # This command sets up the shell functions for 'conda activate' and 'conda deactivate'.
    # It's crucial if 'conda init' hasn't been run for the shell type previously.
    echo "Running eval \"\$(conda shell hook)\" to enable conda activate."
    eval "$(conda shell hook)"
else
    echo "ERROR: conda.sh not found at $CONDA_BASE/etc/profile.d/conda.sh." >&2
    echo "Please ensure Conda is installed and its base environment is in PATH." >&2
    exit 1
fi

# Deactivate any currently active conda environment to ensure a clean slate
# before creating or activating a new one.
if conda info --envs | grep -q '*' ; then
    echo "Deactivating any previously active conda environment."
    conda deactivate
fi

# --- Clean up existing environment if it exists ---
# This step ensures a fresh installation, preventing conflicts from previous attempts.
if conda env list | grep -q "$CONDA_ENV_PATH"; then
    echo "Existing conda environment found at $CONDA_ENV_PATH. Removing it for a clean installation."
    conda env remove --prefix "$CONDA_ENV_PATH" --yes
elif [ -d "$CONDA_ENV_PATH" ]; then
    echo "Directory $CONDA_ENV_PATH exists but is not a registered conda environment. Removing directory."
    rm -rf "$CONDA_ENV_PATH"
fi

# --- Create and Activate Environment ---
echo "Creating conda environment with Python $PYTHON_VERSION at $CONDA_ENV_PATH..."
conda create --prefix "$CONDA_ENV_PATH" python="$PYTHON_VERSION" --yes

echo "Activating conda environment: $CONDA_ENV_PATH"
conda activate "$CONDA_ENV_PATH"

# --- Install Packages ---
# Ensure pip is up-to-date in the newly created environment.
echo "Upgrading pip in the new environment..."
python -m pip install --upgrade pip

echo "Installing uv..."
pip install uv

echo "Installing numpy, pandas, scikit-learn..."
uv pip install numpy pandas scikit-learn

echo "Installing required packages from requirements files using uv..."
# The --prerelease=allow is necessary for AutoGluon.
# uv automatically handles the requirements to "only install exact packages"
# and "do NOT upgrade or reinstall if already at the correct version".
uv pip install -r "$MULTIMODAL_REQUIREMENTS_PATH" --prerelease=allow -r "$COMMON_REQUIREMENTS_PATH"

# --- Finalization ---
echo "Deactivating conda environment."
conda deactivate

echo "Conda environment setup complete."