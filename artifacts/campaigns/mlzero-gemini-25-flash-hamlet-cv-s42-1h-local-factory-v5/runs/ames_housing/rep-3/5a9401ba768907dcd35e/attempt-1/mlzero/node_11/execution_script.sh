#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error.

# Define environment path and requirements files
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-3/5a9401ba768907dcd35e/attempt-1/mlzero/node_11/conda_env"
COMMON_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# --- Conda Initialization and Environment Setup ---

# Ensure conda is initialized in the shell.
# This is crucial for 'conda activate' to work correctly.
# Check if conda is available and initialize it.
if command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base)
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        echo "Sourcing conda.sh from $CONDA_BASE..."
        # Temporarily disable 'set -u' around sourcing conda.sh to prevent issues with unset variables
        # that conda might expect to be set (e.g., CONDA_SHLVL).
        set +u
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        set -u
    else
        echo "ERROR: conda.sh not found at $CONDA_BASE/etc/profile.d/conda.sh. Please ensure Conda is installed and initialized."
        exit 1
    fi
else
    echo "ERROR: Conda command not found. Please ensure Conda is installed and in your PATH."
    exit 1
fi

# Create the parent directory for the conda environment if it doesn't exist
mkdir -p "$(dirname "$CONDA_ENV_PATH")"

# Create the conda environment with Python 3.11
echo "Creating conda environment at $CONDA_ENV_PATH with Python 3.11..."
conda create -p "$CONDA_ENV_PATH" python=3.11 -y || { echo "Failed to create conda environment."; exit 1; }

# Activate the newly created environment.
# Temporarily disable 'set -u' around 'conda activate' to prevent 'unbound variable' errors
# like CONDA_SHLVL, which can occur if the variable isn't set before activation.
echo "Activating conda environment..."
set +u
conda activate "$CONDA_ENV_PATH" || { echo "Failed to activate conda environment."; exit 1; }
set -u
echo "Conda environment activated."

# --- Package Installation using uv ---

# Install uv (Rust-based pip replacement for faster and more reliable installs)
echo "Installing uv..."
pip install uv || { echo "Failed to install uv."; exit 1; }

# Install core packages: numpy, pandas, scikit-learn using uv pip
echo "Installing numpy, pandas, scikit-learn using uv pip..."
uv pip install numpy pandas scikit-learn || { echo "Failed to install core ML packages."; exit 1; }

# Install required packages from the specified requirements files using uv pip.
# uv handles dependency resolution and ensures exact versions are met without unnecessary upgrades.
# The `--prerelease=allow` flag is crucial for AutoGluon.
echo "Installing packages from requirements files using uv pip..."
uv pip install \
    -r "$MULTIMODAL_REQ_FILE" \
    --prerelease=allow \
    -r "$COMMON_REQ_FILE" || { echo "Failed to install packages from requirements files."; exit 1; }

echo "Installation complete for environment: $CONDA_ENV_PATH"