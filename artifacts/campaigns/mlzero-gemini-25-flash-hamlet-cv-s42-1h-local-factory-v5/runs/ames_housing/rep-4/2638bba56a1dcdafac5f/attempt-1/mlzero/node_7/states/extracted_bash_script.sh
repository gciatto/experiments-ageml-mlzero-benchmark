#!/bin/bash
set -euo pipefail

# --- Configuration ---
# Path to the base conda installation. Adjust this if your conda installation is in a different location.
# Common paths include /opt/conda, /usr/local/conda, or ~/miniconda3.
CONDA_BASE_PATH="/opt/conda"

# The full path for the new conda environment as specified in the prompt.
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-4/2638bba56a1dcdafac5f/attempt-1/mlzero/node_7/conda_env"

# Paths to the requirements files as specified in the prompt.
COMMON_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# --- Conda Initialization ---
# Ensure conda is properly initialized in the current shell session.
# This addresses previous errors related to conda activation.
if [ -f "$CONDA_BASE_PATH/etc/profile.d/conda.sh" ]; then
    echo "Sourcing conda initialization script from $CONDA_BASE_PATH..."
    source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
else
    echo "Error: Conda initialization script not found at $CONDA_BASE_PATH/etc/profile.d/conda.sh" >&2
    echo "Please ensure CONDA_BASE_PATH is set correctly and conda is installed." >&2
    exit 1
fi

# --- Environment Setup ---
# Remove existing environment if it exists to ensure a clean slate for installation.
if conda env list | grep -q "$ENV_PATH"; then
    echo "Removing existing conda environment at $ENV_PATH to ensure a clean installation..."
    conda env remove -p "$ENV_PATH" -y || { echo "Failed to remove existing conda environment." >&2; exit 1; }
fi

# Create the new conda environment with the specified Python version.
echo "Creating new conda environment with Python 3.11 at $ENV_PATH..."
conda create -p "$ENV_PATH" python=3.11 -y || { echo "Failed to create conda environment." >&2; exit 1; }

# Activate the newly created environment.
echo "Activating conda environment: $ENV_PATH"
conda activate "$ENV_PATH" || { echo "Failed to activate conda environment." >&2; exit 1; }

# --- Package Installation ---
echo "Installing 'uv', numpy, pandas, scikit-learn, and requirements from specified files..."
# Install 'uv' first, then use it for other package installations.
pip install uv || { echo "Failed to install 'uv'." >&2; exit 1; }

# Install core packages (numpy, pandas, scikit-learn) and all required packages
# from the specified requirements files using 'uv'.
# 'uv' inherently respects exact package versions and avoids reinstallation if already satisfied,
# aligning with the requirement to not upgrade/reinstall autogluon.multimodal if at the correct version.
uv pip install numpy pandas scikit-learn \
    -r "$MULTIMODAL_REQ_FILE" --prerelease=allow \
    -r "$COMMON_REQ_FILE" || { echo "Failed to install required packages." >&2; exit 1; }

echo "Conda environment setup complete at $ENV_PATH."