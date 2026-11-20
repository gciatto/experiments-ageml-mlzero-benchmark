#!/bin/bash
set -euo pipefail # Exit on error, exit on unset variables, propagate errors in pipelines

# Define the target Conda environment path
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-2/ffb0a47cbef81e5b1c5d/attempt-1/mlzero/node_7/conda_env"

# Define paths to requirements files
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda command not found. Please ensure Anaconda or Miniconda is installed and in your PATH." >&2
    exit 1
fi

# Safely initialize Conda for the current shell session if not already done
# This addresses the CONDA_SHLVL unbound variable error with set -u
if [ -z "${CONDA_SHLVL:-}" ]; then
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh" || { echo "Failed to source conda.sh" >&2; exit 1; }
fi

# Create the conda environment with Python 3.11
conda create -p "$CONDA_ENV_PATH" python=3.11 -y || { echo "Failed to create conda environment." >&2; exit 1; }

# Activate the environment
conda activate "$CONDA_ENV_PATH" || { echo "Failed to activate conda environment." >&2; exit 1; }

# Install uv using the activated environment's pip
python -m pip install uv || { echo "Failed to install uv." >&2; exit 1; }

# Install required packages from requirements.txt files using uv
# The 'uv' command should now be available in the PATH of the activated environment
uv pip install -r "$ML_REQUIREMENTS_FILE" --prerelease=allow -r "$COMMON_REQUIREMENTS_FILE" || { echo "Failed to install packages from requirements files." >&2; exit 1; }

# Install additional packages: numpy, pandas, scikit-learn, and autogluon.tabular
uv pip install numpy pandas scikit-learn autogluon.tabular || { echo "Failed to install core ML libraries." >&2; exit 1; }

echo "Conda environment setup complete at $CONDA_ENV_PATH"