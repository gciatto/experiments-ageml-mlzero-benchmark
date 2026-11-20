#!/bin/bash
set -euo pipefail

# Define environment and requirements paths
# The full path to the conda_env directory for this node
ENV_FULL_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-1/4d252b8d3aa6da775378/attempt-1/mlzero/node_4/conda_env"
AUTOGLUON_TABULAR_REQUIREMENTS="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"
COMMON_REQUIREMENTS="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"

echo "--- Starting Conda environment setup ---"

# Robust Conda initialization for non-interactive scripts
# This addresses the previous error regarding 'conda activate' not being configured.
# It finds the conda executable and sources its initialization script directly.
CONDA_EXE=$(which conda)
if [ -z "$CONDA_EXE" ]; then
    echo "Error: 'conda' command not found. Please ensure Conda is installed and in your system's PATH." >&2
    exit 1
fi

# Get the base directory of conda and source conda.sh
CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "Sourcing Conda initialization script from $CONDA_BASE/etc/profile.d/conda.sh"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "Error: Conda initialization script not found at $CONDA_BASE/etc/profile.d/conda.sh" >&2
    exit 1
fi

# 1. Create Conda environment with Python 3.11
echo "Creating conda environment at '$ENV_FULL_PATH' with Python 3.11..."
# Ensure the parent directory exists before creating the environment
mkdir -p "$(dirname "$ENV_FULL_PATH")"
conda create -p "$ENV_FULL_PATH" python=3.11 -y

# 2. Activate the environment
echo "Activating environment '$ENV_FULL_PATH'..."
conda activate "$ENV_FULL_PATH"

# 3. Install uv
echo "Installing uv into the environment..."
# Use the activated environment's pip
pip install uv

# 4. Install required packages using uv
echo "Installing required packages from '$AUTOGLUON_TABULAR_REQUIREMENTS', '$COMMON_REQUIREMENTS', numpy, pandas, and scikit-learn using uv..."
uv pip install \
    -r "$AUTOGLUON_TABULAR_REQUIREMENTS" \
    --prerelease=allow \
    -r "$COMMON_REQUIREMENTS" \
    numpy pandas scikit-learn

echo "--- Conda environment setup complete ---"