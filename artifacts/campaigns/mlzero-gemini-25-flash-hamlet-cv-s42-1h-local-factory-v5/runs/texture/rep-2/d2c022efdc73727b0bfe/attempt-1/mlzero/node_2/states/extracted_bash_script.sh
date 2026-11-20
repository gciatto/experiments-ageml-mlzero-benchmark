#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the full path for the conda environment
ENV_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/texture/rep-2/d2c022efdc73727b0bfe/attempt-1/mlzero/node_2/conda_env"

# Define paths to requirements files
REQUIREMENTS_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_MULTIMODAL="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

echo "--- Starting Conda environment setup ---"
echo "Environment directory: $ENV_DIR"

# 1. Create the conda environment with Python 3.11
# Use -p for a specific path, -y for non-interactive mode.
echo "Creating conda environment at $ENV_DIR with Python 3.11..."
conda create -p "$ENV_DIR" python=3.11 -y

# 2. Activate the environment
# This is crucial for subsequent commands to use the new environment's python/pip.
echo "Activating conda environment..."
# Source conda.sh to make 'conda activate' available.
# This assumes conda is installed and its base directory can be found.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# Verify environment activation
if [[ "$CONDA_PREFIX" != "$ENV_DIR" ]]; then
    echo "Error: Conda environment activation failed. CONDA_PREFIX is '$CONDA_PREFIX', expected '$ENV_DIR'"
    exit 1
fi
echo "Conda environment activated: $CONDA_PREFIX"

# 3. Install uv using the environment's pip
echo "Installing uv..."
pip install uv

# 4. Install required packages using uv
# The prompt explicitly states the uv command and requirements files.
# numpy, pandas, scikit-learn are added explicitly as requested,
# though they are likely dependencies of autogluon.multimodal.
# uv handles exact versions and avoids unnecessary upgrades/reinstalls.
echo "Installing packages from requirements files and core libraries using uv..."
uv pip install \
    -r "$REQUIREMENTS_MULTIMODAL" \
    --prerelease=allow \
    -r "$REQUIREMENTS_COMMON" \
    numpy pandas scikit-learn

echo "--- Conda environment setup complete ---"