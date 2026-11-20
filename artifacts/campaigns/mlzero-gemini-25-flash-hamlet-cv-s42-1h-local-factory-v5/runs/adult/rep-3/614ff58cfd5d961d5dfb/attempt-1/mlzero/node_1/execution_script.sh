#!/bin/bash
set -euo pipefail

# Define the target Conda environment path
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/adult/rep-3/614ff58cfd5d961d5dfb/attempt-1/mlzero/node_1/conda_env"

# Define paths to requirement files
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

echo "--- Starting Conda environment setup ---"

# 1. Create the Conda environment with Python 3.11
echo "Creating conda environment at '$CONDA_ENV_PATH' with Python 3.11..."
# Ensure the parent directory exists
mkdir -p "$(dirname "$CONDA_ENV_PATH")"
conda create -y -p "$CONDA_ENV_PATH" python=3.11

# 2. Activate the environment
echo "Activating conda environment..."
# Initialize conda for the current shell session if not already done
# This is crucial for 'conda activate' to work in non-interactive scripts
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found. Ensure Conda is installed and initialized." >&2
    exit 1
fi
conda activate "$CONDA_ENV_PATH"

# 3. Install uv using the environment's pip
echo "Installing 'uv' into the environment..."
pip install uv

# 4. Install required packages using uv pip install
echo "Installing packages from requirements files and core libraries using uv..."
# The order of -r flags is specified: autogluon.multimodal first, then _common.
# numpy, pandas, scikit-learn are added to be installed by uv for consistency.
uv pip install \
    -r "$MULTIMODAL_REQUIREMENTS_FILE" \
    --prerelease=allow \
    -r "$COMMON_REQUIREMENTS_FILE" \
    numpy pandas scikit-learn

# 5. Deactivate the environment (optional, but good practice)
echo "Deactivating conda environment..."
conda deactivate

echo "--- Conda environment setup complete ---"