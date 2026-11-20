#!/bin/bash
set -euo pipefail

# Define the target conda environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-3/5a9401ba768907dcd35e/attempt-1/mlzero/node_10/conda_env"

# Define paths to the requirements files
COMMON_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

# 1. Ensure conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: 'conda' command not found. Please ensure Anaconda/Miniconda is installed and in your PATH." >&2
    exit 1
fi

# 2. Remove existing environment if it exists, to ensure a clean installation
# Using 'conda env remove -p' is safer than 'rm -rf' as it handles conda's internal tracking.
if [ -d "$ENV_PATH" ]; then
    echo "Removing existing conda environment at $ENV_PATH..."
    conda env remove -p "$ENV_PATH" -y || true # '|| true' prevents script from failing if env doesn't exist
fi

# 3. Create a new conda environment with Python 3.11
echo "Creating new conda environment at $ENV_PATH with Python 3.11..."
conda create -p "$ENV_PATH" python=3.11 -y

# 4. Activate the newly created environment
echo "Activating conda environment..."
# Temporarily disable 'set -u' for conda initialization and activation to avoid 'unbound variable' errors
set +u
# Robust way to initialize conda in a non-interactive script, as suggested by previous errors.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"
set -u # Re-enable 'set -u'

# 5. Install core scientific packages via conda
# This addresses previous ModuleNotFoundError for numpy and ensures robust installation of core libraries.
echo "Installing numpy, pandas, scikit-learn via conda..."
conda install -y numpy pandas scikit-learn

# 6. Install uv (a fast Python package installer) using the activated environment's pip
echo "Installing uv..."
pip install uv

# 7. Install required packages from the specified requirements.txt files and autogluon.tabular using uv
echo "Installing packages from requirements files and autogluon.tabular using uv..."
uv pip install \
    -r "$ML_REQ_FILE" \
    --prerelease=allow \
    -r "$COMMON_REQ_FILE" \
    autogluon.tabular

echo "--- Conda Environment Setup Complete ---"