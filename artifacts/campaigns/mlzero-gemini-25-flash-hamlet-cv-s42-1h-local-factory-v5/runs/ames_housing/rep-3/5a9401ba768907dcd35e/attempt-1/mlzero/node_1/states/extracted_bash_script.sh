#!/bin/bash
set -euo pipefail

# Define the target conda environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-3/5a9401ba768907dcd35e/attempt-1/mlzero/node_1/conda_env"

# Define paths to the requirements files
COMMON_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

echo "--- Starting Conda Environment Setup ---"

# 1. Ensure conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: 'conda' command not found. Please ensure Anaconda/Miniconda is installed and in your PATH."
    exit 1
fi

# 2. Remove existing environment if it exists, to ensure a clean installation
if [ -d "$ENV_PATH" ]; then
    echo "Removing existing conda environment at $ENV_PATH..."
    conda env remove -p "$ENV_PATH" -y || true # '|| true' prevents script from failing if env doesn't exist
fi

# 3. Create a new conda environment with Python 3.11
echo "Creating new conda environment at $ENV_PATH with Python 3.11..."
conda create -p "$ENV_PATH" python=3.11 -y

# 4. Activate the newly created environment
echo "Activating conda environment..."
# Initialize conda for the current shell session if not already done
if [ -z "$CONDA_SHLVL" ]; then
    eval "$(conda shell hook --shell bash)"
fi
conda activate "$ENV_PATH"

# 5. Install uv (a fast Python package installer)
echo "Installing uv..."
pip install uv

# 6. Install required packages from the specified requirements.txt files using uv
echo "Installing packages from requirements files using uv..."
uv pip install -r "$ML_REQ_FILE" --prerelease=allow -r "$COMMON_REQ_FILE"

# 7. Install additional packages required by the factory module:
#    numpy, pandas, scikit-learn, and the selected ML library (autogluon.tabular).
#    autogluon.tabular typically pulls in numpy and pandas as dependencies.
echo "Installing autogluon.tabular and scikit-learn..."
uv pip install autogluon.tabular scikit-learn

echo "--- Conda Environment Setup Complete ---"