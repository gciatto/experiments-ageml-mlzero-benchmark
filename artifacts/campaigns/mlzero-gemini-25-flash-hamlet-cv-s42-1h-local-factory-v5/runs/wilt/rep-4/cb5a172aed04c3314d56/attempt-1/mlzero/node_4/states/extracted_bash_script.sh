#!/bin/bash
set -euo pipefail

# Define the target Conda environment path
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/wilt/rep-4/cb5a172aed04c3314d56/attempt-1/mlzero/node_4/conda_env"

# Define paths to requirement files
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

# Create parent directories for the conda environment path if they don't exist
mkdir -p "$(dirname "$CONDA_ENV_PATH")"

# Initialize Conda for the current shell session
# This is crucial for 'conda activate' to work correctly.
# It sources the conda initialization script from the base conda environment.
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found. Ensure Conda is installed and initialized in your shell's PATH." >&2
    exit 1
fi

# 1. Create the Conda environment with Python 3.11
echo "Creating Conda environment at $CONDA_ENV_PATH with Python 3.11..."
conda create -p "$CONDA_ENV_PATH" python=3.11 -y

# 2. Activate the environment
echo "Activating Conda environment..."
conda activate "$CONDA_ENV_PATH"

# 3. Install uv
echo "Installing uv into the environment..."
pip install uv

# 4. Install required packages from requirements.txt files using uv
echo "Installing packages from requirement files using uv..."
uv pip install -r "$ML_REQUIREMENTS_FILE" --prerelease=allow -r "$COMMON_REQUIREMENTS_FILE"

# 5. Install additional packages: numpy, pandas, scikit-learn, and autogluon
# 'autogluon' is the selected ML library, which includes autogluon.tabular
echo "Installing numpy, pandas, scikit-learn, and autogluon using uv..."
uv pip install numpy pandas scikit-learn autogluon

echo "Conda environment setup complete."