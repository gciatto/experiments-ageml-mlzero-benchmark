#!/bin/bash
set -euo pipefail

# Define the target conda environment path
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/wilt/rep-1/9083f22fdcbb98fc1c37/attempt-1/mlzero/node_4/conda_env"

# Define paths to the requirements files
COMMON_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AUTOGLUON_TABULAR_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: 'conda' command not found. Please ensure Anaconda or Miniconda is installed and in your PATH." >&2
    exit 1
fi

# Create the conda environment with Python 3.11
echo "Creating conda environment at '$CONDA_ENV_PATH' with Python 3.11..."
conda create -p "$CONDA_ENV_PATH" python=3.11 --yes

# Activate the environment
echo "Activating the newly created conda environment..."
# Initialize conda for the current shell session if not already done
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Error: 'conda.sh' not found. Cannot activate conda environment." >&2
    exit 1
fi
conda activate "$CONDA_ENV_PATH"

# Install uv using the environment's pip
echo "Installing 'uv' into the conda environment..."
pip install uv

# Install required packages using uv, as specified
# uv will only install exact packages and their dependencies, and will not
# upgrade/reinstall if already at the correct version.
echo "Installing required packages from '$AUTOGLUON_TABULAR_REQ_PATH' and '$COMMON_REQ_PATH' using uv..."
uv pip install -r "$AUTOGLUON_TABULAR_REQ_PATH" --prerelease=allow -r "$COMMON_REQ_PATH"

echo "Conda environment setup complete at '$CONDA_ENV_PATH'."