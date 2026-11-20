#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
set -o pipefail # Exit if any command in a pipeline fails.

# Define environment path and requirement file paths
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/auto_mpg/rep-1/f5904a8064f3eac0500a/attempt-1/mlzero/node_14/conda_env"
COMMON_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

# 1. Ensure parent directories exist for the conda environment
mkdir -p "$(dirname "$CONDA_ENV_PATH")" || { echo "Error: Failed to create parent directories for conda environment."; exit 1; }

# 2. Create Conda environment with Python 3.11
conda create -p "$CONDA_ENV_PATH" python=3.11 -y -q || { echo "Error: Failed to create conda environment."; exit 1; }

# 3. Activate the environment
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh" || { echo "Error: Could not source conda.sh."; exit 1; }
else
    echo "Error: conda.sh not found at $CONDA_BASE/etc/profile.d/conda.sh. Is conda installed and in PATH?"
    exit 1
fi
conda activate "$CONDA_ENV_PATH" || { echo "Error: Could not activate conda environment."; exit 1; }

# 4. Install uv
pip install uv -q || { echo "Error: Failed to install uv."; exit 1; }

# 5. Install required packages from the specified requirement files using uv
uv pip install -r "$ML_REQ_FILE" --prerelease=allow -r "$COMMON_REQ_FILE" -q || { echo "Error: Failed to install packages from requirement files."; exit 1; }

# 6. Install core packages: numpy, pandas, scikit-learn and autogluon.tabular
uv pip install numpy pandas scikit-learn autogluon.tabular -q || { echo "Error: Failed to install core packages and autogluon.tabular."; exit 1; }