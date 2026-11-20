#!/bin/bash
set -euo pipefail

# Define the target conda environment path
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/wilt/rep-1/9083f22fdcbb98fc1c37/attempt-1/mlzero/node_3/conda_env"

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
# The -y flag automatically confirms prompts. If the environment exists, conda create will prompt to remove it.
# This command will fail if the environment directory exists and is not empty, and -y is used.
# For a minimal script, we rely on `set -e` to catch such failures.
conda create -p "$CONDA_ENV_PATH" python=3.11 -y || { echo "Failed to create conda environment at '$CONDA_ENV_PATH'." >&2; exit 1; }

# Activate the environment
echo "Activating the newly created conda environment..."
# Initialize conda for the current shell session if not already done
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then # Common path in Docker containers
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "Error: 'conda.sh' not found. Cannot activate conda environment. Please check conda installation." >&2
    exit 1
fi
conda activate "$CONDA_ENV_PATH" || { echo "Failed to activate conda environment '$CONDA_ENV_PATH'." >&2; exit 1; }

# Install uv using the environment's pip
echo "Installing 'uv' into the conda environment..."
pip install uv || { echo "Failed to install 'uv'." >&2; exit 1; }

# Install core libraries: numpy, pandas, scikit-learn
echo "Installing numpy, pandas, scikit-learn into the conda environment using uv..."
uv pip install numpy pandas scikit-learn || { echo "Failed to install core libraries (numpy, pandas, scikit-learn)." >&2; exit 1; }

# Install required packages from the specified requirements files using uv
echo "Installing required packages from '$AUTOGLUON_TABULAR_REQ_PATH' and '$COMMON_REQ_PATH' using uv..."
# The prompt specifies the exact uv command, including --prerelease=allow
uv pip install -r "$AUTOGLUON_TABULAR_REQ_PATH" --prerelease=allow -r "$COMMON_REQ_PATH" || { echo "Failed to install packages from requirements files." >&2; exit 1; }

echo "Conda environment setup complete at '$CONDA_ENV_PATH'."