#!/bin/bash
set -euo pipefail

# Define environment and file paths
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/california_housing/rep-4/0667550118be921327cc/attempt-1/mlzero/node_1/conda_env"
PYTHON_VERSION="3.11"
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

echo "--- Starting Conda Environment Setup ---"

# Ensure the parent directory for the conda environment exists
mkdir -p "$(dirname "$CONDA_ENV_PATH")"

# 1. Create the conda environment with the specified Python version
echo "Creating conda environment at $CONDA_ENV_PATH with Python $PYTHON_VERSION..."
conda create -p "$CONDA_ENV_PATH" python="$PYTHON_VERSION" -y

# Check if conda create was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to create conda environment." >&2
    exit 1
fi

# 2. Activate the environment
echo "Activating conda environment..."
# Initialize conda for the current shell if not already done
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

# Check if conda activate was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment." >&2
    exit 1
fi

# 3. Install uv into the activated environment
echo "Installing uv into the environment..."
pip install uv

# Check if uv installation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to install uv." >&2
    exit 1
fi

# 4. Install core packages (numpy, pandas, scikit-learn) using uv
echo "Installing numpy, pandas, scikit-learn using uv..."
uv pip install numpy pandas scikit-learn

# Check if core package installation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to install core packages with uv." >&2
    exit 1
fi

# 5. Install required packages from the specified requirements files using uv
# This command ensures only exact packages and their dependencies are installed,
# and handles prereleases as specified. uv efficiently avoids reinstallation
# if packages are already at the correct version.
echo "Installing packages from requirements files using uv..."
uv pip install -r "$MULTIMODAL_REQUIREMENTS_FILE" --prerelease=allow -r "$COMMON_REQUIREMENTS_FILE"

# Check if requirements file installation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to install packages from requirements files with uv." >&2
    exit 1
fi

echo "--- Conda Environment Setup Complete ---"