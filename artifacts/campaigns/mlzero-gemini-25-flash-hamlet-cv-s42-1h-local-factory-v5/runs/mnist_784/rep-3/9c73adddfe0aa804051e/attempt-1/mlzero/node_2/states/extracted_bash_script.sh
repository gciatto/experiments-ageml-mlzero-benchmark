#!/bin/bash
set -euo pipefail

# Define the target conda environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/mnist_784/rep-3/9c73adddfe0aa804051e/attempt-1/mlzero/node_2/conda_env"

# Define paths to the requirements files
REQ_FILE_MM="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"
REQ_FILE_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"

echo "--- Starting Conda environment setup for AutoGluon MultiModal ---"

# 1. Check if conda is installed and available in PATH
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not found in PATH. Please ensure Conda is installed." >&2
    exit 1
fi

# 2. Remove the existing environment if it exists to ensure a clean installation
if [ -d "$ENV_PATH" ]; then
    echo "Removing existing conda environment at '$ENV_PATH'..."
    conda env remove -p "$ENV_PATH" -y
fi

# 3. Create a new conda environment with Python 3.11
echo "Creating new conda environment at '$ENV_PATH' with Python 3.11..."
conda create -p "$ENV_PATH" python=3.11 -y

# 4. Activate the newly created environment
echo "Activating the conda environment..."
# Source conda.sh to enable 'conda activate' in the script
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Error: conda.sh not found. Cannot activate environment. Ensure Conda is properly installed." >&2
    exit 1
fi
conda activate "$ENV_PATH"

# 5. Install 'uv' into the activated environment
echo "Installing 'uv' package manager into the environment..."
pip install uv

# 6. Install core libraries: numpy, pandas, scikit-learn
# These are explicitly required by the factory module and the prompt.
echo "Installing numpy, pandas, and scikit-learn into the environment using uv..."
uv pip install numpy pandas scikit-learn

# 7. Install required packages from the specified requirements files using uv
# This includes autogluon.multimodal and its dependencies.
echo "Installing packages from requirements files using uv..."
uv pip install \
    -r "$REQ_FILE_MM" \
    --prerelease=allow \
    -r "$REQ_FILE_COMMON"

echo "--- Conda environment setup complete ---"