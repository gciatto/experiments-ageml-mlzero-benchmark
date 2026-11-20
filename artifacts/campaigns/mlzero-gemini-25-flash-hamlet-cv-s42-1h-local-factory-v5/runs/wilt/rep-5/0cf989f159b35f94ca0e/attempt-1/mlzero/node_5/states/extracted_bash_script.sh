#!/bin/bash
set -euo pipefail

# Define the target Conda environment directory for node_5
ENV_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/wilt/rep-5/0cf989f159b35f94ca0e/attempt-1/mlzero/node_5/conda_env"

# Define paths to the requirements files
REQ_COMMON_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_MULTIMODAL_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

echo "--- Starting minimal BASH INSTALLATION script ---"
echo "Target Conda environment: $ENV_DIR"

# 1. Create the Conda environment with Python 3.11
echo "Creating Conda environment with Python 3.11 at $ENV_DIR..."
# If the environment directory already exists, remove it to ensure a clean install.
# This ensures idempotency and prevents issues if a previous attempt left a partial environment.
if [ -d "$ENV_DIR" ]; then
    echo "Existing environment found at $ENV_DIR. Removing for a clean installation."
    rm -rf "$ENV_DIR"
fi
conda create -p "$ENV_DIR" python=3.11 -y

# 2. Activate the environment
echo "Activating the newly created Conda environment..."
# Initialize Conda for the current shell session if not already done
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found. Ensure Conda is installed and initialized correctly." >&2
    exit 1
fi
conda activate "$ENV_DIR"
echo "Conda environment activated."

# 3. Install uv using pip within the activated environment
echo "Installing 'uv' package manager..."
pip install uv

# 4. Install core scientific packages: numpy, pandas, scikit-learn using uv pip
echo "Installing numpy, pandas, and scikit-learn using uv pip..."
uv pip install numpy pandas scikit-learn

# 5. Install required packages from the specified requirements files using uv
# The order of requirements files and --prerelease=allow is as specified in the prompt.
echo "Installing packages from requirements files using uv pip..."
uv pip install -r "$REQ_MULTIMODAL_PATH" --prerelease=allow -r "$REQ_COMMON_PATH"

echo "--- Conda environment setup complete ---"
echo "Environment is ready at $ENV_DIR"