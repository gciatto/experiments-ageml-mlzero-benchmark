#!/bin/bash
set -euo pipefail

# Define environment path and requirements files
ENV_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/adult/rep-5/e936464f2167cf84cde8/attempt-1/mlzero/node_1/conda_env"
PYTHON_VERSION="3.11"
COMMON_REQUIREMENTS="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQUIREMENTS="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

echo "Starting minimal BASH INSTALLATION script for AutoGluon MultiModal..."

# 1. Remove existing environment if it exists to ensure a clean slate
if [ -d "$ENV_DIR" ]; then
    echo "Existing conda environment found at $ENV_DIR. Removing..."
    rm -rf "$ENV_DIR"
fi

# 2. Create the conda environment with the specified Python version
echo "Creating conda environment at $ENV_DIR with Python $PYTHON_VERSION..."
conda create -p "$ENV_DIR" python="$PYTHON_VERSION" -y

# 3. Activate the newly created environment
# Ensure conda is initialized in the current shell session
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "Activating conda environment: $ENV_DIR"
conda activate "$ENV_DIR"

# 4. Install uv (a fast package installer)
echo "Installing uv into the environment..."
pip install uv

# 5. Install core packages (numpy, pandas, scikit-learn) and required packages
# from the specified requirements files using uv.
# The --prerelease=allow flag is included as per instructions.
# uv will handle dependencies and respect exact versions specified in requirements.txt.
echo "Installing numpy, pandas, scikit-learn, and requirements from files using uv..."
uv pip install numpy pandas scikit-learn \
    -r "$MULTIMODAL_REQUIREMENTS" \
    --prerelease=allow \
    -r "$COMMON_REQUIREMENTS"

echo "Installation complete for environment: $ENV_DIR"
echo "Conda environment is ready for use."