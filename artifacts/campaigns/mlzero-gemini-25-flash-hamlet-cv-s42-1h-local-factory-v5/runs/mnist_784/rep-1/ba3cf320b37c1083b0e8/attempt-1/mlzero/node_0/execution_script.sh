#!/bin/bash
set -euo pipefail

# Define the target conda environment directory
ENV_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/mnist_784/rep-1/ba3cf320b37c1083b0e8/attempt-1/mlzero/node_0/conda_env"

# Define paths to requirements files
COMMON_REQUIREMENTS="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
TABULAR_REQUIREMENTS="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "--- Starting Conda environment setup ---"

# 1. Create the conda environment with Python 3.11
echo "Creating conda environment at '$ENV_DIR' with Python 3.11..."
conda create -p "$ENV_DIR" python=3.11 -y

# 2. Activate the environment
echo "Activating conda environment..."
# Ensure conda.sh is sourced to make 'conda activate' available
# This assumes conda is already installed and initialized in the shell where this script runs.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Install uv
echo "Installing uv into the environment..."
pip install uv

# 4. Install required packages using uv pip install
echo "Installing numpy, pandas, scikit-learn, and factory module requirements using uv..."
uv pip install \
    numpy pandas scikit-learn \
    -r "$TABULAR_REQUIREMENTS" \
    --prerelease=allow \
    -r "$COMMON_REQUIREMENTS"

echo "--- Conda environment setup complete at '$ENV_DIR' ---"