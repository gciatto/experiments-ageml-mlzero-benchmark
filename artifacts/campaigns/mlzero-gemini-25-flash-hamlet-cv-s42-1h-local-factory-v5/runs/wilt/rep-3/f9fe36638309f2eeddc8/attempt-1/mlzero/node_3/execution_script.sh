#!/bin/bash
set -euo pipefail

# Define paths
# CONDA_ENV_PATH is set to node_3 as specified in the prompt.
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/wilt/rep-3/f9fe36638309f2eeddc8/attempt-1/mlzero/node_3/conda_env"
COMMON_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

echo "--- Starting Conda environment setup ---"

# Ensure the parent directory for the conda environment exists
mkdir -p "$(dirname "$CONDA_ENV_PATH")" || { echo "Failed to create parent directory for conda environment."; exit 1; }

# 1. Create Conda environment with Python 3.11
echo "Creating conda environment at $CONDA_ENV_PATH with Python 3.11..."
conda create -p "$CONDA_ENV_PATH" python=3.11 -y || { echo "Failed to create conda environment."; exit 1; }

# 2. Activate the environment
echo "Activating conda environment..."
# Initialize conda for the current shell session if not already done
# This line is crucial for 'conda activate' to work in a non-interactive script
source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "Failed to initialize conda. Is conda installed and in PATH?"; exit 1; }
conda activate "$CONDA_ENV_PATH" || { echo "Failed to activate conda environment."; exit 1; }

# 3. Install uv
echo "Installing uv into the environment..."
pip install uv || { echo "Failed to install uv."; exit 1; }

# 4. Install required packages using uv
echo "Installing packages from requirements files and additional libraries using uv..."
# uv will handle dependency resolution, respect version pins in requirements files,
# and avoid unnecessary upgrades/reinstalls if packages are already at the correct version.
uv pip install \
    -r "$MULTIMODAL_REQ_PATH" \
    --prerelease=allow \
    -r "$COMMON_REQ_PATH" \
    numpy pandas scikit-learn || { echo "Failed to install required packages with uv."; exit 1; }

echo "--- Conda environment setup complete ---"