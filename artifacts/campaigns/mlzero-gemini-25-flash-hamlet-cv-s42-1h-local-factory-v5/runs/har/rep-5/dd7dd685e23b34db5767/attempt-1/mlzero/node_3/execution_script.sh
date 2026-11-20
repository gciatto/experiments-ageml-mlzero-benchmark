#!/bin/bash
set -euo pipefail

# Define environment path and requirements file paths
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/har/rep-5/dd7dd685e23b34db5767/attempt-1/mlzero/node_3/conda_env"
REQUIREMENTS_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_MULTIMODAL="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# Ensure conda is initialized for the shell session
source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "Error: conda not found or not initialized. Please ensure conda is in PATH."; exit 1; }

# Create the conda environment with Python 3.11
conda create -y -p "$CONDA_ENV_PATH" python=3.11

# Activate the newly created environment
conda activate "$CONDA_ENV_PATH"

# Get the path to the pip executable within the activated environment
PIP_BIN="$(which pip)"
if [ -z "$PIP_BIN" ]; then
    echo "Error: pip not found in the activated environment. Installation cannot proceed."
    exit 1
fi

# Install uv using the environment's pip
"$PIP_BIN" install uv

# Get the path to the uv executable within the activated environment
UV_BIN="$(which uv)"
if [ -z "$UV_BIN" ]; then
    echo "Error: uv not found after installation. Installation cannot proceed."
    exit 1
fi

# Install core packages (numpy, pandas, scikit-learn) and packages from requirements files using uv.
# The order of requirements files is crucial as specified in the prompt.
"$UV_BIN" pip install \
    numpy pandas scikit-learn \
    -r "$REQUIREMENTS_MULTIMODAL" \
    --prerelease=allow \
    -r "$REQUIREMENTS_COMMON"