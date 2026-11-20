#!/bin/bash
set -e

# Define the target conda environment path
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/wilt/rep-5/0cf989f159b35f94ca0e/attempt-1/mlzero/node_3/conda_env"

# Define paths to the requirements files
COMMON_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AUTOGLUON_TABULAR_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "--- Creating conda environment at $CONDA_ENV_PATH ---"
# Create the conda environment with Python 3.11
# -y for non-interactive mode
conda create -p "$CONDA_ENV_PATH" python=3.11 -y

echo "--- Activating environment and installing uv ---"
# Install uv using the environment's pip.
# 'conda run -p' implicitly activates the environment for the command.
conda run -p "$CONDA_ENV_PATH" pip install uv

echo "--- Installing core ML libraries: numpy, pandas, scikit-learn ---"
# Install numpy, pandas, and scikit-learn using uv pip.
# These are explicitly requested for the node's conda_env.
conda run -p "$CONDA_ENV_PATH" uv pip install numpy pandas scikit-learn

echo "--- Installing required packages from requirements files using uv ---"
# Install packages from both requirements files using uv pip.
# The --prerelease=allow flag is included as specified.
# uv efficiently handles exact package installation and avoids unnecessary upgrades/reinstalls.
conda run -p "$CONDA_ENV_PATH" uv pip install \
    -r "$AUTOGLUON_TABULAR_REQ_PATH" \
    --prerelease=allow \
    -r "$COMMON_REQ_PATH"

echo "--- Conda environment setup complete ---"