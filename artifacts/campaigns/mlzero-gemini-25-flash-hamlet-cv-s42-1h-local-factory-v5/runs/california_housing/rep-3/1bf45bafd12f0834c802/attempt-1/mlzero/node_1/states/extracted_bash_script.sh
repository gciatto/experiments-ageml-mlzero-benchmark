#!/bin/bash
set -euxo pipefail

# Define environment and requirements paths
CONDA_ENV_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/california_housing/rep-3/1bf45bafd12f0834c802/attempt-1/mlzero/node_1/conda_env"
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# Create the conda environment with Python 3.11
echo "Creating conda environment at $CONDA_ENV_DIR with Python 3.11..."
conda create --prefix "$CONDA_ENV_DIR" python=3.11 --yes

# Activate the newly created environment
echo "Activating conda environment..."
source activate "$CONDA_ENV_DIR"

# Install uv into the activated environment
echo "Installing uv..."
pip install uv

# Install required packages using uv, including numpy, pandas, scikit-learn,
# and packages from the specified requirements files.
# --prerelease=allow is used as specified.
echo "Installing packages using uv pip install..."
uv pip install \
    -r "$MULTIMODAL_REQUIREMENTS_FILE" \
    --prerelease=allow \
    -r "$COMMON_REQUIREMENTS_FILE" \
    numpy pandas scikit-learn

echo "Conda environment setup complete."