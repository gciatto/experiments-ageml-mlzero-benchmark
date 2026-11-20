#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the target conda environment path
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/madelon/rep-2/85943d47ce5c45c5561b/attempt-1/mlzero/node_2/conda_env"

# Define paths to requirements files
REQUIREMENTS_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_MULTIMODAL="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

echo "--- Starting Conda Environment Setup ---"

# 1. Create a new conda environment with Python 3.11
echo "Creating conda environment at $CONDA_ENV_PATH with Python 3.11..."
# Use --yes for non-interactive creation, --quiet to suppress excessive output.
# --prefix is used instead of -p for clarity, though -p works.
conda create --prefix "$CONDA_ENV_PATH" python=3.11 --yes --quiet

# 2. Activate the environment
echo "Activating conda environment..."
# Source conda.sh to make 'conda' command available in non-login shells
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

# Verify activation
echo "Active Python interpreter: $(which python)"
echo "Active pip: $(which pip)"

# 3. Install uv using the environment's pip
echo "Installing uv into the environment..."
pip install uv

# 4. Install numpy, pandas, scikit-learn using uv
echo "Installing numpy, pandas, scikit-learn using uv..."
uv pip install numpy pandas scikit-learn

# 5. Install required packages from requirements files using uv
echo "Installing packages from requirements files using uv..."
# The --prerelease=allow flag is specifically for autogluon.multimodal requirements.
# uv will handle dependencies, install exact versions if specified, and avoid
# reinstallation if packages are already present and satisfy requirements.
uv pip install \
    -r "$REQUIREMENTS_MULTIMODAL" --prerelease=allow \
    -r "$REQUIREMENTS_COMMON"

echo "--- Conda Environment Setup Complete ---"
echo "Environment is ready at: $CONDA_ENV_PATH"