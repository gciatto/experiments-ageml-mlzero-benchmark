#!/bin/bash
set -e

# Define environment and file paths
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/california_housing/rep-5/8ea4d1996b754081a6d4/attempt-1/mlzero/node_2/conda_env"
PYTHON_VERSION="3.11"
COMMON_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

echo "--- Starting Conda environment setup ---"

# 1. Create the Conda environment with the specified Python version
echo "Creating conda environment at $CONDA_ENV_PATH with Python $PYTHON_VERSION..."
conda create --prefix "$CONDA_ENV_PATH" python="$PYTHON_VERSION" --yes
echo "Conda environment created."

# 2. Activate the newly created environment
echo "Activating conda environment..."
source activate "$CONDA_ENV_PATH"
echo "Conda environment activated."

# 3. Install uv using the environment's pip
echo "Installing uv into the environment..."
pip install uv
echo "uv installed."

# 4. Install required packages from requirements files using uv
# The --prerelease=allow flag is included as specified.
# uv handles exact package installation and avoids unnecessary upgrades/reinstalls by default.
echo "Installing packages from requirements files using uv..."
uv pip install -r "$MULTIMODAL_REQ_PATH" --prerelease=allow -r "$COMMON_REQ_PATH"
echo "Packages from requirements files installed."

# 5. Install numpy, pandas, and scikit-learn using uv
echo "Installing numpy, pandas, and scikit-learn into the environment using uv..."
uv pip install numpy pandas scikit-learn
echo "numpy, pandas, and scikit-learn installed."

echo "--- Conda environment setup complete ---"