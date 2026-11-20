#!/bin/bash
set -e

# Define the target conda environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-3/5a9401ba768907dcd35e/attempt-1/mlzero/node_0/conda_env"

# Define paths to the requirements files
REQ_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_AUTOGLUON_TABULAR="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "Starting minimal BASH INSTALLATION script for AutoGluonTabularWrapper..."

# 1. Create the conda environment with Python 3.11
echo "Creating conda environment at '$ENV_PATH' with Python 3.11..."
conda create -p "$ENV_PATH" python=3.11 -y

# 2. Install 'uv' into the newly created environment
# We use the full path to the environment's pip to ensure installation into the correct environment.
echo "Installing 'uv' into the environment..."
"$ENV_PATH"/bin/pip install uv

# 3. Install required packages using 'uv'
# This command installs packages from both requirements files and explicitly adds numpy, pandas, and scikit-learn.
# 'uv' will resolve all dependencies, ensuring only specified packages and their dependencies are installed.
# The '--prerelease=allow' flag is included as specified.
# Using the full path to 'uv' ensures it operates within the target environment.
echo "Installing packages from requirements files and additional libraries (numpy, pandas, scikit-learn) using uv..."
"$ENV_PATH"/bin/uv pip install \
    -r "$REQ_AUTOGLUON_TABULAR" \
    --prerelease=allow \
    -r "$REQ_COMMON" \
    numpy pandas scikit-learn

echo "Conda environment setup complete at '$ENV_PATH'."
echo "The environment is prepared with its own interpreter/pip for subsequent operations."