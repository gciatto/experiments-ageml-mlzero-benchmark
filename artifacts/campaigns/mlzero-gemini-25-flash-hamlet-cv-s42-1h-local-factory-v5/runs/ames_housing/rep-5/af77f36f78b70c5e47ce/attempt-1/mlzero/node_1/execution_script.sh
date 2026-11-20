#!/bin/bash
set -euo pipefail

# Define the path for the Conda environment
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-5/af77f36f78b70c5e47ce/attempt-1/mlzero/node_1/conda_env"

# Define paths to requirements files
REQUIREMENTS_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_ML="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

# Remove existing environment if it exists to ensure a clean setup
if [ -d "$CONDA_ENV_PATH" ]; then
    rm -rf "$CONDA_ENV_PATH"
fi

# Create the Conda environment with Python 3.11
# Using --yes for non-interactive creation.
conda create -p "$CONDA_ENV_PATH" python=3.11 --yes

# Install uv into the new environment using conda run --prefix for robustness
# This ensures pip operates within the correct environment.
conda run --prefix "$CONDA_ENV_PATH" pip install uv

# Install all required packages using uv pip install
# This includes packages from both requirements.txt files, plus numpy, pandas, scikit-learn, and autogluon.tabular.
# The --prerelease=allow flag is applied as specified for the machine learning requirements.
conda run --prefix "$CONDA_ENV_PATH" uv pip install \
    -r "$REQUIREMENTS_ML" \
    --prerelease=allow \
    -r "$REQUIREMENTS_COMMON" \
    numpy pandas scikit-learn autogluon.tabular