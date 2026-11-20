#!/bin/bash
set -euo pipefail

# Define environment path and requirements file paths
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/har/rep-4/e3e33218f51e76d801d1/attempt-1/mlzero/node_17/conda_env"
REQUIREMENTS_COMMON_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_ML_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

echo "--- Starting Conda environment setup ---"

# 1. Ensure the parent directory for the conda environment exists
mkdir -p "$(dirname "$CONDA_ENV_PATH")"
echo "Created parent directory for conda environment: $(dirname "$CONDA_ENV_PATH")"

# 2. Create the conda environment with Python 3.11 and pip
echo "Creating conda environment at $CONDA_ENV_PATH with Python 3.11..."
conda create -p "$CONDA_ENV_PATH" python=3.11 pip -y

# 3. Install core scientific packages (numpy, pandas, scikit-learn) directly via conda.
# This addresses previous errors related to numpy not being found and ensures robust installation.
echo "Installing numpy, pandas, scikit-learn into the new conda environment via conda install..."
conda install -p "$CONDA_ENV_PATH" numpy pandas scikit-learn -y

# 4. Install uv using pip within the specified conda environment.
# Using 'conda run -p' to execute commands directly in the environment without explicit activation,
# addressing previous activation script issues.
echo "Installing uv into the conda environment..."
conda run -p "$CONDA_ENV_PATH" pip install uv

# 5. Install required packages from requirements.txt files using uv within the environment.
echo "Installing packages from requirements files using uv pip install..."
conda run -p "$CONDA_ENV_PATH" uv pip install -r "$REQUIREMENTS_ML_PATH" --prerelease=allow -r "$REQUIREMENTS_COMMON_PATH"

# 6. Install autogluon.tabular (the selected ML library from the factory module) using uv within the environment.
echo "Installing autogluon.tabular using uv pip install..."
conda run -p "$CONDA_ENV_PATH" uv pip install autogluon.tabular

echo "--- Conda environment setup complete ---"