#!/bin/bash
set -euo pipefail

# Define environment path and requirements file paths
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/har/rep-4/e3e33218f51e76d801d1/attempt-1/mlzero/node_7/conda_env"
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
# This addresses previous errors related to numpy not being found and ensures robust installation
# for these fundamental libraries, as suggested by previous error summaries.
echo "Installing numpy, pandas, scikit-learn into the new conda environment via conda install..."
conda install -p "$CONDA_ENV_PATH" numpy pandas scikit-learn -y

# 4. Activate the environment
# Note: 'conda activate' modifies the shell environment. For a script,
# we source the activate script directly to apply changes to the current shell.
echo "Activating conda environment..."
source "$CONDA_ENV_PATH/bin/activate"
echo "Conda environment activated."

# 5. Install uv
echo "Installing uv into the activated environment..."
pip install uv

# 6. Install required packages from requirements.txt files using uv
echo "Installing packages from requirements files using uv pip install..."
uv pip install -r "$REQUIREMENTS_ML_PATH" --prerelease=allow -r "$REQUIREMENTS_COMMON_PATH"

# 7. Install autogluon.tabular (the selected ML library from the factory module)
# numpy, pandas, scikit-learn are already installed via conda.
echo "Installing autogluon.tabular using uv pip install..."
uv pip install autogluon.tabular

echo "--- Conda environment setup complete ---"

# Deactivate the environment (optional, as script exit will naturally deactivate it)
# conda deactivate