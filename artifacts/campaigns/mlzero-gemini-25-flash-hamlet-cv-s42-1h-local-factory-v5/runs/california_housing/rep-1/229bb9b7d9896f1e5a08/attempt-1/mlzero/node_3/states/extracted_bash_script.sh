#!/bin/bash
set -e

# Define environment path and requirements file paths
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/california_housing/rep-1/229bb9b7d9896f1e5a08/attempt-1/mlzero/node_3/conda_env"
COMMON_REQ="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AG_TABULAR_REQ="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

# --- 1. Check for Conda installation ---
if ! command -v conda &> /dev/null
then
    echo "Error: conda command not found. Please ensure Anaconda or Miniconda is installed and in your PATH."
    exit 1
fi

# --- 2. Initialize conda for the current shell session ---
# This is crucial for `conda activate` to work in non-interactive scripts
echo "Initializing conda for the current shell session..."
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- 3. Create or activate the conda environment ---
if [ -d "$ENV_PATH" ]; then
    echo "Conda environment already exists at $ENV_PATH. Activating it."
    conda activate "$ENV_PATH"
else
    echo "Creating conda environment at $ENV_PATH with Python 3.11..."
    conda create --prefix "$ENV_PATH" python=3.11 -y
    echo "Activating the newly created environment..."
    conda activate "$ENV_PATH"
fi

# --- 4. Install uv if not already present in the activated environment ---
# Check if uv is available in the current PATH (which should include the conda env's bin)
if ! command -v uv &> /dev/null
then
    echo "Installing uv..."
    pip install uv
else
    echo "uv is already installed."
fi

# --- 5. Install required packages using uv ---
echo "Installing/updating required packages from requirements files using uv..."
# The command is specified exactly in the prompt.
# uv will respect existing versions and only install/upgrade if necessary,
# addressing the "Do NOT upgrade or reinstall autogluon.tabular if it's already at the correct version" constraint.
# This command will also ensure numpy, pandas, and scikit-learn are installed as dependencies
# of autogluon.tabular or if explicitly listed in the requirements files.
uv pip install -r "$AG_TABULAR_REQ" --prerelease=allow -r "$COMMON_REQ"

echo "Conda environment setup complete at $ENV_PATH."