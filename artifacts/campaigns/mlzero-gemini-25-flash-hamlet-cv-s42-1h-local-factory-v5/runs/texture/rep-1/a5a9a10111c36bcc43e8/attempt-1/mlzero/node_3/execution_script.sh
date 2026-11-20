#!/bin/bash
set -euo pipefail

# Define the target conda environment path for node_3
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/texture/rep-1/a5a9a10111c36bcc43e8/attempt-1/mlzero/node_3/conda_env"

# Define paths to the requirements files
COMMON_REQS_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AUTOGLUON_TABULAR_REQS_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

# Ensure conda is initialized for script execution
# This makes 'conda activate' and other conda commands available
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Error: conda.sh not found. Please ensure Conda is installed and initialized in your shell." >&2
    exit 1
fi

# Deactivate any active conda environment to ensure a clean state
conda deactivate &> /dev/null || true

# Create the conda environment with Python 3.11
echo "Creating conda environment at $CONDA_ENV_PATH with Python 3.11..."
conda create --prefix "$CONDA_ENV_PATH" python=3.11 --yes

# Activate the newly created environment
echo "Activating conda environment: $CONDA_ENV_PATH"
conda activate "$CONDA_ENV_PATH"

# Install uv using pip (from the activated environment)
echo "Installing uv..."
pip install uv

# Install required packages using uv pip
# This includes numpy, pandas, scikit-learn, and packages from the specified requirements files.
# uv ensures only specified packages and their dependencies are installed,
# and avoids unnecessary upgrades/reinstalls.
# The order of -r flags and --prerelease=allow is as specified in the prompt.
echo "Installing numpy, pandas, scikit-learn, and requirements from specified files using uv pip..."
uv pip install \
    numpy \
    pandas \
    scikit-learn \
    -r "$AUTOGLUON_TABULAR_REQS_PATH" \
    --prerelease=allow \
    -r "$COMMON_REQS_PATH"

echo "Conda environment setup complete at $CONDA_ENV_PATH"