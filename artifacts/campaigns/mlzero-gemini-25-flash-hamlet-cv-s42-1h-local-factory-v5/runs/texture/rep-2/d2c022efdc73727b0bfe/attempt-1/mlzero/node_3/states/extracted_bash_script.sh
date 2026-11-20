#!/bin/bash
set -euo pipefail

# Define environment and requirements paths
# The full path to the conda environment directory for node_3
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/texture/rep-2/d2c022efdc73727b0bfe/attempt-1/mlzero/node_3/conda_env"
# Path to the common requirements file
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
# Path to the autogluon.tabular specific requirements file
TABULAR_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "Starting Conda environment setup for node_3..."
echo "Target environment path: ${CONDA_ENV_PATH}"

# Ensure conda is initialized for the current shell session.
# This is necessary for 'conda activate' to function correctly within a script.
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Error: Conda initialization script not found. Please ensure Conda is installed and configured." >&2
    exit 1
fi

# Create or update the conda environment with Python 3.11.
# If the environment already exists with Python 3.11, this command will typically do nothing or quickly confirm.
# If it exists with a different Python version, it will attempt to update it to 3.11.
echo "Creating/updating conda environment at ${CONDA_ENV_PATH} with Python 3.11..."
conda create -p "${CONDA_ENV_PATH}" python=3.11 -y

# Activate the newly created or updated environment.
echo "Activating conda environment: ${CONDA_ENV_PATH}"
conda activate "${CONDA_ENV_PATH}"

# Install uv, a fast package installer, into the activated environment.
echo "Installing uv into the environment..."
pip install uv

# Install required packages using uv.
# This includes numpy, pandas, scikit-learn (explicitly requested)
# and all packages specified in the two requirements files.
# The --prerelease=allow flag is used as specified.
# uv's default behavior respects existing package versions if they satisfy requirements,
# aligning with the "Do NOT upgrade or reinstall autogluon.tabular if it's already at the correct version" constraint.
echo "Installing core packages (numpy, pandas, scikit-learn) and requirements from files using uv..."
uv pip install \
    numpy pandas scikit-learn \
    -r "${TABULAR_REQUIREMENTS_FILE}" \
    --prerelease=allow \
    -r "${COMMON_REQUIREMENTS_FILE}"

echo "Conda environment setup complete for node_3."