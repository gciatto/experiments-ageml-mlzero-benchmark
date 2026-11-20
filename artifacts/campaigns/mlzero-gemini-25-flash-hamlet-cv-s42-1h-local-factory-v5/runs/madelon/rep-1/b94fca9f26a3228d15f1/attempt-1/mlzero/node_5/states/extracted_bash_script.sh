#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
set -o pipefail # Exit if any command in a pipeline fails.

# Define the full path for the conda environment as specified for node_5
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/madelon/rep-1/b94fca9f26a3228d15f1/attempt-1/mlzero/node_5/conda_env"

# Define paths to requirements files
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AUTOGLUON_TABULAR_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "--- Starting Conda environment setup for node_5 ---"

# Ensure the parent directory for the conda environment exists
mkdir -p "$(dirname "$CONDA_ENV_PATH")"

# 1. Create the conda environment with Python 3.11
echo "Creating conda environment at $CONDA_ENV_PATH with Python 3.11..."
conda create --prefix "$CONDA_ENV_PATH" python=3.11 --yes

# 2. Activate the environment
echo "Activating conda environment..."
# Initialize conda for the current shell session. This approach was part of the
# 'Previous installation attempt' which was presented after a fix for environment
# activation issues, ensuring 'conda activate' works correctly in scripts.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

# 3. Install uv
echo "Installing uv..."
# Use the pip from the activated environment to install uv
pip install uv

# 4. Install required packages using uv
echo "Installing required packages from $AUTOGLUON_TABULAR_REQUIREMENTS_FILE and $COMMON_REQUIREMENTS_FILE using uv..."
# uv automatically handles exact versions and dependencies, and avoids unnecessary
# upgrades or reinstalls if packages are already at the correct version.
uv pip install -r "$AUTOGLUON_TABULAR_REQUIREMENTS_FILE" --prerelease=allow -r "$COMMON_REQUIREMENTS_FILE"

echo "--- Conda environment setup complete for node_5 ---"