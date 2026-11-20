#!/bin/bash
# This script sets up a minimal Conda environment for the factory module.
# It creates a new Conda environment, installs uv, and then installs
# all necessary Python packages using uv.

# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error.
# Exit if any command in a pipeline fails.
set -euo pipefail

# Define the full path for the Conda environment
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/wilt/rep-2/f3676860d054e0e4d222/attempt-1/mlzero/node_2/conda_env"

# Define paths to the requirements files
COMMON_REQS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

echo "--- Starting Conda environment setup ---"

# 1. Create the Conda environment with Python 3.11
echo "Creating Conda environment at '$ENV_PATH' with Python 3.11..."
# The -y flag automatically answers yes to prompts.
conda create -p "$ENV_PATH" python=3.11 -y

# 2. Activate the newly created environment
echo "Activating the environment..."
# Use 'source' to activate the environment in the current shell context.
source "$ENV_PATH/bin/activate"

# 3. Install uv into the activated environment
echo "Installing 'uv' into the environment..."
pip install uv

# 4. Install required packages from the specified requirements files using uv
echo "Installing packages from '$ML_REQS_FILE' and '$COMMON_REQS_FILE' using uv..."
# --prerelease=allow is specified in the prompt.
uv pip install -r "$ML_REQS_FILE" --prerelease=allow -r "$COMMON_REQS_FILE"

# 5. Install additional packages: numpy, pandas, and scikit-learn
# These are explicitly requested in the prompt. The factory module also uses autogluon.tabular,
# which is expected to be covered by the requirements files.
echo "Installing additional packages: numpy, pandas, scikit-learn using uv..."
uv pip install numpy pandas scikit-learn

echo "--- Conda environment setup complete ---"

# Deactivate the environment to return to the base shell state.
# This is good practice for scripts, though not strictly necessary if the script exits immediately.
conda deactivate