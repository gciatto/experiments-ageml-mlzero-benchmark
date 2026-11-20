#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
set -o pipefail # Also exit if a command in a pipeline fails

# --- Configuration Variables ---
# The full path to the target Conda environment
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/auto_mpg/rep-2/36b7694a16463792f153/attempt-1/mlzero/node_0/conda_env"
# Path to the common requirements file
COMMON_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
# Path to the AutoGluon.tabular specific requirements file
AUTOGLUON_TABULAR_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"
# Desired Python version for the environment
PYTHON_VERSION="3.11"

echo "--- Starting Conda Environment Setup ---"
echo "Target environment path: $ENV_PATH"
echo "Python version: $PYTHON_VERSION"

# --- Step 1: Initialize Conda for the current shell session ---
# This is crucial for 'conda activate' to work correctly within a script.
# It assumes the 'conda' command is available in the system's PATH.
echo "Initializing Conda shell hook..."
eval "$(conda shell hook)" || { echo "Error: Failed to initialize Conda. Please ensure Conda is installed and its base environment is accessible."; exit 1; }

# --- Step 2: Create or update the Conda environment ---
# If the environment already exists, conda will attempt to update it to match
# the specified Python version. If it doesn't exist, it will be created.
echo "Creating/Updating Conda environment at '$ENV_PATH' with Python $PYTHON_VERSION..."
conda create --prefix "$ENV_PATH" python="$PYTHON_VERSION" --yes || { echo "Error: Failed to create or update Conda environment."; exit 1; }

# --- Step 3: Activate the newly created/updated environment ---
echo "Activating Conda environment: '$ENV_PATH'"
conda activate "$ENV_PATH" || { echo "Error: Failed to activate Conda environment."; exit 1; }

# --- Step 4: Install uv (ultra-fast Python package installer) ---
# uv is used for efficient and precise package installation.
echo "Installing uv into the environment..."
pip install uv || { echo "Error: Failed to install uv."; exit 1; }

# --- Step 5: Install required packages using uv ---
# This command installs packages from both specified requirements files,
# along with numpy, pandas, and scikit-learn, as requested.
# '--prerelease=allow' is included as per the instructions.
# uv ensures exact package versions and their dependencies are installed
# without unnecessary upgrades or reinstallation if versions match.
echo "Installing required packages (including numpy, pandas, scikit-learn) using uv pip install..."
uv pip install \
    -r "$AUTOGLUON_TABULAR_REQ_PATH" \
    --prerelease=allow \
    -r "$COMMON_REQ_PATH" \
    numpy pandas scikit-learn \
    || { echo "Error: Failed to install packages using uv."; exit 1; }

# --- Step 6: Optional: Verify key installations ---
# These checks confirm that the Python interpreter and core libraries are
# correctly installed and importable within the activated environment.
echo "--- Verifying key installations ---"
python -c "import sys; print(f'Python version: {sys.version}')" || { echo "Verification Error: Python not found or failed to execute."; exit 1; }
python -c "import numpy; print(f'Numpy version: {numpy.__version__}')" || { echo "Verification Error: Numpy not found or failed to import."; exit 1; }
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')" || { echo "Verification Error: Pandas not found or failed to import."; exit 1; }
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')" || { echo "Verification Error: Scikit-learn not found or failed to import."; exit 1; }
python -c "import autogluon.tabular; print(f'AutoGluon.tabular version: {autogluon.tabular.__version__}')" || { echo "Verification Error: AutoGluon.tabular not found or failed to import."; exit 1; }

echo "--- Conda Environment Setup Complete Successfully ---"