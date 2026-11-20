#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
# set -u # Treat unset variables as an error. Temporarily disabled around conda activate if needed.

# Define the target conda environment path for node_12
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-3/5a9401ba768907dcd35e/attempt-1/mlzero/node_12/conda_env"

# Define paths to the requirements files
REQ_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQ_AUTOGLUON_TABULAR="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "Starting minimal BASH INSTALLATION script for AutoGluonTabularWrapper (node_12)..."

# 1. Create the conda environment with Python 3.11
echo "Creating conda environment at '$ENV_PATH' with Python 3.11..."
# conda create -p will create parent directories if they don't exist.
# Using --yes for non-interactive creation.
conda create -p "$ENV_PATH" python=3.11 --yes

# 2. Initialize conda and activate the environment
echo "Initializing conda and activating environment '$ENV_PATH'..."

# Find the base conda installation path. Use a common fallback if `conda info` fails.
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/opt/conda")

# Source the conda.sh script to make `conda` commands available in the current shell session.
if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "Error: conda.sh not found at $CONDA_BASE/etc/profile.d/conda.sh. Is conda installed correctly?"
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Deactivate any existing environment to ensure a clean state before activating the target one.
# This prevents issues if the script is run in an already active environment.
conda deactivate 2>/dev/null || true

# Activate the newly created environment.
# Temporarily disable 'set -u' if it causes issues with CONDA_SHLVL, though sourcing profile.d usually handles it.
# set +u # Disable unbound variable check
conda activate "$ENV_PATH"
# set -u # Re-enable unbound variable check (if it was enabled initially)

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment at '$ENV_PATH'."
    exit 1
fi
echo "Environment '$ENV_PATH' activated."

# 3. Install 'uv' into the activated environment
echo "Installing 'uv' into the activated environment..."
# 'pip' now refers to the pip within the activated environment.
pip install uv

# 4. Install required packages using 'uv'
echo "Installing packages from requirements files and additional libraries (numpy, pandas, scikit-learn) using uv..."
# 'uv' now refers to the uv within the activated environment.
# uv's default behavior ensures only specified packages and their dependencies are installed,
# and it avoids upgrading/reinstalling if versions are already satisfied.
uv pip install \
    -r "$REQ_AUTOGLUON_TABULAR" \
    --prerelease=allow \
    -r "$REQ_COMMON" \
    numpy pandas scikit-learn

echo "Conda environment setup complete at '$ENV_PATH'."
echo "The environment is prepared with its own interpreter/pip for subsequent operations."