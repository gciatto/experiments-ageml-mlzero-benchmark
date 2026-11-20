#!/bin/bash
set -euo pipefail

# Define paths for the conda environment and requirements files
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/har/rep-3/053ecb30e95bae230b78/attempt-1/mlzero/node_2/conda_env"
COMMON_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

echo "Starting minimal BASH INSTALLATION script for AutoGluon MultiModal."

# 1. Ensure conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda command not found. Please ensure conda is installed and initialized in your shell."
    exit 1
fi

# 2. Create the conda environment with Python 3.11
echo "Creating conda environment at '$CONDA_ENV_PATH' with Python 3.11..."
# Use -p for a specific path, -y for non-interactive mode
conda create -y -p "$CONDA_ENV_PATH" python=3.11

# 3. Activate the environment
echo "Activating conda environment..."
# Source the activate script to modify the current shell's environment
source "$CONDA_ENV_PATH/bin/activate"

# Verify environment activation
if [[ "$CONDA_PREFIX" != "$CONDA_ENV_PATH" ]]; then
    echo "Error: Failed to activate conda environment. CONDA_PREFIX is '$CONDA_PREFIX', expected '$CONDA_ENV_PATH'."
    exit 1
fi
echo "Conda environment successfully activated: $CONDA_PREFIX"

# 4. Install uv into the activated environment
echo "Installing uv (a fast Python package installer) into the environment..."
python -m pip install uv

# 5. Install core packages (numpy, pandas, scikit-learn) using uv
echo "Installing numpy, pandas, and scikit-learn using uv..."
uv pip install numpy pandas scikit-learn

# 6. Install required packages from the specified requirements files using uv
echo "Installing packages from requirements files using uv..."
echo "  - $MULTIMODAL_REQ_PATH"
echo "  - $COMMON_REQ_PATH"
uv pip install \
    -r "$MULTIMODAL_REQ_PATH" \
    --prerelease=allow \
    -r "$COMMON_REQ_PATH"

echo "All specified packages have been installed into the conda environment."
echo "Installation complete."

# Deactivate the environment (optional, as the script will exit)
# conda deactivate