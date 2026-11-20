#!/bin/bash
set -euo pipefail

# --- Configuration ---
ENV_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/auto_mpg/rep-4/6a956d8fefbd8620bdfc/attempt-1/mlzero/node_2/conda_env"
COMMON_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# --- Conda Initialization (Robust for non-interactive scripts) ---
# Check if conda is available in PATH
if ! command -v conda &> /dev/null; then
    echo "Error: 'conda' command not found. Please ensure Conda is installed and in your PATH." >&2
    exit 1
fi

# Find the base conda environment path
CONDA_BASE=$(conda info --base)

# Source the conda initialization script to enable `conda activate`
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [ -f "$CONDA_BASE/bin/activate" ]; then
    # Fallback for older conda versions or different setups
    source "$CONDA_BASE/bin/activate"
    conda activate base # Activate base to ensure conda commands are fully available
else
    echo "Warning: Could not find standard conda initialization script. Assuming conda is already initialized." >&2
fi

# --- Environment Creation ---
echo "Ensuring a clean conda environment at $ENV_DIR..."
# Remove existing environment if it exists, to ensure a clean slate
if [ -d "$ENV_DIR" ]; then
    echo "Existing environment directory found. Attempting to remove it."
    # Use `conda env remove` first, ignore errors if env is not registered
    conda env remove -p "$ENV_DIR" -y || true
    # Force remove the directory to ensure it's completely gone
    rm -rf "$ENV_DIR" || { echo "Error: Failed to remove existing environment directory." >&2; exit 1; }
fi

echo "Creating new conda environment with Python 3.11..."
conda create -y -p "$ENV_DIR" python=3.11 || { echo "Error: Failed to create conda environment." >&2; exit 1; }

# --- Environment Activation ---
echo "Activating conda environment: $ENV_DIR"
conda activate "$ENV_DIR" || { echo "Error: Failed to activate conda environment." >&2; exit 1; }

# --- Package Installation ---
echo "Installing uv (a fast pip installer)..."
pip install uv || { echo "Error: Failed to install uv." >&2; exit 1; }

echo "Installing required packages from requirements files using uv..."
# Install packages from both requirements files.
# `uv` respects exact versions and avoids unnecessary upgrades by default.
# `--prerelease=allow` is included as specified.
uv pip install -r "$MULTIMODAL_REQ_PATH" --prerelease=allow -r "$COMMON_REQ_PATH" \
    || { echo "Error: Failed to install packages from requirements files." >&2; exit 1; }

echo "Installing core ML packages: numpy, pandas, scikit-learn using uv..."
uv pip install numpy pandas scikit-learn \
    || { echo "Error: Failed to install numpy, pandas, scikit-learn." >&2; exit 1; }

echo "Conda environment setup complete at $ENV_DIR."