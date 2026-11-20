#!/bin/bash
set -euo pipefail

# Define environment and file paths
ENV_ROOT="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/auto_mpg/rep-2/36b7694a16463792f153/attempt-1/mlzero/node_3"
ENV_DIR="$ENV_ROOT/conda_env"
COMMON_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

# --- Conda Initialization ---
# Explicitly source conda.sh to properly initialize conda for non-interactive shells.
if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "Error: 'conda' command not found. Please ensure Conda is installed and in your PATH." >&2
    exit 1
fi

# --- Create and Configure Conda Environment ---

echo "Creating conda environment at $ENV_DIR with Python 3.11..."
conda create -p "$ENV_DIR" python=3.11 -y || { echo "Failed to create conda environment." >&2; exit 1; }

echo "Activating conda environment: $ENV_DIR"
conda activate "$ENV_DIR" || { echo "Failed to activate conda environment." >&2; exit 1; }

echo "Installing uv..."
pip install uv || { echo "Failed to install uv." >&2; exit 1; }

echo "Installing required packages from requirements files using uv..."
# Install packages from both requirements files as specified, with --prerelease=allow
uv pip install -r "$ML_REQ_FILE" --prerelease=allow -r "$COMMON_REQ_FILE" || { echo "Failed to install packages from requirements files." >&2; exit 1; }

echo "Installing additional core packages (numpy, pandas, scikit-learn, autogluon.tabular) using uv..."
# Install packages explicitly mentioned in the prompt and required by the factory module
uv pip install numpy pandas scikit-learn autogluon.tabular || { echo "Failed to install core packages." >&2; exit 1; }

echo "Conda environment setup complete."