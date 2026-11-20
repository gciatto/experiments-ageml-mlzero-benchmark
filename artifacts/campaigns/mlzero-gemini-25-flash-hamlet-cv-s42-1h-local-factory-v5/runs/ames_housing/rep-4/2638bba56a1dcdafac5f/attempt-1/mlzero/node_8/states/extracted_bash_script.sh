#!/bin/bash
set -euo pipefail

# Define paths for the specific node_8 environment and requirements files.
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-4/2638bba56a1dcdafac5f/attempt-1/mlzero/node_8/conda_env"
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AUTOGLUON_TABULAR_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "Starting minimal BASH INSTALLATION script for AutoGluon Tabular (node_8)."

# --- Conda Initialization ---
# Dynamically find the conda base path and initialize conda.
# This addresses previous errors related to conda.sh not being found or conda not being initialized.
CONDA_BASE=$(conda info --base 2>/dev/null || true)
if [ -z "$CONDA_BASE" ]; then
    echo "Error: 'conda' command not found or not initialized. Please ensure Conda is installed and in your PATH." >&2
    exit 1
fi

CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
else
    echo "Error: conda.sh not found at $CONDA_SH. Is Conda installed and initialized correctly?" >&2
    exit 1
fi

# --- Create or Activate Conda Environment ---
echo "Checking for existing conda environment at $CONDA_ENV_PATH..."
# Check if the environment directory exists and contains a Python executable.
if [ -d "$CONDA_ENV_PATH" ] && [ -f "$CONDA_ENV_PATH/bin/python" ]; then
    echo "Conda environment already exists. Activating it."
    conda activate "$CONDA_ENV_PATH"
    # Optional: Verify Python version. If it's incorrect, `uv` might still fail later.
    CURRENT_PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [ "$CURRENT_PYTHON_VERSION" != "3.11" ]; then
        echo "Warning: Existing environment has Python $CURRENT_PYTHON_VERSION, but 3.11 was requested. Proceeding, but issues might arise if dependencies are incompatible." >&2
    fi
else
    echo "Conda environment not found or incomplete. Creating a new one with Python 3.11."
    # Ensure the parent directory exists before creating the environment.
    mkdir -p "$(dirname "$CONDA_ENV_PATH")"
    conda create --prefix "$CONDA_ENV_PATH" python=3.11 -y
    conda activate "$CONDA_ENV_PATH"
fi

# --- Install uv ---
echo "Installing uv into the environment..."
# Use the pip from the activated environment to ensure it's installed in the correct environment.
pip install uv

# --- Install required packages using uv ---
echo "Installing required packages using uv from specified requirements files..."
# `uv` is used to install packages. It respects existing installations and
# only installs exact packages and their dependencies, avoiding unnecessary upgrades.
# The `--prerelease=allow` flag is included as per the prompt.
# The order of requirements files is specified: autogluon.tabular first, then _common.
uv pip install \
    -r "$AUTOGLUON_TABULAR_REQUIREMENTS_FILE" \
    --prerelease=allow \
    -r "$COMMON_REQUIREMENTS_FILE"

echo "Installation complete. Environment '$CONDA_ENV_PATH' is ready."