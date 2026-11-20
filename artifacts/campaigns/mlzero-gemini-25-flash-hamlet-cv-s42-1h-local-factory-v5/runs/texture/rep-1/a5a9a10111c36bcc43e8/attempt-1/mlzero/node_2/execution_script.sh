#!/bin/bash
set -euo pipefail

# Define environment and file paths
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/texture/rep-1/a5a9a10111c36bcc43e8/attempt-1/mlzero/node_2/conda_env"
PYTHON_VERSION="3.11"
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# --- Conda Initialization ---
# Ensure 'conda' command is available and 'conda activate' works in the script.
# This block attempts to source the conda initialization script if not already done.
if command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base)
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        echo "Sourcing conda.sh from $CONDA_BASE/etc/profile.d/conda.sh"
        . "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "Warning: conda.sh not found at expected location ($CONDA_BASE/etc/profile.d/conda.sh). 'conda activate' might fail." >&2
    fi
else
    echo "ERROR: 'conda' command not found. Please ensure Conda is installed and in your PATH." >&2
    exit 1
fi
# --- End Conda Initialization ---

# Create the conda environment
echo "Creating conda environment at $CONDA_ENV_PATH with Python $PYTHON_VERSION..."
conda create -p "$CONDA_ENV_PATH" python="$PYTHON_VERSION" -y || { echo "ERROR: Failed to create conda environment." >&2; exit 1; }

# Activate the newly created environment
echo "Activating conda environment: $CONDA_ENV_PATH"
conda activate "$CONDA_ENV_PATH" || { echo "ERROR: Failed to activate conda environment." >&2; exit 1; }

# Install uv using the environment's pip
echo "Installing uv into the activated environment..."
pip install uv || { echo "ERROR: Failed to install uv." >&2; exit 1; }

# Check if requirements files exist
if [ ! -f "$COMMON_REQUIREMENTS_FILE" ]; then
    echo "ERROR: Common requirements file not found: $COMMON_REQUIREMENTS_FILE" >&2
    exit 1
fi
if [ ! -f "$MULTIMODAL_REQUIREMENTS_FILE" ]; then
    echo "ERROR: Multimodal requirements file not found: $MULTIMODAL_REQUIREMENTS_FILE" >&2
    exit 1
fi

# Install required packages using uv
echo "Installing required packages from $MULTIMODAL_REQUIREMENTS_FILE and $COMMON_REQUIREMENTS_FILE using uv..."
# The command is specified to install autogluon.multimodal requirements with --prerelease=allow
# followed by common requirements. uv handles dependencies and avoids unnecessary reinstalls.
uv pip install \
    -r "$MULTIMODAL_REQUIREMENTS_FILE" \
    --prerelease=allow \
    -r "$COMMON_REQUIREMENTS_FILE" \
    || { echo "ERROR: Failed to install packages using uv." >&2; exit 1; }

echo "Conda environment setup complete at $CONDA_ENV_PATH"

# Deactivate the environment (optional, as the script will exit)
# conda deactivate