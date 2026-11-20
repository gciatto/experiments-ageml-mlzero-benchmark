#!/bin/bash
set -euo pipefail

# Define environment and requirements paths
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-5/af77f36f78b70c5e47ce/attempt-1/mlzero/node_2/conda_env"
COMMON_REQUIREMENTS="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQUIREMENTS="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

echo "--- Starting Conda Environment Setup ---"

# 1. Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: 'conda' command not found. Please ensure Conda is installed and in your PATH." >&2
    exit 1
fi

# 2. Remove existing environment if it exists, then create a new one
# This ensures a clean setup as per "Create and configure a conda environment"
if [ -d "$CONDA_ENV_PATH" ]; then
    echo "Existing conda environment found at $CONDA_ENV_PATH. Removing it for a clean setup..."
    conda env remove --prefix "$CONDA_ENV_PATH" -y || { echo "Failed to remove existing conda environment." >&2; exit 1; }
fi

echo "Creating new conda environment at $CONDA_ENV_PATH with Python 3.11 and pip..."
conda create --prefix "$CONDA_ENV_PATH" python=3.11 pip -y || { echo "Failed to create conda environment." >&2; exit 1; }

# 3. Install 'uv' into the newly created environment
echo "Installing 'uv' into the conda environment..."
# Using 'conda run' ensures commands execute within the specified environment,
# addressing previous 'CondaError' issues with 'conda activate'.
conda run --prefix "$CONDA_ENV_PATH" pip install uv || { echo "Failed to install 'uv'." >&2; exit 1; }

# 4. Install required packages using 'uv'
# This includes packages from both requirements files, plus numpy, pandas, scikit-learn.
# 'uv' handles dependency resolution, installs exact versions, and avoids unnecessary upgrades.
echo "Installing core packages (numpy, pandas, scikit-learn) and requirements using 'uv'..."
conda run --prefix "$CONDA_ENV_PATH" uv pip install \
    -r "$COMMON_REQUIREMENTS" \
    -r "$MULTIMODAL_REQUIREMENTS" \
    numpy pandas scikit-learn \
    --prerelease=allow \
    || { echo "Failed to install required packages with 'uv'." >&2; exit 1; }

echo "--- Conda Environment Setup Complete ---"
echo "Environment is ready at: $CONDA_ENV_PATH"
echo "To activate it manually for interactive use, run: conda activate $CONDA_ENV_PATH"