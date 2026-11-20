#!/bin/bash
set -euo pipefail

# Define paths
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/madelon/rep-3/b1070e7c10577bba8210/attempt-1/mlzero/node_1/conda_env"
REQUIREMENTS_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_MULTIMODAL="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

echo "Starting minimal BASH INSTALLATION script for AutoGluon MultiModal..."
echo "Target Conda environment path: $CONDA_ENV_PATH"

# 1. Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: 'conda' command not found. Please ensure Anaconda/Miniconda is installed and in your PATH."
    exit 1
fi

# 2. Remove existing environment if it exists to ensure a clean slate
# Check if the environment is listed by conda
if conda env list | grep -q "${CONDA_ENV_PATH}"; then
    echo "Removing existing conda environment at $CONDA_ENV_PATH..."
    conda env remove --prefix "$CONDA_ENV_PATH" -y
fi
# Ensure the directory is completely gone, even if conda failed to remove it fully
if [ -d "$CONDA_ENV_PATH" ]; then
    echo "Force removing residual directory $CONDA_ENV_PATH..."
    rm -rf "$CONDA_ENV_PATH"
fi

# 3. Create a new conda environment with Python 3.11
echo "Creating new conda environment with Python 3.11 at $CONDA_ENV_PATH..."
conda create --prefix "$CONDA_ENV_PATH" python=3.11 -y

# 4. Activate the environment
# Source conda.sh to make 'conda activate' available in the script's context
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

echo "Conda environment activated. Python interpreter: $(which python)"
echo "Python version: $(python --version)"

# 5. Install uv
echo "Installing uv into the environment..."
pip install uv

# 6. Install core libraries: numpy, pandas, scikit-learn
# These are explicitly required by the factory module and the prompt.
echo "Installing numpy, pandas, scikit-learn using uv..."
uv pip install numpy pandas scikit-learn

# 7. Install required packages from requirements files using uv
# The --prerelease=allow flag is crucial as specified.
# uv respects version pins and avoids unnecessary upgrades/reinstalls.
echo "Installing packages from requirements files using uv..."
uv pip install -r "$REQUIREMENTS_MULTIMODAL" --prerelease=allow -r "$REQUIREMENTS_COMMON"

echo "Installation complete for conda environment at $CONDA_ENV_PATH."
echo "Environment is ready for the factory module."