#!/bin/bash
set -euo pipefail

# Define paths
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-2/ffb0a47cbef81e5b1c5d/attempt-1/mlzero/node_5/conda_env"
COMMON_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AUTOGLUON_TABULAR_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

# 1. Create the conda environment with Python 3.11
echo "Creating conda environment at $CONDA_ENV_PATH with Python 3.11..."
conda create -p "$CONDA_ENV_PATH" python=3.11 -y

# 2. Activate the environment
echo "Activating conda environment..."
# Initialize conda for the current shell. This is crucial for `conda activate` to work in scripts.
# The `conda info --base` command should be available in the execution environment.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

# 3. Install uv
echo "Installing uv into the environment..."
pip install uv

# 4. Install required packages using uv
# Add numpy, pandas, scikit-learn as explicitly requested by the prompt.
# uv handles exact package versions and avoids unnecessary upgrades/reinstalls by default,
# fulfilling the requirement to "Only install the exact packages specified... Do NOT upgrade or reinstall".
echo "Installing required packages (numpy, pandas, scikit-learn, and from requirements files) using uv..."
uv pip install \
    numpy pandas scikit-learn \
    -r "$AUTOGLUON_TABULAR_REQ_PATH" \
    -r "$COMMON_REQ_PATH" \
    --prerelease=allow

echo "Conda environment setup complete at $CONDA_ENV_PATH"