#!/bin/bash
set -euo pipefail

# Define paths
# The target conda environment path for node_9
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-2/ffb0a47cbef81e5b1c5d/attempt-1/mlzero/node_9/conda_env"
COMMON_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AUTOGLUON_TABULAR_REQ_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

# 1. Create the conda environment with Python 3.11
echo "Creating conda environment at $CONDA_ENV_PATH with Python 3.11..."
conda create -p "$CONDA_ENV_PATH" python=3.11 -y || { echo "ERROR: Failed to create conda environment at $CONDA_ENV_PATH"; exit 1; }

# 2. Activate the environment
echo "Activating conda environment..."
# Temporarily disable 'set -u' to prevent issues with conda.sh when CONDA_SHLVL might be unset.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "ERROR: Failed to source conda initialization script."; exit 1; }
conda activate "$CONDA_ENV_PATH" || { echo "ERROR: Failed to activate conda environment $CONDA_ENV_PATH"; exit 1; }
set -u # Re-enable 'set -u'

# 3. Install uv
echo "Installing uv into the environment..."
pip install uv || { echo "ERROR: Failed to install uv."; exit 1; }

# 4. Install required packages using uv
# Explicitly install numpy, pandas, scikit-learn as requested.
# uv handles exact package versions and avoids unnecessary upgrades/reinstalls by default,
# fulfilling the requirement to "Only install the exact packages specified... Do NOT upgrade or reinstall".
echo "Installing required packages (numpy, pandas, scikit-learn, and from requirements files) using uv..."
uv pip install \
    numpy pandas scikit-learn \
    -r "$AUTOGLUON_TABULAR_REQ_PATH" \
    -r "$COMMON_REQ_PATH" \
    --prerelease=allow || { echo "ERROR: Failed to install required packages with uv."; exit 1; }

echo "Conda environment setup complete at $CONDA_ENV_PATH"