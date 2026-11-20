#!/bin/bash
set -euxo pipefail

# Define the target Conda environment directory
ENV_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/har/rep-4/e3e33218f51e76d801d1/attempt-1/mlzero/node_1/conda_env"

# Define paths to requirements files
COMMON_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MM_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# 1. Create the Conda environment with Python 3.11
echo "Creating conda environment at $ENV_DIR with Python 3.11..."
conda create --prefix "$ENV_DIR" python=3.11 --yes
if [ $? -ne 0 ]; then
    echo "ERROR: Conda environment creation failed." >&2
    exit 1
fi

# 2. Activate the environment
echo "Activating conda environment..."
# Ensure conda is initialized for the current shell session
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
if [ $? -ne 0 ]; then
    echo "ERROR: Conda environment activation failed." >&2
    exit 1
fi

# 3. Install uv into the activated environment
echo "Installing uv into the environment..."
pip install uv
if [ $? -ne 0 ]; then
    echo "ERROR: uv installation failed." >&2
    exit 1
fi

# 4. Install required packages using uv
# This command installs packages from both requirements files,
# allows prerelease versions, and includes numpy, pandas, and scikit-learn.
# uv ensures exact package versions from requirements and handles dependencies efficiently,
# avoiding unnecessary upgrades/reinstalls if versions are already satisfied.
echo "Installing packages from requirements files and additional libraries (numpy, pandas, scikit-learn) using uv..."
uv pip install \
    -r "$MM_REQ_FILE" \
    --prerelease=allow \
    -r "$COMMON_REQ_FILE" \
    numpy pandas scikit-learn
if [ $? -ne 0 ]; then
    echo "ERROR: Package installation with uv failed." >&2
    exit 1
fi

echo "Conda environment setup complete at $ENV_DIR."