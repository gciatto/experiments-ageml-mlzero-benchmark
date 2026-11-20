#!/bin/bash
set -euxo pipefail

# Define the target Conda environment directory
ENV_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/har/rep-4/e3e33218f51e76d801d1/attempt-1/mlzero/node_10/conda_env"

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

# 2. Install uv into the newly created environment
echo "Installing uv into the environment..."
# Use 'conda run -p' to execute commands directly within the specified environment,
# bypassing the need for explicit 'conda activate' and addressing previous activation errors.
conda run -p "$ENV_DIR" pip install uv
if [ $? -ne 0 ]; then
    echo "ERROR: uv installation failed." >&2
    exit 1
fi

# 3. Install required packages from requirements files and additional libraries using uv
echo "Installing packages from requirements files and additional libraries (numpy, pandas, scikit-learn) using uv..."
conda run -p "$ENV_DIR" uv pip install \
    -r "$MM_REQ_FILE" \
    --prerelease=allow \
    -r "$COMMON_REQ_FILE" \
    numpy pandas scikit-learn
if [ $? -ne 0 ]; then
    echo "ERROR: Package installation with uv failed." >&2
    exit 1
fi

echo "Conda environment setup complete at $ENV_DIR."