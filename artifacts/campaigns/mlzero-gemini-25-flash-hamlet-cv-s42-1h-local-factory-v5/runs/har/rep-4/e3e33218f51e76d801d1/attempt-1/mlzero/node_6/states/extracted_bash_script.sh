#!/bin/bash
set -euxo pipefail

# Define the target Conda environment directory
ENV_DIR="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/har/rep-4/e3e33218f51e76d801d1/attempt-1/mlzero/node_6/conda_env"

# Define paths to requirements files
COMMON_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MM_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# Ensure the parent directory exists before creating the environment
mkdir -p "$(dirname "$ENV_DIR")"

# Attempt to remove any existing environment at the target path to ensure a clean slate.
# This helps prevent issues from partially failed previous installations.
if [ -d "$ENV_DIR" ]; then
    echo "Existing environment found at $ENV_DIR. Attempting to remove it..."
    # Use conda env remove if possible, fallback to rm -rf if conda command fails
    conda env remove --prefix "$ENV_DIR" --yes || {
        echo "Warning: 'conda env remove' failed for $ENV_DIR. Attempting 'rm -rf $ENV_DIR' as fallback." >&2
        rm -rf "$ENV_DIR"
    }
fi

# 1. Create the Conda environment with Python 3.11
echo "Creating conda environment at $ENV_DIR with Python 3.11..."
conda create --prefix "$ENV_DIR" python=3.11 --yes
if [ $? -ne 0 ]; then
    echo "ERROR: Conda environment creation failed." >&2
    exit 1
fi

# Verify that the activate script was created. This addresses previous errors
# where the environment creation succeeded but the activation script was missing.
if [ ! -f "$ENV_DIR/bin/activate" ]; then
    echo "ERROR: Conda environment created, but $ENV_DIR/bin/activate not found." >&2
    echo "This indicates a corrupted or incomplete environment setup. Please check disk space, permissions, and conda installation." >&2
    exit 1
fi

# 2. Activate the environment
echo "Activating conda environment..."
# Source conda's initialization script to make 'conda activate' available in the current shell.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
if [ $? -ne 0 ]; then
    echo "ERROR: Conda environment activation failed." >&2
    exit 1
fi

# 3. Install uv into the activated environment
echo "Installing uv into the environment..."
# 'pip' here refers to the pip executable within the activated conda environment.
pip install uv
if [ $? -ne 0 ]; then
    echo "ERROR: uv installation failed." >&2
    exit 1
fi

# 4. Install required packages using uv
# This command installs packages from both requirements files,
# allows prerelease versions, and explicitly includes numpy, pandas, and scikit-learn.
# 'uv' ensures only the exact packages specified (and their dependencies) are installed,
# avoiding unnecessary upgrades or reinstalls if versions are already satisfied.
echo "Installing packages from requirements files and additional libraries (numpy, pandas, scikit-learn) using uv..."
# 'uv' here refers to the uv executable within the activated conda environment.
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