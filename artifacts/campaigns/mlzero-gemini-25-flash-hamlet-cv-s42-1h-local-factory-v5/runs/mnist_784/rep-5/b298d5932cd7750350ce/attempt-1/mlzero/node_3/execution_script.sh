#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Print commands and their arguments as they are executed.
set -x

# Define the target conda environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/mnist_784/rep-5/b298d5932cd7750350ce/attempt-1/mlzero/node_3/conda_env"

# Define paths to the requirements files
COMMON_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQ_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# --- Step 1: Ensure conda is available ---
if ! command -v conda &> /dev/null
then
    echo "Error: 'conda' command not found. Please ensure Anaconda/Miniconda is installed and in your PATH."
    exit 1
fi

# --- Step 2: Create the conda environment with Python 3.11 ---
echo "Creating conda environment at $ENV_PATH with Python 3.11..."
# Use --yes for non-interactive creation.
conda create --prefix "$ENV_PATH" python=3.11 --yes || {
    echo "Error: Failed to create conda environment at $ENV_PATH."
    exit 1
}

# --- Step 3: Install uv into the newly created environment ---
echo "Installing uv (a fast Python package installer) into the environment..."
# Use 'conda run -p' to execute commands directly within the specified environment
# without needing to activate it explicitly. This addresses previous activation issues.
conda run -p "$ENV_PATH" pip install uv || {
    echo "Error: Failed to install uv into $ENV_PATH."
    exit 1
}

# --- Step 4: Install required packages using uv ---
echo "Installing numpy, pandas, scikit-learn, and packages from requirements files using uv..."
# The prompt specifies to install numpy, pandas, scikit-learn directly,
# and to use the exact uv command for the requirements files.
# uv will handle dependency resolution, respect specified versions,
# and avoid reinstallation if packages are already at the correct version.
conda run -p "$ENV_PATH" uv pip install \
    numpy pandas scikit-learn \
    -r "$MULTIMODAL_REQ_FILE" \
    --prerelease=allow \
    -r "$COMMON_REQ_FILE" \
    || {
    echo "Error: Failed to install packages using uv into $ENV_PATH."
    exit 1
}

echo "Conda environment setup complete at $ENV_PATH."
echo "To use this environment, you can run: conda activate $ENV_PATH"