#!/bin/bash
set -euo pipefail

# Define the target conda environment path
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/mnist_784/rep-3/9c73adddfe0aa804051e/attempt-1/mlzero/node_3/conda_env"

# Define paths to the requirements files
COMMON_REQUIREMENTS_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AUTOGLUON_TABULAR_REQUIREMENTS_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "Starting minimal BASH INSTALLATION script for AutoGluon Tabular dependencies."
echo "Target Conda environment path: $CONDA_ENV_PATH"

# Ensure conda is initialized for the current shell session.
# This block attempts to source the conda initialization script if not already done.
if ! type "conda" > /dev/null; then
    echo "Error: 'conda' command not found. Please ensure Conda is installed and initialized in your shell."
    exit 1
fi

# Check if CONDA_SHLVL is set, which indicates conda is already active or initialized.
if [ -z "${CONDA_SHLVL:-}" ]; then
    echo "Initializing conda for the current shell session..."
    # Use 'conda info --base' to reliably find the conda installation path
    CONDA_BASE=$(conda info --base)
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "Warning: Could not find conda.sh script at $CONDA_BASE/etc/profile.d/conda.sh. 'conda activate' might fail."
    fi
fi

# Remove existing environment directory to ensure a clean installation
if [ -d "$CONDA_ENV_PATH" ]; then
    echo "Existing conda environment found at $CONDA_ENV_PATH. Removing it to ensure a clean install."
    rm -rf "$CONDA_ENV_PATH" || { echo "Failed to remove existing conda environment."; exit 1; }
fi

# 1. Create the conda environment with Python 3.11
echo "Creating conda environment with Python 3.11 at $CONDA_ENV_PATH..."
conda create --prefix "$CONDA_ENV_PATH" python=3.11 --yes || { echo "Failed to create conda environment."; exit 1; }

# 2. Activate the newly created environment
echo "Activating conda environment: $CONDA_ENV_PATH"
conda activate "$CONDA_ENV_PATH" || { echo "Failed to activate conda environment."; exit 1; }

# 3. Install uv
echo "Installing uv into the activated environment..."
# Use the pip from the activated environment to ensure it's installed correctly within the env
"$CONDA_ENV_PATH/bin/pip" install uv || { echo "Failed to install uv."; exit 1; }

# 4. Install required packages from the specified requirements files using uv
echo "Installing packages from requirements files using uv pip install..."
# The command is exactly as specified in the prompt, ensuring specific versions and dependencies.
"$CONDA_ENV_PATH/bin/uv" pip install \
    -r "$AUTOGLUON_TABULAR_REQUIREMENTS_PATH" \
    --prerelease=allow \
    -r "$COMMON_REQUIREMENTS_PATH" || { echo "Failed to install requirements using uv."; exit 1; }

# 5. Install numpy, pandas, and scikit-learn
echo "Installing numpy, pandas, scikit-learn using uv pip install..."
"$CONDA_ENV_PATH/bin/uv" pip install numpy pandas scikit-learn || { echo "Failed to install numpy, pandas, scikit-learn."; exit 1; }

echo "Installation complete. Deactivating conda environment."
conda deactivate || { echo "Failed to deactivate conda environment."; exit 1; }

echo "Script finished successfully."