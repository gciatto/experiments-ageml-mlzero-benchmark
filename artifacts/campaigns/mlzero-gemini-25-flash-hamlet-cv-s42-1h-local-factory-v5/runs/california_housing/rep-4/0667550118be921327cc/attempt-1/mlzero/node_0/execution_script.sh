#!/bin/bash
set -euo pipefail # Exit immediately if a command exits with a non-zero status, exit if an unset variable is used, propagate pipe errors

# Define the target Conda environment path
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/california_housing/rep-4/0667550118be921327cc/attempt-1/mlzero/node_0/conda_env"

# Define paths to the requirements files
REQUIREMENTS_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_AUTOGLUON_TABULAR="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "Starting minimal BASH INSTALLATION script for factory module..."
echo "Target Conda environment path: $CONDA_ENV_PATH"

# --- Step 1: Ensure Conda is available and initialized ---
if ! command -v conda &> /dev/null; then
    echo "Error: 'conda' command not found. Please ensure Anaconda/Miniconda is installed and initialized in your PATH."
    exit 1
fi

# Deactivate any currently active conda environment to ensure a clean state
conda deactivate 2>/dev/null || true

# Initialize conda for the current shell session if not already done
# This is crucial for 'conda activate' to work correctly in a script
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Warning: conda.sh not found. Conda activation might fail. Ensure conda is properly installed."
fi

# --- Step 2: Create the Conda environment ---
echo "Creating Conda environment with Python 3.11 at $CONDA_ENV_PATH..."
# Remove existing environment if it exists to ensure a clean creation
if [ -d "$CONDA_ENV_PATH" ]; then
    echo "Existing environment found at $CONDA_ENV_PATH. Removing it..."
    conda env remove --prefix "$CONDA_ENV_PATH" --yes || true # Use || true to not fail if removal fails for non-critical reasons
fi
conda create --prefix "$CONDA_ENV_PATH" python=3.11 --yes

# --- Step 3: Activate the environment ---
echo "Activating the newly created Conda environment..."
conda activate "$CONDA_ENV_PATH"

# Verify environment activation
if [[ "$CONDA_PREFIX" != "$CONDA_ENV_PATH" ]]; then
    echo "Error: Failed to activate Conda environment. CONDA_PREFIX is '$CONDA_PREFIX', expected '$CONDA_ENV_PATH'."
    exit 1
fi
echo "Conda environment successfully activated: $CONDA_PREFIX"

# --- Step 4: Install uv ---
echo "Installing 'uv' into the active environment..."
pip install uv

# --- Step 5: Install core packages (numpy, pandas, scikit-learn) ---
echo "Installing numpy, pandas, and scikit-learn using uv..."
uv pip install numpy pandas scikit-learn

# --- Step 6: Install required packages from requirements files using uv ---
echo "Installing packages from requirements files using uv..."
# The prompt specifies the order: autogluon.tabular/requirements.txt first, then _common/requirements.txt.
# uv handles multiple -r flags and ensures exact versions/dependencies.
uv pip install \
    -r "$REQUIREMENTS_AUTOGLUON_TABULAR" \
    --prerelease=allow \
    -r "$REQUIREMENTS_COMMON"

echo "All specified packages have been installed into the Conda environment."
echo "Installation script finished successfully."