#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Define the full path for the conda environment
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/madelon/rep-1/b94fca9f26a3228d15f1/attempt-1/mlzero/node_3/conda_env"

# Define paths to requirements files
COMMON_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AUTOGLUON_TABULAR_REQUIREMENTS_FILE="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "--- Starting Conda environment setup ---"

# Ensure the parent directory for the conda environment exists
mkdir -p "$(dirname "$CONDA_ENV_PATH")" || { echo "ERROR: Failed to create parent directory for conda environment: $(dirname "$CONDA_ENV_PATH")"; exit 1; }

# 1. Create the conda environment with Python 3.11
echo "Creating conda environment at $CONDA_ENV_PATH with Python 3.11..."
# Use --yes to automatically proceed with the installation
# Use --prefix to specify the exact installation path
conda create --prefix "$CONDA_ENV_PATH" python=3.11 --yes || { echo "ERROR: Failed to create conda environment at $CONDA_ENV_PATH. Please check conda installation, disk space, and permissions."; exit 1; }

# 2. Activate the environment
echo "Activating conda environment..."
# Initialize conda for the current shell session if not already done.
# This is crucial for 'conda activate' to work in non-interactive scripts.
# It's assumed that 'conda' command is available in the PATH.
# If 'conda info --base' fails, it indicates a fundamental issue with conda setup.
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: 'conda info --base' failed. Conda might not be installed or initialized correctly in this shell's PATH."
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh" || { echo "ERROR: Failed to source conda.sh from $CONDA_BASE. Check conda installation and initialization."; exit 1; }

conda activate "$CONDA_ENV_PATH" || { echo "ERROR: Failed to activate conda environment $CONDA_ENV_PATH. Check if the environment was created successfully and if conda is properly initialized."; exit 1; }

# 3. Install uv
echo "Installing uv..."
# Use the pip from the activated conda environment
pip install uv || { echo "ERROR: Failed to install uv into $CONDA_ENV_PATH."; exit 1; }

# 4. Install required packages using uv
echo "Installing required packages from $AUTOGLUON_TABULAR_REQUIREMENTS_FILE and $COMMON_REQUIREMENTS_FILE using uv..."
echo "Also installing numpy, pandas, scikit-learn explicitly as requested."
# uv's default behavior is to resolve and install dependencies without upgrading existing packages unless specified.
# This ensures "Only install the exact packages specified in the requirements files with their dependencies."
# and "Do NOT upgrade or reinstall autogluon.tabular if it's already at the correct version".
uv pip install -r "$AUTOGLUON_TABULAR_REQUIREMENTS_FILE" --prerelease=allow -r "$COMMON_REQUIREMENTS_FILE" numpy pandas scikit-learn || { echo "ERROR: Failed to install required packages into $CONDA_ENV_PATH."; exit 1; }

echo "--- Conda environment setup complete ---"