#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Define the target conda environment path
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/texture/rep-5/f597436d592bb5952d1c/attempt-1/mlzero/node_8/conda_env"

# Define paths to the requirements files
COMMON_REQUIREMENTS="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQUIREMENTS="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"

# --- Conda Initialization (Addresses previous CondaError) ---
# Find the conda.sh script and source it to initialize conda in the current shell.
# This ensures 'conda activate' works correctly regardless of initial PATH configuration.
echo "Attempting to initialize Conda..."
CONDA_BASE=$(conda info --base 2>/dev/null || true)
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "Sourcing conda.sh from $CONDA_BASE/etc/profile.d/conda.sh"
    . "$CONDA_BASE/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then # Common path in many container environments
    echo "Sourcing conda.sh from /opt/conda/etc/profile.d/conda.sh"
    . "/opt/conda/etc/profile.d/conda.sh"
else
    echo "Error: Could not find conda.sh. Please ensure Conda is installed and accessible." >&2
    exit 1
fi

# Create the conda environment with Python 3.11
echo "Creating conda environment at $CONDA_ENV_PATH with Python 3.11..."
conda create --prefix "$CONDA_ENV_PATH" python=3.11 --yes || { echo "Failed to create conda environment." >&2; exit 1; }

# Activate the newly created environment
echo "Activating conda environment: $CONDA_ENV_PATH"
conda activate "$CONDA_ENV_PATH" || { echo "Failed to activate conda environment." >&2; exit 1; }

# Verify activation
if [[ "$CONDA_PREFIX" != "$CONDA_ENV_PATH" ]]; then
    echo "Error: Conda environment activation failed. CONDA_PREFIX is '$CONDA_PREFIX', expected '$CONDA_ENV_PATH'." >&2
    exit 1
fi
echo "Conda environment activated successfully."

# Install uv using the environment's pip
echo "Installing uv into the environment..."
pip install uv || { echo "Failed to install uv." >&2; exit 1; }

# Install numpy, pandas, scikit-learn explicitly using uv as requested ("Use its interpreter/pip")
echo "Installing numpy, pandas, scikit-learn into the environment using uv..."
uv pip install numpy pandas scikit-learn || { echo "Failed to install core packages (numpy, pandas, scikit-learn)." >&2; exit 1; }

# Install required packages from the specified requirements files using uv.
# The --prerelease=allow flag is included as per the prompt.
# uv's default behavior ensures only specified packages and their dependencies are installed,
# and it avoids unnecessary upgrades if versions match requirements.
echo "Installing packages from requirements files using uv..."
uv pip install \
    -r "$MULTIMODAL_REQUIREMENTS" \
    --prerelease=allow \
    -r "$COMMON_REQUIREMENTS" || { echo "Failed to install packages from requirements files." >&2; exit 1; }

echo "Installation complete for environment: $CONDA_ENV_PATH"

# Deactivate the environment to leave the shell in a clean state (optional, as script exits)
# conda deactivate