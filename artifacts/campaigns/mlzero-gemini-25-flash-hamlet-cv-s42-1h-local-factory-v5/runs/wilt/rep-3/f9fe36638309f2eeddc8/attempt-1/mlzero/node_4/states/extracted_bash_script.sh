#!/bin/bash
set -euo pipefail

# Define the target Conda environment path as specified in the prompt
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/wilt/rep-3/f9fe36638309f2eeddc8/attempt-1/mlzero/node_4/conda_env"

# Define paths to requirements files
REQUIREMENTS_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_TABULAR="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

echo "Starting Conda environment setup at $CONDA_ENV_PATH"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda command not found. Please ensure Conda is installed and in your PATH."
    exit 1
fi

# Remove existing conda environment to ensure a clean slate for installation
if [ -d "$CONDA_ENV_PATH" ]; then
    echo "Removing existing conda environment at $CONDA_ENV_PATH..."
    # Use -y for non-interactive removal. `|| true` prevents script exit if removal fails for non-critical reasons.
    conda env remove --prefix "$CONDA_ENV_PATH" -y || true
fi

echo "Creating conda environment with Python 3.11 and pip..."
conda create --prefix "$CONDA_ENV_PATH" python=3.11 pip -y

# Initialize conda for the current shell session to enable 'conda activate'
# This is crucial for non-interactive scripts.
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "Error: conda.sh not found at $CONDA_BASE/etc/profile.d/conda.sh. Conda initialization failed."
    exit 1
fi

# Activate the newly created environment
echo "Activating conda environment..."
conda activate "$CONDA_ENV_PATH"

# Verify environment activation
if [[ "$CONDA_PREFIX" != "$CONDA_ENV_PATH" ]]; then
    echo "Error: Failed to activate conda environment. CONDA_PREFIX is '$CONDA_PREFIX', expected '$CONDA_ENV_PATH'"
    exit 1
fi
echo "Conda environment activated: $CONDA_PREFIX"

# Install uv, a fast package installer
echo "Installing uv..."
pip install uv

# Explicitly install numpy, pandas, and scikit-learn as requested.
# This addresses the previous 'numpy not found' error and ensures these core libraries are present.
echo "Installing core libraries: numpy, pandas, scikit-learn..."
uv pip install numpy pandas scikit-learn

# Install required packages from the specified requirements files using uv.
# The --prerelease=allow flag is included as per the prompt.
# uv's default behavior respects existing installations if versions match,
# fulfilling the "Do NOT upgrade or reinstall autogluon.tabular" requirement.
echo "Installing packages from requirements files using uv..."
uv pip install -r "$REQUIREMENTS_TABULAR" --prerelease=allow -r "$REQUIREMENTS_COMMON"

echo "Conda environment setup complete."