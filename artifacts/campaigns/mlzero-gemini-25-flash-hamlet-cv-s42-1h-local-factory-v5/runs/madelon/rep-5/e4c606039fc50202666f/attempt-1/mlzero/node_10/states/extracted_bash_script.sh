#!/bin/bash
set -euo pipefail

# Define environment and requirements paths
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/madelon/rep-5/e4c606039fc50202666f/attempt-1/mlzero/node_10/conda_env"
COMMON_REQUIREMENTS_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
MULTIMODAL_REQUIREMENTS_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt"
PYTHON_VERSION="3.11"

echo "Starting minimal BASH INSTALLATION script for AutoGluon MultiModal..."

# --- 1. Initialize Conda ---
# Robustly initialize conda for the current shell session.
# This addresses previous errors where 'conda' command was not found or activated.
CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: Could not determine conda base path. Is conda installed and in PATH?" >&2
    exit 1
fi

if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "Sourcing conda initialization script from $CONDA_BASE/etc/profile.d/conda.sh"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found at $CONDA_BASE/etc/profile.d/conda.sh. Conda environment activation may fail." >&2
    exit 1
fi

# --- 2. Create or Recreate Conda Environment ---
# Remove existing environment if it exists to ensure a clean slate for installation.
# Using 'conda env list --json' for more robust check
if conda env list --json | grep -q "\"$CONDA_ENV_PATH\""; then
    echo "Removing existing conda environment at $CONDA_ENV_PATH..."
    conda env remove -p "$CONDA_ENV_PATH" -y
fi

echo "Creating new conda environment at $CONDA_ENV_PATH with Python $PYTHON_VERSION..."
conda create -p "$CONDA_ENV_PATH" python="$PYTHON_VERSION" -y

# --- 3. Activate the Environment ---
echo "Activating conda environment: $CONDA_ENV_PATH"
conda activate "$CONDA_ENV_PATH"

# --- 4. Install uv ---
echo "Installing uv into the activated environment..."
pip install uv

# --- 5. Install Required Packages using uv ---
# Install numpy, pandas, scikit-learn explicitly, along with packages from requirements files.
# The --prerelease=allow flag is included as specified.
# uv handles dependency resolution and avoids unnecessary upgrades if versions are satisfied.
echo "Installing core packages (numpy, pandas, scikit-learn) and requirements using uv..."
uv pip install \
    numpy pandas scikit-learn \
    -r "$MULTIMODAL_REQUIREMENTS_PATH" \
    --prerelease=allow \
    -r "$COMMON_REQUIREMENTS_PATH"

echo "Conda environment setup complete at $CONDA_ENV_PATH."
echo "All specified packages have been installed."