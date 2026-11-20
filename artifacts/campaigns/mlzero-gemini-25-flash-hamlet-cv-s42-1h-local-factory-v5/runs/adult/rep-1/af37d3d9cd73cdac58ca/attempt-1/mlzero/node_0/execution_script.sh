#!/bin/bash
# Minimal BASH script to create and configure a Conda environment.

# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error when substituting.
# The exit status of a pipeline is the status of the last command to exit with a non-zero status, or zero if no command exited with a non-zero status.
set -euo pipefail

# --- Configuration Variables ---
# Path where the conda environment will be created.
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/adult/rep-1/af37d3d9cd73cdac58ca/attempt-1/mlzero/node_0/conda_env"

# Paths to the requirements files.
TABULAR_REQUIREMENTS_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"
COMMON_REQUIREMENTS_PATH="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"

# --- Script Execution ---

# 1. Check for Conda availability
if ! command -v conda &> /dev/null; then
    echo "Error: 'conda' command not found. Please ensure Anaconda or Miniconda is installed and in your PATH." >&2
    exit 1
fi

# 2. Create the Conda environment with Python 3.11.
# The -p flag specifies the exact path for the environment.
# The -y flag automatically answers yes to prompts.
conda create -y -p "${ENV_PATH}" python=3.11

# 3. Install 'uv' into the newly created environment.
# Using 'conda run -p' ensures the command executes within the specified environment
# without needing to explicitly activate it in the current shell.
conda run -p "${ENV_PATH}" pip install uv

# 4. Install required packages using 'uv pip install'.
# The order of -r flags is maintained as specified in the prompt.
# --prerelease=allow is included as specified.
conda run -p "${ENV_PATH}" uv pip install \
    -r "${TABULAR_REQUIREMENTS_PATH}" \
    --prerelease=allow \
    -r "${COMMON_REQUIREMENTS_PATH}"