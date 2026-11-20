#!/bin/bash
set -e

# Define environment and file paths
CONDA_ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5/runs/har/rep-5/dd7dd685e23b34db5767/attempt-1/mlzero/node_2/conda_env"
PYTHON_VERSION="3.11"
REQUIREMENTS_COMMON="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
REQUIREMENTS_ML="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

echo "--- Starting Conda Environment Setup ---"

# 1. Create the conda environment
echo "Creating conda environment at $CONDA_ENV_PATH with Python $PYTHON_VERSION..."
conda create -p "$CONDA_ENV_PATH" python="$PYTHON_VERSION" -y

# Check if conda environment creation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to create conda environment."
    exit 1
fi
echo "Conda environment created successfully."

# 2. Activate the environment
echo "Activating conda environment..."
# Source conda.sh to make 'conda' command available in non-interactive scripts
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment."
    exit 1
fi
echo "Conda environment activated."

# Define paths to the environment's python and pip
PYTHON_BIN="$CONDA_ENV_PATH/bin/python"
PIP_BIN="$CONDA_ENV_PATH/bin/pip"

# 3. Install uv using the environment's pip
echo "Installing uv into the environment..."
"$PIP_BIN" install uv

# Check if uv installation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to install uv."
    exit 1
fi
echo "uv installed successfully."

# Define path to the environment's uv
UV_BIN="$CONDA_ENV_PATH/bin/uv"

# 4. Install required packages using uv
echo "Installing numpy, pandas, scikit-learn, autogluon.tabular and packages from requirements files using uv..."
"$UV_BIN" pip install \
    numpy pandas scikit-learn autogluon.tabular \
    -r "$REQUIREMENTS_ML" \
    --prerelease=allow \
    -r "$REQUIREMENTS_COMMON"

# Check if package installation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to install required packages with uv."
    exit 1
fi
echo "All required packages installed successfully."

echo "--- Conda Environment Setup Complete ---"