#!/bin/bash

# Define the target environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/madelon/rep-5/3f50df3b16a6c7f07fcf/attempt-1/mlzero/node_3/conda_env"

# Create conda environment with Python 3.11
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate the environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Install base dependencies
uv pip install numpy pandas scikit-learn

# Install requirements from specified files
# --no-upgrade ensures we do not reinstall/upgrade if already satisfied
# --prerelease=allow allows for the requested prerelease versions
uv pip install \
    --no-upgrade \
    --prerelease=allow \
    -r /root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt \
    -r /root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt

# Propagate installation errors
if [ $? -ne 0 ]; then
    echo "Installation failed."
    exit 1
fi