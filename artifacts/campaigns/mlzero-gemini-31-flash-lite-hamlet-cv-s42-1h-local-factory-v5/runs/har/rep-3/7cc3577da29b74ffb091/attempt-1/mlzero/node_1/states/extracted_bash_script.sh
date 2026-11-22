#!/bin/bash
set -e

# Define environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/har/rep-3/7cc3577da29b74ffb091/attempt-1/mlzero/node_1/conda_env"

# Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Install dependencies from specified requirement files
uv pip install \
    -r /root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine\ learning/requirements.txt \
    --prerelease=allow \
    -r /root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt

# Install core ML packages required by the factory module
uv pip install numpy pandas scikit-learn