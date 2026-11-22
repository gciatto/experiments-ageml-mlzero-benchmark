#!/bin/bash

# Define environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/ames_housing/rep-2/21f956854eff2b2d705a/attempt-1/mlzero/node_4/conda_env"

# Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Install core dependencies
uv pip install numpy pandas scikit-learn

# Install requirements from specified paths
# --no-upgrade ensures that if autogluon.multimodal is already present, it is not reinstalled/upgraded
uv pip install \
    -r /root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.multimodal/requirements.txt \
    --prerelease=allow \
    -r /root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt \
    --no-upgrade