#!/bin/bash

# Define the target environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/auto_mpg/rep-5/895c57407cc2f003dad9/attempt-1/mlzero/node_4/conda_env"

# Create the conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate the environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Install required packages
# Using --no-upgrade to respect existing versions and prevent unnecessary updates
# uv pip install handles dependencies automatically
uv pip install \
    --no-upgrade \
    numpy \
    pandas \
    scikit-learn \
    -r /root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt \
    --prerelease=allow \
    -r /root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt