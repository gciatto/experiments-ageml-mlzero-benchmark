#!/bin/bash

# Define the target environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/auto_mpg/rep-5/895c57407cc2f003dad9/attempt-1/mlzero/node_0/conda_env"

# Create the conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate the environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Install required packages using uv
# We use --no-upgrade to ensure we don't unnecessarily update existing packages
# and rely on the specified requirements files.
uv pip install \
    --no-upgrade \
    -r /root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt \
    --prerelease=allow \
    -r /root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt

# Ensure core dependencies are present (numpy, pandas, scikit-learn)
# These are typically included in the requirements files, but explicitly 
# ensuring they are installed in the local environment.
uv pip install numpy pandas scikit-learn