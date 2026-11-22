#!/bin/bash

# Define the environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5-textfix/runs/adult/rep-3/b5ebccdce10a37599328/attempt-1/mlzero/node_1/conda_env"

# Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate the environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Install required packages from registry files
# Using --prerelease=allow as requested
uv pip install -r /root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine\ learning/requirements.txt \
    --prerelease=allow \
    -r /root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt

# Install specific dependencies required by the factory module
# numpy, pandas, and scikit-learn are typically included in the registry, 
# but we ensure they are present here.
uv pip install numpy pandas scikit-learn

# Verify installation exit status
if [ $? -ne 0 ]; then
    echo "Installation failed."
    exit 1
fi