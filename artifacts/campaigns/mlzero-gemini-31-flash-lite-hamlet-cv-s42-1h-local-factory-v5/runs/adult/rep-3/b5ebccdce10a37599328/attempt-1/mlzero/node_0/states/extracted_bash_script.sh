#!/bin/bash
set -e

# Define environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5-textfix/runs/adult/rep-3/b5ebccdce10a37599328/attempt-1/mlzero/node_0/conda_env"

# Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y

# Activate environment
source activate "$ENV_PATH"

# Install uv
pip install uv

# Define requirement paths
COMMON_REQ="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
AG_REQ="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/autogluon.tabular/requirements.txt"

# Install base dependencies and ML library
# uv pip install handles dependencies efficiently
uv pip install numpy pandas scikit-learn
uv pip install -r "$AG_REQ" --prerelease=allow -r "$COMMON_REQ"

# Verification: Ensure autogluon.tabular is present without forcing reinstall
python -c "import autogluon.tabular; print('AutoGluon installed successfully')"