#!/bin/bash
set -e

# Define environment path
ENV_PATH="/root/experiments-ageml-mlzero-benchmark/artifacts/campaigns/mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5/runs/texture/rep-1/406275f8e7879a518f80/attempt-1/mlzero/node_1/conda_env"

# Create conda environment
conda create --prefix "$ENV_PATH" python=3.11 -y
source activate "$ENV_PATH"

# Install uv
pip install uv

# Install core dependencies
uv pip install numpy pandas scikit-learn

# Install requirements from registry
COMMON_REQ="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/_common/requirements.txt"
ML_REQ="/root/experiments-ageml-mlzero-benchmark/.mlzero-local/source/src/autogluon/assistant/tools_registry/machine learning/requirements.txt"

uv pip install -r "$ML_REQ" --prerelease=allow -r "$COMMON_REQ"

# Verify installation of required libraries for the factory module
python -c "import numpy; import sklearn.ensemble; import sklearn.pipeline; import sklearn.preprocessing; import sklearn.base; print('Dependencies verified successfully.')"