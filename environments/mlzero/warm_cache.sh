#!/usr/bin/env bash
# Warm the shared uv/conda caches (mounted at /bench/cache) OUTSIDE any timed
# search, by building one throwaway environment the way MLZero's generated
# scripts do. Same UV_EXCLUDE_NEWER as the runs, so the cache holds the frozen
# versions. Usage (from the host):
#   docker run --rm -v <artifact_root>/cache:/bench/cache \
#     -e UV_CACHE_DIR=/bench/cache/uv -e CONDA_PKGS_DIRS=/bench/cache/conda \
#     ageml-mlzero:f228cd7 bash /opt/bench/warm_cache.sh
set -euo pipefail
REG=/opt/mlzero/src/autogluon/assistant/tools_registry
ENV=/bench/cache/warm-env
rm -rf "$ENV"
conda create -y -p "$ENV" python=3.11
"$ENV/bin/python" -m pip install uv
"$ENV/bin/python" -m uv pip install --python "$ENV/bin/python" --prerelease=allow \
  -r "$REG/_common/requirements.txt" -r "$REG/autogluon.tabular/requirements.txt"
rm -rf "$ENV"
