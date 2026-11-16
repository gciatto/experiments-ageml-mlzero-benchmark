#!/usr/bin/env bash
# Repair runtime dependencies and indices; never delete old search artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."
benchmark_python="${AGEML_BENCH_PYTHON:-$PWD/.venv/bin/python}"
prefix="${1:-$PWD/.mlzero-local}"
prefix="$(cd "$prefix" && pwd)"
exec 9>>.hamlet-cv.lock
if ! flock -n 9; then
  echo 'Campaign active: stop the launcher in tmux with Ctrl+C before repairing.' >&2; exit 1
fi
if [[ -n "$(systemctl list-units --state=running --no-legend 'mlzero-cv-*.scope')" ]]; then
  echo 'A local search scope is still active; stop it before repairing.' >&2; exit 1
fi
if [[ ! -x "$prefix/env/bin/python" ]]; then
  echo 'No existing runtime found; use setup-local-mlzero.sh first.' >&2; exit 1
fi
if [[ -e "$prefix/runtime.json" ]]; then
  if [[ -e "$prefix/runtime-before-retrieval-v2.json" ]]; then
    echo 'Repair already sealed; do not overwrite the previous runtime manifest.' >&2; exit 1
  fi
  mv -- "$prefix/runtime.json" "$prefix/runtime-before-retrieval-v2.json"
fi
export PATH="$prefix/env/bin:$prefix/conda/bin:$PATH"
export HF_HOME="$prefix/hf"
export CUDA_VISIBLE_DEVICES=''
python -m uv pip install --python "$prefix/env/bin/python" \
  --exclude-newer 2026-11-15T00:00:00Z \
  -r environments/mlzero/retrieval-constraints.txt \
  'langchain-google-genai==4.2.1' opencv-python-headless -e "$prefix/source"
python -m pip check
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python environments/mlzero/check_retrieval.py --rebuild
"$benchmark_python" -m ageml_mlzero_benchmark.local_runtime "$prefix"
echo 'Repair complete. Bootstrap new configs using --campaign-suffix retrieval-v2.'
