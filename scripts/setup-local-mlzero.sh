#!/usr/bin/env bash
# Dedicated local runtime; does not modify AGE-ML, the system Python or Docker.
set -euo pipefail
cd "$(dirname "$0")/.."
benchmark_python="${AGEML_BENCH_PYTHON:-$PWD/.venv/bin/python}"
prefix="${1:-$PWD/.mlzero-local}"
mkdir -p "$prefix"
prefix="$(cd "$prefix" && pwd)"
if [[ -e "$prefix/runtime.json" ]]; then
  echo "Runtime already sealed: $prefix/runtime.json (use a new prefix to reinstall)."
  exit 0
fi
if [[ ! -x "$benchmark_python" ]]; then
  echo 'Create .venv and install this repository first.' >&2; exit 1
fi
for executable in git curl sha256sum tar; do command -v "$executable" >/dev/null; done
if [[ "$(uname -m)" != x86_64 || "$(uname -s)" != Linux ]]; then
  echo 'This pinned local runtime targets Linux x86_64.' >&2; exit 1
fi
if [[ ! -x "$prefix/conda/bin/conda" ]]; then
  installer=Miniforge3-26.7.2-0-Linux-x86_64.sh
  base=https://github.com/conda-forge/miniforge/releases/download/26.7.2-0
  curl --fail --location "$base/$installer" -o "$prefix/$installer"
  curl --fail --location "$base/$installer.sha256" -o "$prefix/$installer.sha256"
  (cd "$prefix" && sha256sum -c "$installer.sha256")
  bash "$prefix/$installer" -b -p "$prefix/conda"
  rm -- "$prefix/$installer" "$prefix/$installer.sha256"
fi
export PATH="$prefix/conda/bin:$PATH"
if [[ ! -d "$prefix/source/.git" ]]; then
  git clone https://github.com/autogluon/autogluon-assistant.git "$prefix/source"
  git -C "$prefix/source" checkout f228cd7028e197750391c02d94afcb7eea8da5ff
  for patch in "$PWD"/patches/*.patch; do git -C "$prefix/source" apply --index "$patch"; done
  sha256sum "$PWD"/patches/*.patch > "$prefix/source-patches.sha256"
fi
if [[ ! -f "$prefix/source-patches.sha256" ]]; then
  echo 'Incomplete source checkout: rerun setup with a new empty runtime prefix.' >&2; exit 1
fi
sha256sum -c "$prefix/source-patches.sha256"
if [[ "$(git -C "$prefix/source" rev-parse HEAD)" != f228cd7028e197750391c02d94afcb7eea8da5ff ]]; then
  echo 'Unexpected source commit; use a new runtime prefix.' >&2; exit 1
fi
if ! git -C "$prefix/source" diff --quiet; then
  echo 'Unstaged source edits; use a new runtime prefix.' >&2; exit 1
fi
if [[ ! -x "$prefix/env/bin/python" ]]; then
  conda create -y -p "$prefix/env" python=3.11 pip
fi
export PATH="$prefix/env/bin:$PATH"
python -m pip install 'uv==0.12.21'
uv pip install --python "$prefix/env/bin/python" --exclude-newer 2026-11-15T00:00:00Z \
  -r environments/mlzero/retrieval-constraints.txt \
  'langchain-google-genai==4.2.1' opencv-python-headless -e "$prefix/source"
export HF_HOME="$prefix/hf"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-base-en-v1.5', revision='a5beb1e3e68b9ab74eb54cfd186867f64f240e1a')
PY
python environments/mlzero/check_retrieval.py --rebuild
export AGEML_MLZERO_SOURCE="$prefix/source"
export AGEML_CACHE_ROOT="$PWD/artifacts/cache"
export UV_CACHE_DIR="$AGEML_CACHE_ROOT/uv"
export CONDA_PKGS_DIRS="$AGEML_CACHE_ROOT/conda"
export UV_EXCLUDE_NEWER=2026-10-01T00:00:00Z
bash environments/mlzero/warm_cache.sh
python environments/mlzero/check_cv_gate.py --source "$prefix/source"
"$benchmark_python" -m ageml_mlzero_benchmark.local_runtime "$prefix"
