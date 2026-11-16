#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
benchmark_python="${AGEML_BENCH_PYTHON:-$PWD/.venv/bin/python}"
prefix="${1:-$PWD/.mlzero-local}"
prefix="$(cd "$prefix" && pwd)"
exec 9>>.hamlet-cv.lock
if ! flock -n 9; then
  echo 'Campaign active: stop the launcher with Ctrl+C before updating.' >&2; exit 1
fi
if [[ -n "$(systemctl list-units --state=running --no-legend 'mlzero-cv-*.scope')" ]]; then
  echo 'A local search scope is still active; stop it before updating.' >&2; exit 1
fi
export PATH="$prefix/env/bin:$prefix/conda/bin:$PATH"
export HF_HOME="$prefix/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$PWD/environments/mlzero/search_seed:$PWD"
export AGEML_SEARCH_SEED=42 PYTHONHASHSEED=42
if [[ -e "$prefix/runtime.json" ]]; then
  if [[ -e "$prefix/runtime-before-factory-v4.json" ]]; then
    echo 'Factory interface already installed; use the factory-v4 configurations.' >&2; exit 1
  fi
  # Verify the old recorded source and environment before applying the new patch.
  "$benchmark_python" - "$prefix" <<'PY'
import json,sys,subprocess
from pathlib import Path
from ageml_mlzero_benchmark import sha256_file
prefix=Path(sys.argv[1]);old=json.loads((prefix/'runtime.json').read_text())
for name,expected in old['source_hashes'].items():
    if sha256_file(prefix/'source'/name)!=expected: raise RuntimeError('Source changed: '+name)
freeze=subprocess.check_output([str(prefix/'env/bin/python'),'-m','pip','freeze'],text=True)
if sorted(freeze.splitlines())!=old['freeze']: raise RuntimeError('Runtime packages changed; do not mix cohorts')
PY
  mv -- "$prefix/runtime.json" "$prefix/runtime-before-factory-v4.json"
fi
patch="$PWD/patches/0004-normalize-factory-prompts.patch"
if git -C "$prefix/source" apply --reverse --check "$patch" 2>/dev/null; then
  echo 'Factory interface patch already applied; finishing validation.'
else
  git -C "$prefix/source" apply --check "$patch"
  git -C "$prefix/source" apply --index "$patch"
fi
sha256sum "$PWD"/patches/*.patch > "$prefix/source-patches.sha256"
python environments/mlzero/check_retrieval.py
python environments/mlzero/check_cv_gate.py --source "$prefix/source"
"$benchmark_python" -m ageml_mlzero_benchmark.local_runtime "$prefix"
echo 'Factory interface repair complete. Prepare a NEW factory-v4 cohort.'
