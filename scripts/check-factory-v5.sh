#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
benchmark_python="${AGEML_BENCH_PYTHON:-$PWD/.venv/bin/python}"
runtime="${AGEML_LOCAL_RUNTIME:-$PWD/.mlzero-local/runtime.json}"
exec 9>>.hamlet-cv.lock
if ! flock -n 9; then
  echo 'Campaign active: stop the launcher before updating.' >&2; exit 1
fi
if [[ -n "$(systemctl list-units --state=running --no-legend 'mlzero-cv-*.scope')" ]]; then
  echo 'A local search scope is active: stop it before updating.' >&2; exit 1
fi
"$benchmark_python" - "$runtime" <<'PYTHON'
import subprocess
import sys
from pathlib import Path
from ageml_mlzero_benchmark.local_runtime import verify
from ageml_mlzero_benchmark.cv_gate import factory_source
runtime = verify(sys.argv[1])
# No resealing, dependency changes, or repairs to generated candidates.
source = "import logging\nfrom sklearn.metrics import make_scorer, balanced_accuracy_score\nSCORER = make_scorer(balanced_accuracy_score)\nlogging.getLogger('autogluon').setLevel(logging.WARNING)\nCONFIG = dict(seed=42)\ndef build_model(x,y): return None\n"
if factory_source(source) != source:
    raise RuntimeError('The adapter must preserve ordinary Python initialization')
subprocess.run([runtime['python'], 'environments/mlzero/check_cv_gate.py', '--source', runtime['source']], check=True)
print('Factory-v5 checks passed. Prepare new factory-v5 configurations; do not resume v4 with this adapter.')
PYTHON
