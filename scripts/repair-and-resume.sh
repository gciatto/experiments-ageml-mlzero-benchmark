#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
benchmark_python="${AGEML_BENCH_PYTHON:-$PWD/.venv/bin/python}"
if [[ ! -x "$benchmark_python" ]]; then
  echo 'Interprete .venv mancante.' >&2
  exit 1
fi
"$benchmark_python" -m unittest discover -s tests/repair -v
"$benchmark_python" scripts/campaign_layout.py organize
unset GEMINI_API_KEY AGEML_BENCH_PILOT
exec bash scripts/run-local-hamlet-cv.sh configs/gemini25-hamlet-cv-local-factory-v5.yml
