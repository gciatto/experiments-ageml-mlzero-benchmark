#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f artifacts/campaign-layout.json ]]; then
    benchmark_python="${AGEML_BENCH_PYTHON:-$PWD/.venv/bin/python}"
    mkdir -p logs
    "$benchmark_python" scripts/campaign_layout.py run "$@" \
        2>&1 | tee "logs/mlzero-campaign-$(date +%Y%m%d-%H%M%S).log"
    exit 0
fi
exec bash scripts/run-hamlet-cv.sh "$@"
