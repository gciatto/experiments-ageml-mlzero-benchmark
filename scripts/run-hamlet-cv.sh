#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
benchmark_python="${AGEML_BENCH_PYTHON:-$PWD/.venv/bin/python}"
if [[ ! -x "$benchmark_python" ]]; then
  echo 'Create .venv and install this repository first.' >&2
  exit 1
fi
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  if [[ ! -t 0 ]]; then
    echo 'Export GEMINI_API_KEY or launch this script in an interactive tmux terminal.' >&2
    exit 1
  fi
  read -r -s -p 'Gemini API key: ' GEMINI_API_KEY
  echo
  export GEMINI_API_KEY
fi
if [[ "$#" -eq 0 ]]; then
  set -- configs/gemini25-hamlet-cv.yml configs/gemini31-hamlet-cv.yml
fi
run_options=()
if [[ "${AGEML_BENCH_PILOT:-0}" == 1 ]]; then
  run_options=(--only adult --only-repetition 1)
fi
mkdir -p logs
for campaign_config in "$@"; do
  campaign_label="$(basename "$campaign_config" .yml)"
  "$benchmark_python" -m ageml_mlzero_benchmark.hamlet_cv doctor \
    --config "$campaign_config" --check-llm
  "$benchmark_python" -m ageml_mlzero_benchmark.hamlet_cv run \
    --config "$campaign_config" "${run_options[@]}" \
    2>&1 | tee "logs/${campaign_label}-$(date +%Y%m%d-%H%M%S).log"
  "$benchmark_python" -m ageml_mlzero_benchmark.hamlet_cv report \
    --config "$campaign_config"
done
