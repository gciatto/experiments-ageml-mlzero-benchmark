#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
benchmark_python="${AGEML_BENCH_PYTHON:-$PWD/.venv/bin/python}"
age_root="${AGE_ML_ROOT:-/root/age-ml}"
runtime="${AGEML_LOCAL_RUNTIME:-$PWD/.mlzero-local/runtime.json}"
for version in 25 31; do
  if [[ "$version" == 25 ]]; then age_name=gemini25-flash-hamlet-cv-w8-storage
  else age_name=gemini31-flash-lite-hamlet-cv-w8-storage; fi
  "$benchmark_python" -m ageml_mlzero_benchmark.hamlet_cv bootstrap \
    --age-campaign "$age_root/experiments/results/$age_name" \
    --age-data "$age_root/resources/datasets" --local-runtime "$runtime" \
    --campaign-suffix factory-v5 --output "configs/gemini${version}-hamlet-cv-local-factory-v5.yml" \
    --cpus 16 --memory-gib 50
  "$benchmark_python" -m ageml_mlzero_benchmark.hamlet_cv run \
    --config "configs/gemini${version}-hamlet-cv-local-factory-v5.yml" --dry-run
done
