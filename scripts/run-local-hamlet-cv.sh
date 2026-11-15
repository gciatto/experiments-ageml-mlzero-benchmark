#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$#" -eq 0 ]]; then
  set -- configs/gemini25-hamlet-cv-local.yml configs/gemini31-hamlet-cv-local.yml
fi
# Shared launcher prompts securely, runs doctor, both sequential searches and reports.
exec bash scripts/run-hamlet-cv.sh "$@"
