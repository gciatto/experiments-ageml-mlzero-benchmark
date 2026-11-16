> Existing campaigns: use [the mandatory CV gate update](cv-gate-v3.md) before running matched CV. The earlier README-only export request was insufficient.

# Run matched Gemini campaigns directly inside the Proxmox container

Docker and access to the Proxmox host shell are not required. Everything below
runs as root **inside CT 123**, alongside /root/age-ml. Systemd/cgroup v2 resource
delegation must work; the runner probes it before starting a timed search.

Apply this modified-files ZIP at the benchmark repository root. It does not
include or overwrite the deployment workflow. Do not run the Docker setup steps.

## 1. Install the dedicated MLZero environment

The orchestration .venv uses the container's Python 3.12. The separate MLZero
runtime and generated ML environments use Python 3.11, matching the original
benchmark image/cache setup. Do not install MLZero into the AGE-ML environment.

```bash
cd /root/experiments-ageml-mlzero-benchmark
apt-get update
apt-get install -y git curl ca-certificates build-essential libgomp1

# The .venv from the earlier installation can be reused.
./.venv/bin/python -m pip install -e .

tmux new -s mlzero-setup
```

Inside that tmux session:

```bash
cd /root/experiments-ageml-mlzero-benchmark
set -o pipefail
bash scripts/setup-local-mlzero.sh 2>&1 | tee setup-local-mlzero.log
```

Setup downloads Miniforge 26.7.2-0 and verifies the installer checksum, checks out
MLZero f228cd7028e197750391c02d94afcb7eea8da5ff, applies both benchmark patches,
creates .mlzero-local/env, installs uv 0.12.21 and native Gemini client 4.2.1,
warms the pinned embedding model and retrieval indices, and warms node caches.
Base dependencies use the 2026-11-15 cutoff; node caches/search environments
retain the original 2026-10-01 cutoff. Setup is outside the search budget.

It records actual source hashes, patch hashes, Python version and installed
package freeze in .mlzero-local/runtime.json. The runtime directory is ignored
by Git. Installation can download several GB and take a while. It does not
change the system Python, the AGE-ML checkout or the Docker installation.
A completed setup is reused. For a failed partially patched source checkout,
use a new empty prefix argument, rather than mixing old and new patches. Use
that prefix's runtime.json in the following commands.

Detach with Ctrl+B, D. Reconnect with tmux attach -t mlzero-setup. Wait for setup
to complete successfully before generating campaign configurations.

## 2. Create local campaign configurations from the existing AGE-ML manifests

Use NEW configuration paths. Existing frozen campaign files are never overwritten.
These commands preserve the exact CSV bytes, row order, targets and ten folds.
Local README copies contain real absolute evaluator paths; Docker inputs are
left intact. Local campaign IDs end with -local, keeping the execution cohorts
separate.

```bash
cd /root/experiments-ageml-mlzero-benchmark

./.venv/bin/python -m ageml_mlzero_benchmark.hamlet_cv bootstrap \
  --age-campaign /root/age-ml/experiments/results/gemini25-flash-hamlet-cv-w8-storage \
  --age-data /root/age-ml/resources/datasets \
  --local-runtime /root/experiments-ageml-mlzero-benchmark/.mlzero-local/runtime.json \
  --output configs/gemini25-hamlet-cv-local.yml \
  --cpus 16 --memory-gib 50

./.venv/bin/python -m ageml_mlzero_benchmark.hamlet_cv bootstrap \
  --age-campaign /root/age-ml/experiments/results/gemini31-flash-lite-hamlet-cv-w8-storage \
  --age-data /root/age-ml/resources/datasets \
  --local-runtime /root/experiments-ageml-mlzero-benchmark/.mlzero-local/runtime.json \
  --output configs/gemini31-hamlet-cv-local.yml \
  --cpus 16 --memory-gib 50

./.venv/bin/python -m ageml_mlzero_benchmark.hamlet_cv run \
  --config configs/gemini25-hamlet-cv-local.yml --dry-run
./.venv/bin/python -m ageml_mlzero_benchmark.hamlet_cv run \
  --config configs/gemini31-hamlet-cv-local.yml --dry-run
```

Expect 45 invocations per model, seed 42, temperature 0, ten folds and 3600 s.

## 3. Start both campaigns sequentially

```bash
tmux new -s mlzero-local
```

Inside tmux:

```bash
cd /root/experiments-ageml-mlzero-benchmark
bash scripts/run-local-hamlet-cv.sh
```

The script prompts for GEMINI_API_KEY without displaying it; an existing
GEMINI_API_KEY is reused. It checks the actual local native client and makes a
small live probe per model. No key is embedded in commands/configs. Gemini 2.5
runs first, then 3.1; total nominal search allocation is 90 hours.

Each dataset/repetition starts a fresh mlzero-cv-*.scope with:

- CPUQuota=1600% (16 CPU equivalents, not 16 workers).
- MemoryHigh=40G, MemoryMax=50G, MemorySwapMax=0.
- One-hour search timeout; up to 30 s termination grace. A separate systemd
  RuntimeMaxSec=3630s bounds the scope if the orchestration process disappears.
- Local Python/NumPy/sklearn seed 42 and native Gemini seed 42, temperature zero.
- GPU visibility disabled and existing pinned model/cache downloads reused offline.

Initialization/generation/repair/evaluation count inside the hour. Dataset
preparation, setup and doctor are outside it. Native MCTS/search and candidate
model choices remain unchanged. The evaluator uses ten fresh fits and computes
mean-fold balanced accuracy or RMSE; these are selection CV scores, not test
scores. Local execution changes the environment boundary, not the scoring rule.

A local run executes generated code under the same container user (root here),
without Docker's read-only input mounts. This is suitable only for the dedicated
experiment container and trusted benchmark workflow; do not run unrelated jobs
concurrently. Provenance hashes detect input/runtime changes; they do not make
generated Python a security sandbox or prove absence of leakage.

## Monitor, resume and report

```bash
tmux attach -t mlzero-local
systemctl list-units 'mlzero-cv-*.scope'
tail -n 40 logs/gemini25-hamlet-cv-local-*.log
df -h /
```

Detach using Ctrl+B, D. Do not interrupt the active search to inspect its output.
If the campaign exits, rerun the same launcher in tmux: terminal outcomes are
skipped; incomplete attempts retain usage and are retried after their recorded
scope has been stopped. An entire scope, including all nested generated-code
processes, is stopped after search completion or timeout. Genuine method failures
remain outcomes and are not automatically repeated until success.

```bash
./.venv/bin/python -m ageml_mlzero_benchmark.hamlet_cv report \
  --config configs/gemini25-hamlet-cv-local.yml
./.venv/bin/python -m ageml_mlzero_benchmark.hamlet_cv report \
  --config configs/gemini31-hamlet-cv-local.yml
```

Artifacts remain under artifacts/campaigns/<campaign-id>/runs; each attempt
retains generated code, events, usage, identity and records. Reports include
mlzero_runs.csv, all_attempts.csv and paired_cv_scores.csv. The free-space guard
requires at least 25 GiB before a new invocation; it is not a disk quota.

## Validation boundary

Offline tests cover local argv/resource/seed configuration, runtime-drift
rejection, separate input preparation, deadline selection and resume skipping
for both executors. Shell syntax, Python lint and typing have been checked.
No full Miniforge/dependency installation, systemd scope on CT 123, or live
MLZero/Gemini search has been executed in the authoring environment. The local
preflight must succeed on the experiment container before starting searches.
