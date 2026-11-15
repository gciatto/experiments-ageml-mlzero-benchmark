# Matched campaigns: Gemini 2.5 Flash and Gemini 3.1 Flash Lite

This workflow is for the **current AGE-ML ageml-evaluation-v3 / hamlet_cv campaigns**.
Do not use configs/comparison.yml: that legacy configuration targets an obsolete
holdout proposal, uses 7200 seconds and reads the October historical reference.

## What is aligned

- Actual AGE-ML exported CSV bytes, target, row order and ten prescribed folds.
  Classification: StratifiedKFold(10, shuffle=False); regression: KFold(10,
  shuffle=False). The seed is 42; unshuffled fold assignment itself uses no RNG.
- Local Python/NumPy and supported sklearn random_state parameters: 42. The
  generated factory must also use seed 42 for wrappers and other libraries.
- Native ChatGoogleGenerativeAI 4.2.1, the client version in AGE-ML's current lock:
  gemini-2.5-flash or gemini-3.1-flash-lite; temperature 0, requested seed 42 on
  every LLM agent. No determinism guarantee is made for Google's backend.
- 3600 seconds for initialization, generation, repair and candidate evaluation.
  A candidate is evaluated by the supplied evaluator using ten fresh fits, with
  fold-local preprocessing. The score is the arithmetic mean of fold balanced
  accuracies or fold RMSEs, never pooled OOF RMSE or an independent test score.
- No full-data refit or extra test evaluation. MLZero's native MCTS, retrieval,
  tools and model choice are retained. Its ML hyperparameter search space is not
  forced to equal AGE-ML's: that is part of the competing method.
- Defaults: 16 CPU cores, 50 GiB hard memory cap, no container swap, no GPUs.
  AGE-ML's eight workers are method parallelism, not an eight-core allocation.
  Adjust these only to match the actual machine on which AGE-ML ran. Docker's
  hard limit does not reproduce AGE-ML's 40 GiB memory.high throttling threshold.

AGE-ML may plan outside its exploration deadline; this runner includes MLZero
initialization in the hour. That conservative timing difference is recorded, not
silently declared equivalent. These remain selection CV scores: no claim about
unbiased generalization or independent statistical repetitions is warranted.

## Set up on the experiment container

From a directory alongside /root/age-ml, clone this repository and apply the
provided modified-files ZIP at its root before building the image.

```bash
cd /root/experiments-ageml-mlzero-benchmark
python3 -m venv .venv
./.venv/bin/python -m pip install -e .
```

Linux Docker with cgroup v2 must already work:

```bash
docker info
stat -fc %T /sys/fs/cgroup
```

In an unprivileged Proxmox LXC, Docker may need host-side nesting/cgroup settings.
If docker info fails, ask the Proxmox administrator before proceeding. Root in
the container is not root on the Proxmox host. Do not change host firewall,
security profiles or privileged settings merely to get this benchmark running.

Build and warm retrieval caches once, outside timed runs:

```bash
docker build -f environments/mlzero/Dockerfile \
  -t ageml-mlzero:hamlet-cv-native-gemini .
mkdir -p artifacts/cache
docker run --rm -v "$PWD/artifacts/cache:/bench/cache" \
  -e UV_CACHE_DIR=/bench/cache/uv -e CONDA_PKGS_DIRS=/bench/cache/conda \
  ageml-mlzero:hamlet-cv-native-gemini bash /opt/bench/warm_cache.sh
```

The extra patch adds only a native Gemini provider; it does not alter MLZero's
MCTS rewards or search logic. The base installation pins the native Gemini
client and resolves packages no newer than 2026-11-15. Per-node environments
retain the upstream benchmark's 2026-10-01 cutoff. Image and patch hashes are
recorded; the image is checked before starting searches.

## Capture both completed AGE-ML references and prepare inputs

These commands read your existing campaign manifests and exported datasets.
They do not run AGE-ML, change its checkout or download replacement CSVs.

```bash
./.venv/bin/python -m ageml_mlzero_benchmark.hamlet_cv bootstrap \
  --age-campaign /root/age-ml/experiments/results/gemini25-flash-hamlet-cv-w8-storage \
  --age-data /root/age-ml/resources/datasets \
  --output configs/gemini25-hamlet-cv.yml --cpus 16 --memory-gib 50

./.venv/bin/python -m ageml_mlzero_benchmark.hamlet_cv bootstrap \
  --age-campaign /root/age-ml/experiments/results/gemini31-flash-lite-hamlet-cv-w8-storage \
  --age-data /root/age-ml/resources/datasets \
  --output configs/gemini31-hamlet-cv.yml --cpus 16 --memory-gib 50

./.venv/bin/python -m ageml_mlzero_benchmark.hamlet_cv run \
  --config configs/gemini25-hamlet-cv.yml --dry-run
./.venv/bin/python -m ageml_mlzero_benchmark.hamlet_cv run \
  --config configs/gemini31-hamlet-cv.yml --dry-run
```

Expect 45 rows per model: nine datasets times five repetitions. A model, source
hash, seed, fold or budget mismatch is an error; no fallback to the old split is
allowed. The frozen outputs are newly created files; existing configs are not
overwritten. Protect configurations and generated-code artifacts from secrets.

## Start in tmux

```bash
tmux new -s mlzero-gemini
bash scripts/run-hamlet-cv.sh
```

The script asks for GEMINI_API_KEY without echoing it, verifies the built image,
then makes one tiny native Gemini probe for each model using temperature 0 and
seed 42. Both campaigns run **sequentially**, so two 50 GiB containers are never
scheduled concurrently by this runner. The expected search allocation is about
90 hours, plus build/cache warming, reporting and termination overhead.

Detach with Ctrl+B then D; reconnect with tmux attach -t mlzero-gemini.

Run records, every candidate's source, native events and usage remain under
artifacts/campaigns/<campaign>/runs/. Console logs are under logs/.

## Resume and results

Re-run the same script/configs. Terminal results, including genuine failed
searches, are skipped. An abandoned search is stopped by its recorded container
name before retrying, and its partial usage is saved. An in-container timeout
also bounds searches if the SSH session or orchestration process disappears.
Changing image, patches, configuration or harness creates new run identities;
use a new campaign_id for a new cohort and do not merge its results into the old
one. Do not change models or resource limits mid-campaign.

Reports:

```bash
./.venv/bin/python -m ageml_mlzero_benchmark.hamlet_cv report \
  --config configs/gemini25-hamlet-cv.yml
./.venv/bin/python -m ageml_mlzero_benchmark.hamlet_cv report \
  --config configs/gemini31-hamlet-cv.yml
```

mlzero_runs.csv contains one logical outcome, all_attempts.csv retains charges
from interrupted attempts, and paired_cv_scores.csv joins only AGE-ML Original
of the matching dataset/repetition. The metric is labelled cv_score, and
independent test_score is always null. No re-selection uses the trusted scorer;
the native incumbent committed before the 3600-second boundary is retained.

## Compliance and practical limits

Model factories expose an unfitted fit/predict interface (wrappers are allowed).
The trusted script checks folds, fresh fits, nested sklearn seeds and scores;
the host verifies OOF row/fold identities, finite metrics and export checksums.
A completed export is required: interrupted or malformed exports are failures.
Factory code is still generated Python and can access the search dataset. This
is audit instrumentation, not a security boundary or proof against malicious
code. Review exported factories before asserting full fold-local compliance.

There is a 25 GiB pre-run free-space check, not a disk quota. Candidate models
and environments can be large; monitor storage during the first dataset. No
candidate data is automatically deleted, to preserve auditability. Do not start
other large campaigns concurrently. Native Gemini live availability and a real
MLZero search require the container-side doctor and execution; offline tests do
not establish provider or nested-Docker availability.
