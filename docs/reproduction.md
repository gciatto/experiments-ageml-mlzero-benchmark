# Reproduction

The baseline and pilots need Linux with Docker (cgroup v2) and a Gemini API key,
supplied through `$GEMINI_API_KEY`. Reports and offline evaluation need neither.

## 1. Orchestration environment

```bash
pip install -r requirements.txt      # Poetry
poetry env use python3.12
poetry install
poetry run poe test                  # offline; uses a fake mlzero CLI
```

## 2. MLZero image (once per campaign; outside every timed region)

```bash
docker build -f environments/mlzero/Dockerfile -t ageml-mlzero:f228cd7 .
docker image inspect -f '{{.Id}}' ageml-mlzero:f228cd7   # record in upstreams.lock.yml
mkdir -p artifacts/cache
docker run --rm -v "$PWD/artifacts/cache:/bench/cache" \
  -e UV_CACHE_DIR=/bench/cache/uv -e CONDA_PKGS_DIRS=/bench/cache/conda \
  ageml-mlzero:f228cd7 bash /opt/bench/warm_cache.sh
```

Inputs are pinned as follows:

- The base image is pinned by digest.
- MLZero is pinned by commit, and the patch is listed in `upstreams.lock.yml`.
- uv and Miniforge are pinned by version.
- The retrieval model is pinned by revision.
- `UV_EXCLUDE_NEWER` freezes PyPI resolution, including the per-node
  environments MLZero creates.

Per-node conda environments are kept inside each node folder, and final inference
reuses them.

## 3. Reference manifest (reads AGE-ML; never runs it)

```bash
git clone https://github.com/Mala1180/age-ml && git -C age-ml checkout <commit>
poetry run ageml-bench reference --age-source age-ml \
  --output references/age-ml/reference.json \
  --campaign-id <age-campaign-id> \
  [--protocol-manifest <ageml-v2-manifest.json>] \
  [--resources-file <measured-resources.yml>]   # cpus, memory_gib, gpus
```

`references/age-ml/reference.json` is committed for `86e4af6`. It reports
`historical_protocol`, and its resources are unknown, so `configs/comparison.yml`
will not resolve until the corrected AGE-ML campaign supplies them.

## 4. Data

```bash
poetry run ageml-bench prepare --config configs/comparison.yml \
  [--split-dir <ageml-exported-splits>] [--export-dir <ageml resources/datasets>]
```

## 5. Checks

```bash
poetry run ageml-bench doctor --config configs/pilot.yml --offline
poetry run ageml-bench doctor --config configs/pilot.yml --check-llm   # live model listing + 1 call
poetry run ageml-bench run --config configs/pilot.yml --dry-run
```

## 6. Runs

```bash
poetry run ageml-bench run --config configs/pilot.yml                 # Wilt, Auto MPG, Adult; 10 min
poetry run ageml-bench run --config configs/comparison.yml --resume   # 9 x 5 = 45 invocations, <= 90 h search
```

`--resume` works as follows:

- Completed logical runs are skipped.
- An attempt left without `record.json` is closed as `harness_interrupted`, keeping
  its partial usage, and then restarted as a new attempt.
- Changing the source, data, split, protocol, model, budget or configuration
  creates a new run identity.

## 7. AGE-ML results, evaluation, reports

```bash
poetry run ageml-bench import-age --reference references/age-ml/reference.json \
  --results <age-ml>/experiments/results/<model>/specification/general --campaign ageml-mlzero-main
poetry run ageml-bench evaluate --campaign ageml-mlzero-main   # re-scores saved predictions
poetry run ageml-bench report   --campaign ageml-mlzero-main
```

Outputs go to `artifacts/campaigns/<campaign>/report/`:

- `runs.csv`
- `comparison.csv`
- `pairs.csv`
- `table_*.tex`
- `test_*.pdf/svg`
- `trajectories.pdf/svg`
- `provenance.md`

## Patch rationale (`patches/0001-ageml-bench-instrumentation.patch`)

Every hook does nothing unless `AGEML_BENCH_DIR` is set.

1. **Gemini through the OpenAI transport.**
   - Model discovery uses the same `OPENAI_BASE_URL` as chat.
   - Discovery drops the OpenAI-only prefix filter and strips the `models/` prefix.
   - `top_p` and `reasoning_effort` are forwarded; upstream drops them.
2. **Accounting.**
   - Each completed call appends its raw `usage_metadata` and reported model to
     `usage.jsonl`, with fsync.
   - Each chat creation logs its effective settings.
3. **Effective configuration.** The configuration after CLI overrides is dumped to
   `effective_config.yaml`.
4. **Incumbent commits.**
   - After each `create_best_run_copy()`, `incumbent.json` is replaced atomically,
     using the same selection rule.
   - `step` and `search_stopped` events are logged.

Search, rewards, tool retrieval and repair are unchanged.
