# Alignment review (2026-11-15)

Inspected benchmark commit: 2dc8cd09e0fc6367fb23634f88b6ce2f9d5d54ac.
Inspected current AGE-ML protocol.py, fitting.py and the committed Gemini 2.5/3.1
campaign configurations. Runtime references must come from the actual completed
campaign manifests, not from a moving main branch.

## Findings in the original benchmark

| Item | Existing configuration/implementation | Current AGE-ML cohort |
|---|---|---|
| Search budget | comparison.yml: 7200 s | 3600 s |
| Evaluation | proposed train/validation/test holdout | hamlet_cv, 10 unshuffled folds |
| Reference | October AGE-ML 86e4af6, test-selected v1 | ageml-evaluation-v3 |
| Imported score | old Best Test-set Score column | Search CV Score (selection; not independent test) |
| Fit/search seeds | search_seed None; seed 42 only in regenerated split | fit/run seed 42 |
| LLM transport | OpenAI-compatible Gemini, no seed forwarded | native Gemini, seed 42 requested |
| LLM temperature | upstream defaults often 0.1 | 0.0 |
| Resources | main unresolved; pilot 8 CPU/32 GiB | experiment container 16 CPU/64 GiB; campaign hard cap 50 GiB |
| Base image record | lock digest differed from Dockerfile | lock corrected to existing Dockerfile digest |

The existing holdout workflow is retained. A separate hamlet_cv entry point
captures exact CSV checksums, targets and folds from the actual AGE-ML campaigns.
It supplies a trusted fold evaluator and audits native-selected exports instead
of relabelling holdout scores as CV. MLZero's MCTS and search tools are unchanged;
generated code implements a fit/predict factory interface for matched evaluation.

## Checks actually completed

- 32 offline tests pass: the original 24 plus 8 matched-CV cases.
- Includes mean-fold vs pooled RMSE, fit seed normalization, OOF/factory tampering,
  bootstrap hashes and model configuration, rejection of old holdout manifests,
  simulated deadline selection and resume without rerunning completed results.
- Ruff and format checks pass on changed/new Python files; mypy passes on the
  new workflow and modified adapter. An existing NumPy typing issue in the old
  tests/conftest.py is independent of this change; no global mypy pass is claimed.
- Both instrumentation patches apply cleanly to the exact pinned MLZero commit;
  the patched modules compile.
- Native Google client 4.2.1 constructors were instantiated for both model IDs,
  checking seed=42 and temperature=0; no API request was sent.
- The pinned base image digest exists in the Docker registry (manifest HTTP 200).
- Bash launcher syntax and Python compilation pass.

## Checks that require the user's container

No Docker daemon or Gemini credentials are available in the authoring workspace.
A complete image build, nested-LXC Docker execution, retrieval-cache warming,
live model calls and a real MLZero search have NOT been validated here. The
container-side doctor checks the built image, exact patch/client versions and
native Gemini construction; --check-llm probes the real model with the requested
settings. Do not treat offline fixtures as a completed competitor experiment.

## Scientific limits retained explicitly

The one-hour allocation includes MLZero initialization; AGE-ML planning may be
outside its exploration clock. Docker's memory cap does not reproduce the
40 GiB memory.high throttle. The scorer checks retained OOF predictions and
provenance but generated Python is not isolated from the dataset as an
adversarial security boundary. Review factories before claiming complete data-use
compliance. These are selection CV scores, not independent test scores. A common
seed/model ID does not make a remote Gemini backend bitwise deterministic.
