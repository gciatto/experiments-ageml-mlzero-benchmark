# Manuscript discrepancies

These are factual differences between the manuscript and the code-defined
experiment. In every case the code-defined experiment is kept and the paper is
updated.

## Dataset statistics

These were checked on 2026-10-20 against the OpenML bytes, exported as AGE-ML
exports them (`ageml-bench prepare`).

| Dataset | Manuscript | Actual (OpenML export) | Note |
|---|---|---|---|
| Ames Housing (42165) | 2,930 rows | **1,460 rows, 80 features** | OpenML 42165 is `house_prices` (the Kaggle subset), not De Cock's full Ames data. The features include the `Id` row identifier, and 19 columns have missing values. |
| California Housing (43939) | 8 features | **9 features** | Includes the categorical `ocean_proximity`, and `total_bedrooms` has missing values. The target `median_house_value` is in dollars (14,999–500,001), unlike scikit-learn's 8-feature loader, which uses units of $100k. |
| Auto MPG (196) | 6 numeric / 1 categorical | **7 / 0 after CSV round-trip** | `origin` is read back as an integer, and the target column is named `class` (mpg, 9.0–46.6). One feature has missing values. |

The dataset manifests (`artifacts/datasets/<key>/<split>/manifest.json`) record
the exact schema, missing values and checksums.

## Methodology

- **Test-selected winner.** At `86e4af6`, the global pipeline winner is chosen by
  test score (`get_best_run_across_pipelines`). The paper must describe the
  corrected `ageml-holdout-v2` protocol and report only results produced under it.
  Historical numbers must not be presented as held-out estimates.
- **Target identification.** AGE-ML v1 asks an LLM to identify the target. The
  comparison uses OpenML's declared targets: `class`, `Class`,
  `median_house_value`, `SalePrice`.
- **Partition.** The effective split is about 64/16/20: an outer 80/20, then an
  inner 80/20 of the development set, both with seed 42.
- **Repetitions.** All five repetitions reuse one split, so they measure search
  stochasticity only.
- **Budget accounting.** "Equivalent Time" sums per-pipeline durations. It is
  neither wall-clock time nor CPU time. The nominal 2 h deadline wraps the pipeline
  jobs only. Eight workers do not establish the CPU count.
- **Model.** `gemini-3.1-flash-lite-preview`, the default at `86e4af6`, has been
  shut down. A rerun on `gemini-3.1-flash-lite` is a new cohort and must be
  labelled as such.

## Tables

The predictive, timing, token and cost tables must be regenerated from the final
declared campaign with `ageml-bench report`. Differences between individual
stochastic repetitions are not implementation bugs.
