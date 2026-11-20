# Plot appearance

`ageml_mlzero_benchmark.plotting` provides the same DejaVu Sans font, 10-point
labels, 13-point titles, palette and embedded TrueType PDF fonts used by AGE-ML's
`scripts/campaign_analysis.py`. Repetitions are points and means are diamonds.
The appearance is scoped to figure generation and does not modify global plotting
settings in the caller. Figures are exported as PNG (200 dpi), PDF and editable SVG.

The existing holdout report uses this style automatically. Its scores remain
holdout test scores; they must not be interpreted as the HAMLET-CV campaign scores.

## Saved Gemini 2.5 full audit

To regenerate the three figures delivered with the full Gemini 2.5 audit, use:

```bash
poetry run python scripts/plot_full_audit.py --outdir /path/to/audit/output
```

The directory must contain the existing `run_audit.csv`, `run_diagnostics.csv`,
`node_diagnostics.csv` and `dataset_summary.csv` from that audit. This script is
specific to its complete 45-run, five-repetition Gemini 2.5 dataset; its captions
describe that audited cohort. It writes `01_outcomes_and_scores`,
`02_regression_comparison` and `03_failure_diagnosis` in all three formats.

The regression plot shows absolute selection-CV RMSE, individual repetitions,
paired available results, success-only arithmetic means and artifact counts.
Unsuccessful runs remain in the artifact-count denominator and are never assigned
zero scores. These are selection-CV scores, not independent test results.

The script only reads saved tables. It does not run models, change experiment
records, regenerate predictions or replace the CV validator.
