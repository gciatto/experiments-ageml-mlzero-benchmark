# MLZero + Gemini 2.5 Flash: full evidence audit

Read analysis_report.md. All final outcomes are preserved; no model was rerun or imported.
Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib.

Extract the supplied artifacts ZIP into a separate INPUT directory. ARTIFACTS below is the path to its artifacts/ folder.
From this bundle directory, run:

```bash
python scripts/audit_mlzero.py \
  --campaign ARTIFACTS/campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5 \
  --reference-dir reference \
  --paper-reference reference/paper-analysis-reference.json \
  --outdir regenerated
python scripts/audit_evidence.py --artifacts ARTIFACTS --outdir regenerated
python scripts/plot_full_audit.py --outdir regenerated
```

Replace ARTIFACTS with the extracted folder path. The audit reads .gz files directly.
The raw archive and dataset CSVs are not duplicated here. Diagnostic paths are relative to ARTIFACTS; base static-audit paths are relative to the canonical campaign.

Plots are provided as PNG and vector PDF. CSV labels and report text are in English.
