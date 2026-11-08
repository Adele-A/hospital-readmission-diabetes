# CLAUDE.md — Hospital Readmissions (Diabetes 130-US)

**Authoritative task spec:** `../init-prompts/01_hospital_readmissions_diabetes.md`.
Read it in full and follow it. This file holds the durable guardrails only.

Independent portfolio project for Adele Amanzhol (GitHub `Adele-A`). Stack: Python 3.14,
DuckDB, scikit-learn. Entry point: `python run_pipeline.py`. Value = methodological honesty,
not ROC-AUC.

## Guardrails (always)
- **Language:** talk to the user in Russian; all repo content in English.
- **Honesty:** independent portfolio project; no "deployed/clinical/validated/reduced
  readmissions" claims; every README number backed by a file; never fabricate — stop and ask.
- **Stop at Stage 0:** write `docs/PLAN.md`, show it, wait for "ок" before building.
- **Git:** stage-by-stage commits after tests; `git init` if needed (branch `main`);
  **never push without explicit confirmation.**
- **Data:** `data/` gitignored except `data/raw/MANIFEST.json` (url + sha256 + date);
  `download.py` mandatory; stop if UCI is unreachable.
- **Cross-platform (Win+macOS):** `pathlib.Path` relative to repo root; explicit utf-8;
  wheels-only deps (prefer `HistGradientBoostingClassifier` over lightgbm);
  `.gitattributes eol=lf`; `lower_snake_case`; CI matrix ubuntu/windows/macos.
- **Power BI:** prepare CSVs + DAX + `POWERBI_SPEC.md` only; Adele builds `.pbix`; also emit
  a self-contained Plotly `reports/dashboard_preview.html`.

## Project-specific traps (see spec §2)
- Read CSV with `keep_default_na=False, na_values=["?"]` — `"None"` in `A1Cresult` /
  `max_glu_serum` is a real category, not missing.
- Rows are **encounters, not patients**; split by `patient_nbr` (no leakage); exclude
  death/hospice `discharge_disposition_id` (verify against `IDS_mapping.csv`).
- ICD-9 grouping lives in a versioned, tested `src/features/icd9_groups.py`.

## Environment
Use the installed Python 3.14 — all required wheels exist (verified). `lightgbm` has no 3.14
wheel, so stick with `HistGradientBoostingClassifier` (the spec already prefers it). Full
preflight: `../init-prompts/ENVIRONMENT.md`.
