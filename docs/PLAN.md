# PLAN — Hospital Readmissions (Diabetes 130-US)

Predicting 30-day readmission in patients with diabetes. Independent portfolio
project (Adele Amanzhol, GitHub `Adele-A`). The value of this project is
**methodological honesty** — correct cohort definition, no leakage, patient-level
splits, honest comparison with simple baselines, calibrated probabilities, and
clearly stated limitations — **not** a high ROC-AUC.

This plan is written at Stage 0. **No modelling or data download happens until it
is approved.** It records the staged workflow, the repository layout, the key
methodological decisions, and the open questions that need a decision before build.

---

## 1. Question & target

At the **moment of discharge**, estimate the probability that a patient with
diabetes is readmitted within 30 days.

- **Target:** `y = 1` if `readmitted == "<30"`, else `0`.
- `readmitted == ">30"` and `readmitted == "NO"` both map to `0`. Rationale:
  the clinical/operational question is specifically *early* (30-day) readmission;
  a later readmission is not the outcome of interest and is collapsed with "none".
  This is stated explicitly and is the single target used throughout.
- Not a clinical tool, no causal claims, no claim of reducing readmissions.

## 2. Data

- **Source:** UCI ML Repository, *Diabetes 130-US Hospitals for Years 1999–2008*
  (dataset id 296). License CC BY 4.0 — dataset + Strack et al. (2014) cited in README.
- **Download URL (verified reachable, HTTP 200, 2026-11-08):**
  `https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip`
  Fallback: `ucimlrepo` package. `src/data/download.py` is mandatory; it writes
  `data/raw/MANIFEST.json` (url + sha256 + download date). Raw data is gitignored
  (`data/` except `data/raw/MANIFEST.json`). If UCI is unreachable the script stops
  and asks for manual placement in `data/raw/`.
- **Files:** `diabetic_data.csv` (~101,766 rows = **encounters, not patients**),
  `IDS_mapping.csv` (decodes `admission_type_id`, `discharge_disposition_id`,
  `admission_source_id`; several blocks in one file — parsed carefully).
- **Read rule:** `keep_default_na=False, na_values=["?"]`. The string `"None"` in
  `A1Cresult` and `max_glu_serum` is a real category ("test not performed"), not missing.

## 3. Staged workflow

Each stage ends with: run tests → short Russian report (done / key numbers / next)
→ git commit. **No `git push` without explicit confirmation.**

- **Stage 0 — Plan (this document). STOP for approval.**
- **Stage 1 — Load & SQL audit.** CSV → DuckDB (`encounters_raw`, `ids_mapping`).
  SQL quality checks (`sql/02_quality_checks.sql`): duplicate `encounter_id`;
  unique `patient_nbr` count; encounters-per-patient distribution; per-column
  missingness; `readmitted` distribution; invalid values; numeric ranges.
  Output: `reports/data_quality_report.md`.
- **Stage 2 — Cohort & target.** Freeze `docs/cohort_definition.md` *before*
  modelling: inclusion/exclusion, unit of observation, target. CONSORT-style
  exclusion flow (table + figure): rows and patients dropped at each step.
- **Stage 3 — EDA.** Readmission rate (<30d) with 95% CI by: age band, sex, race
  (careful wording), length of stay, `number_inpatient`/`_emergency`/`_outpatient`,
  diagnosis count, diagnosis groups, meds (insulin, `change`, `diabetesMed`),
  `A1Cresult`, admission/discharge type. Descriptive associations, not causal.
- **Stage 4 — Splits & leakage guards.** `GroupShuffleSplit` by `patient_nbr`
  (~80/20, seed from config); test held out untouched until final eval. Inside
  train: `StratifiedGroupKFold` (5 folds, groups = `patient_nbr`). All transforms
  inside `sklearn.Pipeline`/`ColumnTransformer`, fit on train folds only. Never use
  `encounter_id`, `patient_nbr`, `readmitted` (or anything derived) as features.
  `tests/test_leakage.py`: zero train/test patient overlap; zero overlap across CV
  folds; pipeline never sees test at `fit`.
- **Stage 5 — Models.** (a) Baseline = train prevalence constant; (b) baseline =
  simple risk score on `number_inpatient`. Then Logistic Regression (L2/elastic
  net, `C` by CV) and `HistGradientBoostingClassifier` (modest CV tuning, early
  stopping). ~11% positives → **no oversampling by default**; if `class_weight`
  is used, restore calibration (`CalibratedClassifierCV` on CV inside train) and
  explain it.
- **Stage 6 — Hold-out evaluation.** ROC-AUC, PR-AUC (vs prevalence baseline),
  Brier, calibration curve + intercept/slope. Thresholds chosen on CV inside train
  (not on test): top-10% / top-20% risk + a fixed-sensitivity threshold; for each —
  sensitivity, specificity, PPV, NPV, flagged-per-1000, confusion matrix. 95% CIs
  by **cluster bootstrap over patients** (≥1000 iters). Model-vs-baseline table
  with ΔAUC + CI. Subgroup performance (age, sex, race) as a robustness check, not
  a full fairness audit. Explainability: LR coefficients (odds ratios, association
  wording); permutation importance for the booster. Sensitivity analysis: one
  (first) encounter per patient. Decision-curve analysis optional if done correctly.
- **Stage 7 — Power BI export + spec.** Star-schema CSVs in `dashboard/data/`
  (`fact_encounters`, `fact_metrics`, `fact_calibration`, `fact_threshold_table`,
  `fact_subgroup_performance`, `dim_*`), UTF-8, `.` decimal, documented in
  `dashboard/DATA_DICTIONARY.md`. `dashboard/POWERBI_SPEC.md` with DAX measures and
  4 pages (Cohort overview / Associated characteristics / Patient groups / Model
  performance). Cross-platform alternative: self-contained Plotly
  `reports/dashboard_preview.html`. Static figures in `reports/figures/`. README
  says "dashboard: data model + spec ready; screenshots coming soon" (no `.pbix` yet).
- **Stage 8 — Report, README, final check.** `reports/model_report.md`; English
  `README.md` (honest numbers with CIs, both-OS run instructions, Limitations,
  Disclaimer, citations). Final check: delete `data/`+outputs, run
  `python run_pipeline.py` from scratch, confirm reproducibility (fixed seeds),
  green tests, clean `ruff`, every README number backed by a file.

## 4. Repository layout

```
README.md · pyproject.toml · run_pipeline.py · config/config.yaml
sql/        01_load · 02_quality_checks · 03_cohort · 04_eda_views
src/data/   download · load · cohort
src/features/ icd9_groups (versioned + tested) · feature defs
src/models/ pipelines · baselines · training
src/evaluation/ metrics · calibration · thresholds · bootstrap · subgroups
src/export/ power bi tables
reports/    data_quality_report.md · model_report.md · figures/ · dashboard_preview.html
dashboard/  data/ · POWERBI_SPEC.md · DATA_DICTIONARY.md · screenshots/
docs/       PLAN.md · cohort_definition.md · decisions.md
tests/      + tests/fixtures/ (small synthetic CSV for CI)
.github/workflows/ci.yml (ubuntu+windows+macos × py3.14, ruff + pytest)
```

## 5. Key methodological decisions (recommended, pending your OK)

1. **Exclusions.** Drop encounters whose `discharge_disposition_id` means death or
   hospice — **verified against `IDS_mapping.csv` in Stage 1** before applying
   (spec expects 11, 13, 14, 19, 20, 21; I will confirm the exact codes from the
   file, not assume). Reason: no further readmission is possible/meaningful.
2. **Unit of observation.** Primary analysis = **all eligible encounters**, with
   splits by `patient_nbr` to avoid leakage. Sensitivity analysis = **first
   encounter per patient** (by `encounter_id`).
3. **`encounter_id` as time order.** Used only as an *assumption* for "first
   encounter" and prior-visit interpretation; logged in Limitations (no real dates).
4. **Class imbalance.** No oversampling. Report calibration honestly.
5. **ICD-9 grouping.** Strack et al. scheme in versioned, tested
   `src/features/icd9_groups.py` (circulatory, respiratory, digestive, diabetes,
   injury, musculoskeletal, genitourinary, neoplasms, other; V/E codes → other).
6. **Honesty.** Every README number backed by a result file. If the model only
   marginally beats the `number_inpatient` baseline, that is stated plainly.

## 6. Open questions for you (need a decision before Stage 1 build)

1. **Missing / low-value columns.** My recommendation:
   - `weight` — **drop** (~97% missing, not usable).
   - `medical_specialty`, `payer_code` — **keep** as explicit `"Missing"` category
     + group rare levels. But `payer_code` is administrative (insurance), not
     clinical, and can encode socioeconomic signal. **Option A (recommended):**
     keep `payer_code` but flag it in Limitations. **Option B:** exclude
     `payer_code` entirely to keep the model clinically interpretable. Your call.
2. **Download path.** Primary = direct verified UCI URL; fallback = `ucimlrepo`.
   OK to proceed this way? (No manual placement needed — UCI is reachable now.)
3. **Optional extras.** `lightgbm` has no Python 3.14 wheel → skipped (spec prefers
   `HistGradientBoostingClassifier` anyway). `shap` is optional; I plan to **skip
   SHAP** and use LR coefficients + permutation importance unless you want SHAP.
   OK?
4. **Repo visibility / push.** Everything stays local, committed stage-by-stage.
   I will **not** push to the `Adele-A` remote until you explicitly say so.

---

**Next step:** your «ок» (and answers to §6) → Stage 1 (download + SQL audit).
