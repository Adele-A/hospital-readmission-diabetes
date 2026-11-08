# Decision journal

Chronological log of methodological decisions, each with a short rationale. This
is a portfolio project; the point is that choices are explicit and defensible.

## Stage 0 — Plan

- **D0.1 Target.** `y = 1` iff `readmitted == "<30"`, else `0`; `">30"` and `"NO"`
  both map to `0`. The question is specifically *early* (30-day) readmission.
- **D0.2 Unit of observation.** Primary = all eligible encounters with
  patient-level splits; sensitivity analysis = first encounter per patient. Keeps
  all information while preventing patient leakage.
- **D0.3 No oversampling.** ~11% positives; keep natural prevalence and report
  calibration honestly rather than distort probabilities.
- **D0.4 Honesty constraints.** No "deployed/clinical/validated" claims; every
  README number must trace to a result file.

## Stage 1 — Load & SQL audit

- **D1.1 Missing marker.** `"?"` is the only NA value on read. `"None"` in
  `A1Cresult` / `max_glu_serum` is a real category ("test not performed"),
  confirmed in the audit (A1Cresult `None` = 84,748; max_glu_serum `None` = 96,420).
- **D1.2 Raw load as VARCHAR.** `encounters_raw` is loaded with `all_varchar=true`
  so `"?"` and `"None"` survive verbatim for auditing; typing happens in the
  modelling code later.
- **D1.3 Death/hospice exclusion codes verified.** `discharge_disposition_id` in
  {11, 13, 14, 19, 20, 21} confirmed against `IDS_mapping.csv` (11=Expired,
  13=Hospice/home, 14=Hospice/medical facility, 19/20=Expired…hospice; 21 has 0
  encounters). Total to exclude ≈ 2,423. Applied in Stage 2, not Stage 1.
- **D1.4 Missing / low-value columns (approved).** `weight` dropped (96.86%
  missing). `medical_specialty` (49.08% missing) and `payer_code` (39.56% missing)
  kept with an explicit `Missing` category and rare levels grouped. `payer_code`
  is administrative (a socioeconomic proxy) — flagged in Limitations (Variant A).
- **D1.5 Download provenance.** Primary = verified UCI zip URL; fallback =
  `ucimlrepo`. Raw data gitignored; `data/raw/MANIFEST.json` (url + sha256 + date)
  is the committed provenance record.
- **D1.6 Optional libraries skipped.** `lightgbm` has no Python 3.14 wheel →
  `HistGradientBoostingClassifier` used instead. `shap` skipped → LR coefficients
  + permutation importance for explainability.

## Stage 2 — Cohort & target

- **D2.1 Target frozen.** `y = 1` iff `readmitted == "<30"`, else `0`
  (`">30"` and `"NO"` → `0`). One target for all stages; no relabeling.
- **D2.2 Unit of observation.** Primary = all eligible encounters with
  `patient_nbr`-grouped splits; sensitivity = first encounter per patient
  (`is_first_encounter`, min `encounter_id`).
- **D2.3 Exclusions (frozen, applied in order).** (1) death/hospice discharge
  `{11,13,14,19,20,21}` → −2,423 encounters / −1,528 patients; (2) gender
  `Unknown/Invalid` → −3 encounters. **Analytic cohort = 99,340 encounters /
  69,987 patients; 11,314 positives (11.39%).**
- **D2.4 Not excluded (kept on purpose).** Transfers / "still patient"
  dispositions and `Not Mapped`/`Unknown` dispositions (25, 26) are kept — the
  30-day outcome is still defined for them. Documented in `cohort_definition.md`.
- **D2.5 Prevalence baseline.** 11.39% is the constant-probability baseline for
  Stage 5 model comparison.

## Stage 3 — Exploratory analysis

- **D3.1 ICD-9 grouping.** Built `src/features/icd9_groups.py` with the Strack et
  al. scheme (circulatory, respiratory, digestive, diabetes, injury,
  musculoskeletal, genitourinary, neoplasms, other; V/E codes → other; `?` →
  Missing). Versioned and tested (`test_icd9_groups.py`).
- **D3.2 Uncertainty = 95% Wilson interval.** Wilson score interval for every
  proportion (better small-sample behaviour than the normal approximation).
- **D3.3 Descriptive only.** All Stage-3 numbers are encounter-level associations,
  not causal, and ignore within-patient dependence (handled by splits in Stage 4).
  Stated at the top of `reports/eda_report.md`.
- **D3.4 Binning.** Counts bucketed (prior inpatient 0/1/2/3+, emergency &
  outpatient 0/1/2+), length of stay and number-of-diagnoses binned, age kept in
  the dataset's 10-year bands. `race == "?"` shown as an explicit `Missing`
  category with careful wording.
- **Key signals (descriptive):** strongest gradient is prior inpatient visits
  (8.59% at 0 → 26.40% at 3+); prior emergency visits (10.69% → 21.71%); discharge
  to psychiatric (36.7%) / rehab (27.7%); diabetes as primary diagnosis 13.1%.
  HbA1c measured vs not shows no large difference in this descriptive view.

## Stage 4 — Splits & leakage protection

- **D4.1 Patient-grouped hold-out.** `GroupShuffleSplit` by `patient_nbr`,
  test_size 0.2, seed 42. **train = 79,682 enc / 55,989 patients (11.47% pos);
  test = 19,658 enc / 13,998 patients (11.06% pos); patient overlap = 0.** Test is
  untouched until Stage 6.
- **D4.2 CV inside train.** `StratifiedGroupKFold` (5 folds, groups =
  `patient_nbr`, shuffle, seed 42). Each fold's validation positive rate ≈ 11.47%;
  fold group overlap = 0.
- **D4.3 Deterministic, not persisted.** Cohort is loaded ordered by
  `encounter_id`, so the split is reproducible from the seed without storing
  indices.
- **D4.4 Leakage-safe transforms.** Stateless row-wise features (ICD-9 groups,
  `"?"`→`Missing`, numeric cast) in `feature_defs.py`; all *learned* transforms
  (median imputation, scaling, one-hot with rare-level grouping via `min_frequency`
  and `handle_unknown="infrequent_if_exist"`) inside a `ColumnTransformer` fit on
  train folds only (`models/pipeline.py`).
- **D4.5 Never-features.** `encounter_id`, `patient_nbr`, `readmitted`, `y`,
  `is_first_encounter`, `weight`, raw `diag_1..3` excluded from `X`.
- **D4.6 Automated guards.** `tests/test_leakage.py`: zero train/test patient
  overlap, zero CV fold overlap, preprocessor fit never sees test categories,
  feature frame excludes identifiers/outcome.

## Stage 5 — Models

- **D5.1 Baselines.** (a) constant train prevalence; (b) prior-inpatient-visits
  rule (empirical rate per 0/1/2/3+ bucket). Both scored with the same CV/metrics.
- **D5.2 Logistic regression.** L2, `C` selected by patient-grouped CV (grid
  {0.01, 0.1, 1.0}; chosen `C=0.01`). Odds-ratio table in `reports/odds_ratios.md`
  with an explicit "adjusted associations, not causal" disclaimer.
- **D5.3 Gradient boosting.** `HistGradientBoostingClassifier` with early stopping;
  modest CV grid (learning_rate, max_leaf_nodes, l2). Needs **dense** one-hot input
  (LR uses sparse) — `build_preprocessor(sparse=...)` handles both.
- **D5.4 No oversampling / no class weights.** ~11% positives kept as-is so
  predicted probabilities stay calibration-ready (calibration checked on test in
  Stage 6). Thresholds NOT selected here; test set untouched.
- **D5.5 Bug fixed — `"?"`→`Missing`.** DuckDB's `.df()` returns string columns as
  pandas `StringDtype`, not `object`; the original dtype-gated replace silently
  skipped `payer_code`/`medical_specialty`. Fixed to replace unconditionally; CV
  metrics unchanged (relabeling only).
- **CV results (train only, 5-fold grouped, mean ± std):**
  prevalence ROC-AUC 0.500 / PR-AUC 0.1147; inpatient rule 0.609 / 0.165;
  logistic regression 0.663 / 0.214; gradient boosting **0.672 / 0.234**. The
  models beat the simple rule modestly — reported honestly; final hold-out
  evaluation with bootstrap CIs is Stage 6.

## Stage 6 — Hold-out evaluation

- **D6.1 Patient-cluster bootstrap.** All 95% CIs resample **patients** (clusters)
  with replacement, 1,000 replicates, preserving within-patient correlation.
- **D6.2 Thresholds set on train CV only.** Operating points (top-10% / top-20% by
  risk; ~50% sensitivity) are derived from train out-of-fold probabilities and
  applied unchanged to test — never tuned on test.
- **D6.3 Calibration = Cox intercept/slope.** Fit of outcome on logit(p);
  unpenalised logistic (`C=1e10`, since `penalty=None` is deprecated in sklearn
  >= 1.8). Models slightly over-predict (intercept ~0.17, slope ~1.10).
- **D6.4 Subgroups = robustness check, not a fairness audit.** Age/sex/race AUC +
  calibration; subgroups < 100 or single-class reported with size only. Careful
  wording; small groups are noisy.
- **D6.5 Explainability.** LR odds ratios (Stage 5) + permutation importance for
  the booster on test. Both framed as predictive/associational, not causal.
- **Headline test results (95% patient-bootstrap CI):**
  ROC-AUC — prevalence 0.500; inpatient rule 0.597 (0.583–0.611); logistic
  0.660 (0.646–0.674); **gradient boosting 0.674 (0.660–0.688)**. PR-AUC (no-skill
  0.111): logistic 0.207, boosting 0.231. ΔROC-AUC boosting − inpatient rule
  **0.077 (0.067–0.087)** (CI excludes 0); boosting − logistic 0.014 (0.009–0.019).
  First-encounter sensitivity analysis consistent (boosting ROC-AUC 0.663).
- **Honest takeaway.** The models clearly beat the no-information baseline and
  modestly beat the simple prior-inpatient-visits rule; the dominant signals are
  prior inpatient visits and discharge disposition. This is a reasonable, honest
  result for this dataset, not a clinically validated tool.

## Stage 7 — Power BI export & dashboard

- **D7.1 Star schema.** `fact_encounters` (one row per cohort encounter, lean
  aggregatable dimensions + target + out-of-sample risk) linked to `dim_model`,
  `dim_age_group`, `dim_diag_group`; plus `fact_metrics`, `fact_calibration`,
  `fact_threshold_table`, `fact_subgroup_performance`. UTF-8, dot-decimal,
  documented in `dashboard/DATA_DICTIONARY.md`.
- **D7.2 Predictions only on test.** `predicted_risk` / `high_risk_top20` populated
  for the test split only (out-of-sample); train rows blank. Risk visuals filter
  `split = test`; descriptive visuals use the full cohort.
- **D7.3 Spec, not .pbix.** `dashboard/POWERBI_SPEC.md` gives import steps, model
  relationships, DAX measures, and the four pages (cohort overview / associated
  characteristics / patient groups / model performance). Adele builds the `.pbix`
  on Windows; README says "data model + spec ready; screenshots coming soon".
- **D7.4 Cross-platform preview.** Self-contained Plotly
  `reports/dashboard_preview.html` (Plotly inlined) opens in any browser without
  Power BI.

## Stage 8 — Report, README, reproducibility

- **D8.1 Honest framing enforced.** README and `reports/model_report.md` state the
  models only modestly beat the simple rule, report CIs, and carry an explicit
  Disclaimer ("independent portfolio project, not a clinical tool; historical US
  1999–2008; no claim of reducing readmissions"). No "deployed/validated" language.
- **D8.2 Both-OS run instructions.** README gives macOS/Linux and Windows
  PowerShell venv + `python run_pipeline.py`; dashboard marked "spec ready,
  screenshots coming soon" with a TODO (no `.pbix` claimed).
- **D8.3 From-scratch reproducibility check.** `data/` wiped and
  `python run_pipeline.py` re-run end-to-end (re-download → … → export); headline
  numbers match the committed reports (fixed seeds), tests green, ruff clean.
