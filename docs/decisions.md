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
