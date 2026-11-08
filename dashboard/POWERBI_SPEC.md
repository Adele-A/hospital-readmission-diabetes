# Power BI dashboard — build specification

Step-by-step guide to build `readmission_dashboard.pbix` from the CSVs in
`dashboard/data/`. The analysis, models and figures do **not** depend on Power BI;
this is the final presentation layer. A cross-platform preview without Power BI is
`reports/dashboard_preview.html` (open in any browser).

> **Platform note.** Power BI Desktop runs on **Windows only**. On macOS, build it
> in a Windows VM (Parallels/UTM) or on a Windows PC. Everything else in this repo
> runs on macOS and Windows alike.

> **Honesty note.** This is an independent portfolio project on historical US data
> (1999–2008). It is not a clinical tool and makes no claim of reducing
> readmissions. Keep that framing in titles and captions.

---

## 1. Import the data

1. **Home → Get data → Text/CSV**, import all files from `dashboard/data/`:
   `fact_encounters`, `fact_metrics`, `fact_calibration`, `fact_threshold_table`,
   `fact_subgroup_performance`, `dim_model`, `dim_age_group`, `dim_diag_group`.
2. In Power Query confirm column types (set `predicted_risk`, `value`, `ci_*`,
   `mean_predicted`, `observed_rate`, rates → Decimal number; counts → Whole
   number; `readmitted_lt30`, `high_risk_top20` → Whole number). Keep locale set so
   `.` is the decimal separator. **Close & Apply.**

## 2. Model relationships (Model view)

Create these single-direction, one-to-many relationships (dimension → fact):

| from (dimension, 1) | to (fact, many) | key |
| --- | --- | --- |
| dim_age_group[age_group] | fact_encounters[age_group] | age_group |
| dim_diag_group[diag_1_group] | fact_encounters[diag_1_group] | diag_1_group |
| dim_model[model] | fact_metrics[model] | model |
| dim_model[model] | fact_threshold_table[model] | model |
| dim_model[model] | fact_calibration[model] | model |

`fact_subgroup_performance` is standalone (its own `dimension`/`subgroup`). Set
`dim_age_group` sort: **Column tools → Sort by column → sort_order** (same for
`dim_diag_group`) so bands/axes order correctly.

## 3. DAX measures (new table `_Measures`)

```DAX
Encounters = COUNTROWS ( fact_encounters )

Patients (approx) = DISTINCTCOUNT ( fact_encounters[encounter_id] )

Readmissions <30 = CALCULATE ( SUM ( fact_encounters[readmitted_lt30] ) )

Readmission Rate =
DIVIDE ( [Readmissions <30], [Encounters] )          -- format as %

Test Encounters =
CALCULATE ( [Encounters], fact_encounters[split] = "test" )

Avg Predicted Risk =
CALCULATE ( AVERAGE ( fact_encounters[predicted_risk] ),
            fact_encounters[split] = "test" )

High-Risk Flagged =
CALCULATE ( SUM ( fact_encounters[high_risk_top20] ),
            fact_encounters[split] = "test" )

Flagged per 1000 =
DIVIDE ( [High-Risk Flagged], [Test Encounters] ) * 1000

-- Threshold-table measures (use with a slicer on fact_threshold_table[operating_point])
PPV = SELECTEDVALUE ( fact_threshold_table[ppv] )
Sensitivity = SELECTEDVALUE ( fact_threshold_table[sensitivity] )
Specificity = SELECTEDVALUE ( fact_threshold_table[specificity] )

-- Metric with CI label (for cards / tooltips)
Metric Value = SELECTEDVALUE ( fact_metrics[value] )
Metric CI =
VAR lo = SELECTEDVALUE ( fact_metrics[ci_low] )
VAR hi = SELECTEDVALUE ( fact_metrics[ci_high] )
RETURN FORMAT ( lo, "0.000" ) & " – " & FORMAT ( hi, "0.000" )
```

Format `Readmission Rate`, `Avg Predicted Risk` as percentage; AUC/Brier as
`0.000`.

## 4. Pages

### Page 1 — Cohort overview
- **Title:** "Cohort overview — 30-day readmission (Diabetes 130-US, 1999–2008)".
- Cards: `Encounters`, `Patients (approx)`, `Readmission Rate`.
- **Exclusion flow:** image `reports/figures/exclusion_flow.png` (Insert → Image),
  with a short text box of the CONSORT numbers.
- Donut: `readmitted_lt30` share (0 vs 1).
- Slicer: `split`.

### Page 2 — Associated characteristics
- **Title:** "Readmission rate by characteristic (descriptive, not causal)".
- Clustered bar charts of `Readmission Rate` by: `age_group` (sorted),
  `prior_inpatient_grp`, `diag_1_group`, `a1c_result`, `discharge_group`,
  `insulin`. Add a constant line at the overall rate.
- Caption text box: "Descriptive associations with 95% CIs in
  `reports/eda_report.md`; not adjusted for confounding."

### Page 3 — Patient groups (risk)
- **Title:** "Predicted risk by patient group (test set, out-of-sample)".
- Page-level filter: `split = test`.
- `Avg Predicted Risk` by `age_group`, by `prior_inpatient_grp`, by
  `discharge_group`.
- Matrix: rows = `prior_inpatient_grp` × `age_group`, values = `Readmission Rate`
  and `Avg Predicted Risk`.
- Card: `Flagged per 1000`; slicers: `diag_1_group`, `discharge_group`.

### Page 4 — Model performance
- **Title:** "Model performance on the held-out test set".
- Slicer: `dim_model[model]`.
- Cards: `Metric Value` + `Metric CI` (filter `fact_metrics[metric]` per card:
  ROC-AUC, PR-AUC, Brier).
- **ROC / PR:** image `reports/figures/roc_pr_curves.png`.
- **Calibration:** line chart from `fact_calibration` — `observed_rate` vs
  `mean_predicted` by `model`, plus a reference diagonal (or image
  `reports/figures/calibration.png`).
- **Threshold explorer:** slicer `fact_threshold_table[operating_point]`; cards
  `Sensitivity`, `Specificity`, `PPV`; table of `tp/fp/tn/fn` and
  `flagged_per_1000`.
- **Subgroups:** table from `fact_subgroup_performance` (`dimension`, `subgroup`,
  `n`, `roc_auc`, `calib_slope`) with a caption: "robustness check, not a fairness
  audit; small subgroups are noisy."

## 5. Finishing
- Add a footer text box on every page: "Independent portfolio project · historical
  US data 1999–2008 · not a clinical tool."
- Export screenshots into `dashboard/screenshots/` and reference them from the
  README (replace the "coming soon" note once the `.pbix` and screenshots exist).
