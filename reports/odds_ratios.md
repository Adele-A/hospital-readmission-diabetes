# Logistic-regression odds ratios (Stage 5)

Top 25 features by absolute coefficient from the CV-selected L2 logistic regression, fit on the full training split. Numeric features are standardised, so their odds ratio is **per 1 standard deviation**; categorical features are one-hot encoded (odds ratio vs the implicit reference level).

**These are adjusted associations, not causal effects.** They describe how features co-vary with 30-day readmission in this dataset after mutual adjustment; they do not license any causal or clinical claim.

| feature | coefficient | odds ratio |
| --- | ---: | ---: |
| cat__discharge_disposition_id_22 | 0.6195 | 1.858 |
| cat__discharge_disposition_id_1 | -0.4276 | 0.652 |
| num__number_inpatient | 0.3288 | 1.389 |
| cat__discharge_disposition_id_5 | 0.3231 | 1.381 |
| cat__discharge_disposition_id_6 | -0.2331 | 0.792 |
| cat__diag_2_group_neoplasms | 0.2065 | 1.229 |
| cat__discharge_disposition_id_25 | -0.1807 | 0.835 |
| cat__medical_specialty_Cardiology | -0.1705 | 0.843 |
| cat__medical_specialty_Nephrology | 0.1655 | 1.180 |
| cat__diag_3_group_neoplasms | 0.1574 | 1.170 |
| cat__medical_specialty_Orthopedics-Reconstructive | -0.1544 | 0.857 |
| cat__age_infrequent_sklearn | -0.1542 | 0.857 |
| cat__payer_code_Missing | 0.1339 | 1.143 |
| cat__discharge_disposition_id_18 | -0.1322 | 0.876 |
| cat__diag_2_group_respiratory | -0.1263 | 0.881 |
| cat__discharge_disposition_id_2 | 0.1130 | 1.120 |
| cat__diag_1_group_diabetes | 0.1114 | 1.118 |
| cat__admission_source_id_17 | -0.1045 | 0.901 |
| cat__A1Cresult_None | 0.1031 | 1.109 |
| cat__insulin_Down | 0.0993 | 1.104 |
| cat__diag_1_group_circulatory | 0.0990 | 1.104 |
| cat__diag_1_group_respiratory | -0.0990 | 0.906 |
| cat__admission_source_id_1 | 0.0979 | 1.103 |
| cat__age_[50-60) | -0.0917 | 0.912 |
| cat__glipizide_infrequent_sklearn | 0.0915 | 1.096 |
