# Model cross-validation report (Stage 5)

Patient-grouped 5-fold CV on the **training split only** (`StratifiedGroupKFold`, seed from config). The held-out test set is not used here; thresholds are not selected here. Scores are mean ± std across folds.

Train prevalence (positive rate): **11.47%** — this is also the PR-AUC reference for a no-skill classifier.

| model | ROC-AUC (CV) | PR-AUC (CV) |
| --- | --- | --- |
| Baseline: constant prevalence | 0.5000 ± 0.0000 | 0.1147 ± 0.0000 |
| Baseline: prior-inpatient-visits rule | 0.6087 ± 0.0019 | 0.1649 ± 0.0024 |
| Logistic regression (L2, C=0.01) | 0.6626 ± 0.0040 | 0.2143 ± 0.0056 |
| HistGradientBoosting | 0.6722 ± 0.0035 | 0.2340 ± 0.0058 |

Notes:
- The constant-prevalence baseline has ROC-AUC ≈ 0.50 by construction; its PR-AUC equals the prevalence.
- No oversampling and no class weights were used, so predicted probabilities remain calibration-ready (calibration is evaluated on test in Stage 6).
- Gradient-boosting hyper-parameters chosen by CV: `{'clf__l2_regularization': 1.0, 'clf__learning_rate': 0.05, 'clf__max_leaf_nodes': 31}`.

Interpretation is deferred to Stage 6 (hold-out evaluation with bootstrap CIs). If the models only modestly beat the prior-inpatient-visits rule, that is reported honestly.
