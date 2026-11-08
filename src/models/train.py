"""Stage 5 — train and cross-validate the models (train split only).

Fits and compares, using the patient-grouped CV from Stage 4 (never touching the
held-out test set):

  1. Baselines: constant train prevalence, and the prior-inpatient-visits rule.
  2. Logistic regression (L2), ``C`` chosen by CV. Odds-ratio table exported.
  3. HistGradientBoostingClassifier, modest CV tuning with early stopping.

Class imbalance (~11% positives) is handled by **not** oversampling and **not**
using class weights, so the predicted probabilities stay calibrated (calibration
is quantified in Stage 6). Thresholds are NOT selected here and the test set is
untouched.

Outputs:
  - ``reports/model_cv_report.md``  — CV ROC-AUC / PR-AUC per model + chosen params.
  - ``reports/odds_ratios.md``      — logistic-regression odds ratios (associations).
  - ``data/processed/models/*.joblib`` — fitted models for Stage 6 (gitignored).

Run:  python -m src.models.train
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.data.cohort import build_cohort_table
from src.data.load import build_database, connect
from src.features.feature_defs import build_feature_frame
from src.models.baselines import InpatientRateClassifier, PrevalenceClassifier
from src.models.pipeline import build_preprocessor
from src.models.splits import cv_splitter, holdout_split, load_cohort_ordered
from src.paths import repo_path

SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}


def load_train(cfg: dict):
    """Return (X_train, y_train, groups_train, numeric, categorical). Test untouched."""
    con = connect(cfg)
    try:
        if not con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'cohort';"
        ).fetchone()[0]:
            build_database(con, cfg)
            build_cohort_table(con, cfg)
        df = load_cohort_ordered(con)
    finally:
        con.close()

    train_idx, _ = holdout_split(df, cfg)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    groups = train_df[cfg["split"]["group_column"]].to_numpy()
    y = train_df["y"].astype(int).to_numpy()
    x, numeric, categorical = build_feature_frame(train_df)
    return x, y, groups, numeric, categorical


def _cv_scores(estimator, x, y, groups, cv) -> dict:
    res = cross_validate(estimator, x, y, groups=groups, cv=cv, scoring=SCORING, n_jobs=1)
    return {
        "roc_auc_mean": float(np.mean(res["test_roc_auc"])),
        "roc_auc_std": float(np.std(res["test_roc_auc"])),
        "pr_auc_mean": float(np.mean(res["test_pr_auc"])),
        "pr_auc_std": float(np.std(res["test_pr_auc"])),
    }


def _grid_best(grid: GridSearchCV) -> dict:
    i = grid.best_index_
    r = grid.cv_results_
    return {
        "roc_auc_mean": float(r["mean_test_roc_auc"][i]),
        "roc_auc_std": float(r["std_test_roc_auc"][i]),
        "pr_auc_mean": float(r["mean_test_pr_auc"][i]),
        "pr_auc_std": float(r["std_test_pr_auc"][i]),
    }


def tune_logistic(x, y, groups, cv, numeric, categorical) -> tuple[Pipeline, dict, float]:
    pipe = Pipeline(
        [
            ("pre", build_preprocessor(numeric, categorical, scale=True)),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    grid = GridSearchCV(
        pipe,
        {"clf__C": [0.01, 0.1, 1.0]},
        scoring=SCORING,
        refit="roc_auc",
        cv=cv,
        n_jobs=1,
    )
    grid.fit(x, y, groups=groups)
    return grid.best_estimator_, _grid_best(grid), grid.best_params_["clf__C"]


def tune_hgb(x, y, groups, cv, numeric, categorical, seed: int) -> tuple[Pipeline, dict, dict]:
    pipe = Pipeline(
        [
            ("pre", build_preprocessor(numeric, categorical, scale=False, sparse=False)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    early_stopping=True, validation_fraction=0.1,
                    max_iter=300, random_state=seed,
                ),
            ),
        ]
    )
    grid = GridSearchCV(
        pipe,
        {
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_leaf_nodes": [31, 63],
            "clf__l2_regularization": [0.0, 1.0],
        },
        scoring=SCORING,
        refit="roc_auc",
        cv=cv,
        n_jobs=1,
    )
    grid.fit(x, y, groups=groups)
    return grid.best_estimator_, _grid_best(grid), grid.best_params_


def odds_ratio_table(lr_pipeline: Pipeline, top_n: int = 25) -> pd.DataFrame:
    pre = lr_pipeline.named_steps["pre"]
    clf = lr_pipeline.named_steps["clf"]
    names = pre.get_feature_names_out()
    coefs = clf.coef_[0]
    df = pd.DataFrame(
        {
            "feature": names,
            "coef": coefs,
            "odds_ratio": np.exp(coefs),
        }
    )
    df["abs_coef"] = df["coef"].abs()
    return df.sort_values("abs_coef", ascending=False).head(top_n).drop(columns="abs_coef")


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    x, y, groups, numeric, categorical = load_train(cfg)
    cv = cv_splitter(cfg)
    prevalence = float(y.mean())

    print("Cross-validating baselines...")
    prev_cv = _cv_scores(PrevalenceClassifier(), x, y, groups, cv)
    inpt_cv = _cv_scores(InpatientRateClassifier(), x, y, groups, cv)

    print("Tuning logistic regression...")
    lr_model, lr_cv, best_c = tune_logistic(x, y, groups, cv, numeric, categorical)

    print("Tuning gradient boosting...")
    hgb_model, hgb_cv, hgb_params = tune_hgb(x, y, groups, cv, numeric, categorical, seed)

    # Persist fitted models for Stage 6 (gitignored).
    import joblib

    models_dir = repo_path(cfg["paths"]["processed_dir"], "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    prev_model = PrevalenceClassifier().fit(x, y)
    inpt_model = InpatientRateClassifier().fit(x, y)
    joblib.dump(prev_model, models_dir / "baseline_prevalence.joblib")
    joblib.dump(inpt_model, models_dir / "baseline_inpatient.joblib")
    joblib.dump(lr_model, models_dir / "logistic_regression.joblib")
    joblib.dump(hgb_model, models_dir / "hist_gradient_boosting.joblib")

    _write_cv_report(cfg, prevalence, prev_cv, inpt_cv, lr_cv, best_c, hgb_cv, hgb_params)
    _write_or_report(cfg, lr_model)
    print("Stage 5 complete. Reports written.")


def _row(name: str, m: dict) -> str:
    return (
        f"| {name} | {m['roc_auc_mean']:.4f} ± {m['roc_auc_std']:.4f} | "
        f"{m['pr_auc_mean']:.4f} ± {m['pr_auc_std']:.4f} |"
    )


def _write_cv_report(cfg, prevalence, prev_cv, inpt_cv, lr_cv, best_c, hgb_cv, hgb_params):
    lines = [
        "# Model cross-validation report (Stage 5)",
        "",
        "Patient-grouped 5-fold CV on the **training split only** "
        "(`StratifiedGroupKFold`, seed from config). The held-out test set is not "
        "used here; thresholds are not selected here. Scores are mean ± std across folds.",
        "",
        f"Train prevalence (positive rate): **{100 * prevalence:.2f}%** — this is also "
        f"the PR-AUC reference for a no-skill classifier.",
        "",
        "| model | ROC-AUC (CV) | PR-AUC (CV) |",
        "| --- | --- | --- |",
        _row("Baseline: constant prevalence", prev_cv),
        _row("Baseline: prior-inpatient-visits rule", inpt_cv),
        _row(f"Logistic regression (L2, C={best_c})", lr_cv),
        _row("HistGradientBoosting", hgb_cv),
        "",
        "Notes:",
        "- The constant-prevalence baseline has ROC-AUC ≈ 0.50 by construction; its "
        "PR-AUC equals the prevalence.",
        "- No oversampling and no class weights were used, so predicted probabilities "
        "remain calibration-ready (calibration is evaluated on test in Stage 6).",
        f"- Gradient-boosting hyper-parameters chosen by CV: `{hgb_params}`.",
        "",
        "Interpretation is deferred to Stage 6 (hold-out evaluation with bootstrap CIs). "
        "If the models only modestly beat the prior-inpatient-visits rule, that is "
        "reported honestly.",
        "",
    ]
    out = repo_path(cfg["paths"]["reports_dir"], "model_cv_report.md")
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _write_or_report(cfg, lr_model):
    table = odds_ratio_table(lr_model)
    lines = [
        "# Logistic-regression odds ratios (Stage 5)",
        "",
        "Top 25 features by absolute coefficient from the CV-selected L2 logistic "
        "regression, fit on the full training split. Numeric features are standardised, "
        "so their odds ratio is **per 1 standard deviation**; categorical features are "
        "one-hot encoded (odds ratio vs the implicit reference level).",
        "",
        "**These are adjusted associations, not causal effects.** They describe how "
        "features co-vary with 30-day readmission in this dataset after mutual "
        "adjustment; they do not license any causal or clinical claim.",
        "",
        "| feature | coefficient | odds ratio |",
        "| --- | ---: | ---: |",
    ]
    for r in table.itertuples(index=False):
        lines.append(f"| {r.feature} | {r.coef:.4f} | {r.odds_ratio:.3f} |")
    lines.append("")
    out = repo_path(cfg["paths"]["reports_dir"], "odds_ratios.md")
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
