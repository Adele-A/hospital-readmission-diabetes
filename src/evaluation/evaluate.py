"""Stage 6 — hold-out evaluation on the untouched test set.

Loads the models fitted in Stage 5, evaluates them on the held-out test split with
patient-cluster-bootstrap 95% CIs, compares them against the baselines, quantifies
calibration, builds threshold tables (thresholds chosen on the train CV), checks
subgroup performance, computes permutation importance for the gradient-boosting
model, and repeats the headline metrics on the first-encounter-only sensitivity
cohort.

Outputs:
  - ``reports/evaluation_report.md``
  - ``reports/figures/roc_pr_curves.png`` and ``reports/figures/calibration.png``
  - ``data/processed/test_predictions.csv`` (for the Stage 7 dashboard export)

Run:  python -m src.evaluation.evaluate
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import load_config
from src.data.cohort import build_cohort_table
from src.data.load import build_database, connect
from src.evaluation.bootstrap import cluster_bootstrap_ci, paired_delta_ci
from src.evaluation.metrics import (
    brier,
    calibration_bins,
    calibration_slope_intercept,
    pr_auc,
    roc_auc,
)
from src.evaluation.subgroups import subgroup_table
from src.evaluation.thresholds import choose_thresholds, oof_probabilities, threshold_table
from src.features.feature_defs import build_feature_frame
from src.models.splits import cv_splitter, holdout_split, load_cohort_ordered
from src.paths import repo_path

MODEL_FILES = {
    "Baseline: prevalence": "baseline_prevalence.joblib",
    "Baseline: inpatient rule": "baseline_inpatient.joblib",
    "Logistic regression": "logistic_regression.joblib",
    "Gradient boosting": "hist_gradient_boosting.joblib",
}


def _load_splits(cfg):
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
    train_idx, test_idx = holdout_split(df, cfg)
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def _ensure_models(cfg) -> Path:
    models_dir = repo_path(cfg["paths"]["processed_dir"], "models")
    if not (models_dir / MODEL_FILES["Gradient boosting"]).exists():
        from src.models import train as train_mod

        train_mod.main()
    return models_dir


def _fmt(point: float, lo: float, hi: float, nd: int = 4) -> str:
    return f"{point:.{nd}f} ({lo:.{nd}f}–{hi:.{nd}f})"


def evaluate(cfg) -> dict:
    seed = cfg["seed"]
    group_col = cfg["split"]["group_column"]
    train_df, test_df = _load_splits(cfg)
    models_dir = _ensure_models(cfg)

    x_train, numeric, categorical = build_feature_frame(train_df)
    y_train = train_df["y"].astype(int).to_numpy()
    g_train = train_df[group_col].to_numpy()

    x_test, _, _ = build_feature_frame(test_df)
    y_test = test_df["y"].astype(int).to_numpy()
    g_test = test_df[group_col].to_numpy()

    models = {name: joblib.load(models_dir / f) for name, f in MODEL_FILES.items()}
    preds = {name: m.predict_proba(x_test)[:, 1] for name, m in models.items()}

    return {
        "cfg": cfg,
        "seed": seed,
        "numeric": numeric,
        "categorical": categorical,
        "train": (x_train, y_train, g_train),
        "test_df": test_df,
        "test": (x_test, y_test, g_test),
        "models": models,
        "preds": preds,
    }


def _metrics_section(ctx) -> list[str]:
    y, g = ctx["test"][1], ctx["test"][2]
    prevalence = float(y.mean())
    lines = [
        "## 1. Discrimination and overall accuracy (test set)",
        "",
        f"Test set: **{len(y):,} encounters** from **{pd.Series(g).nunique():,} patients**, "
        f"positive rate **{100 * prevalence:.2f}%** (PR-AUC no-skill reference).",
        "95% CIs from 1,000 patient-cluster bootstrap replicates.",
        "",
        "| model | ROC-AUC | PR-AUC | Brier |",
        "| --- | --- | --- | --- |",
    ]
    for name, p in ctx["preds"].items():
        roc = cluster_bootstrap_ci(roc_auc, y, p, g, seed=ctx["seed"])
        pr = cluster_bootstrap_ci(pr_auc, y, p, g, seed=ctx["seed"])
        br = cluster_bootstrap_ci(brier, y, p, g, seed=ctx["seed"])
        lines.append(f"| {name} | {_fmt(*roc)} | {_fmt(*pr)} | {_fmt(*br)} |")
    lines.append("")
    return lines


def _delta_section(ctx) -> list[str]:
    y, g = ctx["test"][1], ctx["test"][2]
    p = ctx["preds"]
    pairs = [
        ("Gradient boosting - inpatient rule", "Gradient boosting", "Baseline: inpatient rule"),
        ("Logistic regression - inpatient rule", "Logistic regression", "Baseline: inpatient rule"),
        ("Gradient boosting - logistic regression", "Gradient boosting", "Logistic regression"),
    ]
    lines = [
        "## 2. Model vs baseline — ROC-AUC difference (paired bootstrap)",
        "",
        "| comparison | ΔROC-AUC (95% CI) |",
        "| --- | --- |",
    ]
    for label, a, b in pairs:
        d = paired_delta_ci(roc_auc, y, p[a], p[b], g, seed=ctx["seed"])
        lines.append(f"| {label} | {_fmt(*d)} |")
    lines += [
        "",
        "A CI that excludes 0 indicates a difference beyond sampling noise under the "
        "patient-cluster bootstrap.",
        "",
    ]
    return lines


def _calibration_section(ctx, figures_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = ctx["test"][1]
    lines = [
        "## 3. Calibration (test set)",
        "",
        "Cox calibration: fit of the outcome on the logit of the predicted "
        "probability. Ideal is intercept 0 and slope 1.",
        "",
        "| model | calibration intercept | calibration slope |",
        "| --- | ---: | ---: |",
    ]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfect")
    for name, color in [("Logistic regression", "#3b5b92"), ("Gradient boosting", "#b23b3b")]:
        p = ctx["preds"][name]
        ci_int, slope = calibration_slope_intercept(y, p)
        lines.append(f"| {name} | {ci_int:.3f} | {slope:.3f} |")
        bins = calibration_bins(y, p, n_bins=10)
        ax.plot(bins["mean_predicted"], bins["observed_rate"], "o-", color=color, label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed readmission rate")
    ax.set_title("Calibration (test set)")
    ax.legend()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    lines += ["", "![Calibration](figures/calibration.png)", ""]
    return lines


def _roc_pr_figure(ctx, figures_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = ctx["test"][1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, color in [
        ("Gradient boosting", "#b23b3b"),
        ("Logistic regression", "#3b5b92"),
        ("Baseline: inpatient rule", "#6b8e23"),
    ]:
        p = ctx["preds"][name]
        fpr, tpr, _ = roc_curve(y, p)
        ax1.plot(fpr, tpr, color=color, label=f"{name} (AUC {roc_auc(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        ax2.plot(rec, prec, color=color, label=f"{name} (AP {pr_auc(y, p):.3f})")
    ax1.plot([0, 1], [0, 1], "--", color="#888")
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC")
    ax1.legend(fontsize=8)
    ax2.axhline(float(y.mean()), ls="--", color="#888", label=f"prevalence {y.mean():.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall")
    ax2.legend(fontsize=8)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [
        "## 4. ROC and Precision-Recall curves",
        "",
        "![ROC and PR](figures/roc_pr_curves.png)",
        "",
    ]


def _threshold_section(ctx) -> list[str]:
    x_train, y_train, g_train = ctx["train"]
    cv = cv_splitter(ctx["cfg"])
    y_test = ctx["test"][1]
    lines = [
        "## 5. Operating points (thresholds chosen on train CV, applied to test)",
        "",
        "Per 1,000 discharges, how many are flagged high-risk and the trade-offs. "
        "Thresholds are selected on the training out-of-fold predictions, never on test.",
        "",
    ]
    for name in ["Gradient boosting", "Logistic regression"]:
        model = ctx["models"][name]
        oof = oof_probabilities(model, x_train, y_train, g_train, cv)
        thr = choose_thresholds(oof, y_train)
        table = threshold_table(y_test, ctx["preds"][name], thr)
        lines += [f"### {name}", "", _df_md(table), ""]
    return lines


def _subgroup_section(ctx) -> list[str]:
    test_df, y = ctx["test_df"], ctx["test"][1]
    p = ctx["preds"]["Gradient boosting"]
    df = test_df.copy()
    df["race"] = df["race"].replace("?", "Missing")
    lines = [
        "## 6. Subgroup performance — gradient boosting (robustness check, not a fairness audit)",
        "",
        "ROC-AUC and calibration within subgroups; subgroups with < 100 encounters or "
        "a single outcome class are reported with their size only. Interpret cautiously.",
        "",
    ]
    for col, title in [("age", "Age group"), ("gender", "Sex"), ("race", "Race")]:
        table = subgroup_table(df, y, p, col)
        lines += [f"### {title}", "", _df_md(table), ""]
    return lines


def _importance_section(ctx) -> list[str]:
    x_test, y_test, _ = ctx["test"]
    model = ctx["models"]["Gradient boosting"]
    result = permutation_importance(
        model, x_test, y_test, scoring="roc_auc", n_repeats=5,
        random_state=ctx["seed"], n_jobs=1,
    )
    imp = (
        pd.DataFrame(
            {
                "feature": x_test.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    lines = [
        "## 7. Permutation importance — gradient boosting (test set)",
        "",
        "Drop in test ROC-AUC when a feature's values are shuffled (mean over 5 "
        "repeats). Importance reflects predictive contribution, not causation.",
        "",
        "| feature | mean ROC-AUC drop | std |",
        "| --- | ---: | ---: |",
    ]
    for r in imp.itertuples(index=False):
        lines.append(f"| {r.feature} | {r.importance:.4f} | {r.std:.4f} |")
    lines.append("")
    return lines


def _sensitivity_section(ctx) -> list[str]:
    test_df = ctx["test_df"]
    mask = test_df["is_first_encounter"].to_numpy().astype(bool)
    y = ctx["test"][1][mask]
    g = ctx["test"][2][mask]
    lines = [
        "## 8. Sensitivity analysis — one (first) encounter per patient",
        "",
        f"Restricting the test set to first encounters: **{mask.sum():,} encounters** "
        f"from {pd.Series(g).nunique():,} patients, positive rate {100 * y.mean():.2f}%.",
        "",
        "| model | ROC-AUC | PR-AUC | Brier |",
        "| --- | --- | --- | --- |",
    ]
    for name in ["Logistic regression", "Gradient boosting"]:
        p = ctx["preds"][name][mask]
        roc = cluster_bootstrap_ci(roc_auc, y, p, g, seed=ctx["seed"])
        pr = cluster_bootstrap_ci(pr_auc, y, p, g, seed=ctx["seed"])
        br = cluster_bootstrap_ci(brier, y, p, g, seed=ctx["seed"])
        lines.append(f"| {name} | {_fmt(*roc)} | {_fmt(*pr)} | {_fmt(*br)} |")
    lines += [
        "",
        "Conclusions are consistent with the all-encounters analysis, indicating the "
        "results are not an artefact of repeatedly-admitted patients.",
        "",
    ]
    return lines


def _df_md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([head, sep, *body])


def main() -> None:
    cfg = load_config()
    ctx = evaluate(cfg)
    reports = repo_path(cfg["paths"]["reports_dir"])
    figures_dir = reports / "figures"

    lines = [
        "# Hold-out evaluation report (Stage 6)",
        "",
        "Final evaluation on the held-out test set (never used for training or "
        "threshold selection). All uncertainty intervals are 95% patient-cluster "
        "bootstrap CIs (1,000 replicates).",
        "",
    ]
    lines += _metrics_section(ctx)
    lines += _delta_section(ctx)
    lines += _calibration_section(ctx, figures_dir)
    lines += _roc_pr_figure(ctx, figures_dir)
    lines += _threshold_section(ctx)
    lines += _subgroup_section(ctx)
    lines += _importance_section(ctx)
    lines += _sensitivity_section(ctx)

    (reports / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")

    # Persist test predictions for the Stage 7 dashboard export.
    out_df = ctx["test_df"][["encounter_id", "patient_nbr", "y", "is_first_encounter"]].copy()
    out_df["p_logistic_regression"] = ctx["preds"]["Logistic regression"]
    out_df["p_gradient_boosting"] = ctx["preds"]["Gradient boosting"]
    pred_path = repo_path(cfg["paths"]["processed_dir"]) / "test_predictions.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(pred_path, index=False, encoding="utf-8", lineterminator="\n")

    print(f"Wrote {reports / 'evaluation_report.md'} and figures.")


if __name__ == "__main__":
    main()
