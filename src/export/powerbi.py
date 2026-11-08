"""Stage 7 — export star-schema CSVs for the Power BI dashboard.

Writes tidy, UTF-8, dot-decimal CSVs into ``dashboard/data/``:

  fact tables:
    fact_encounters            one row per cohort encounter (dimensions + target +
                               out-of-sample predicted risk on the test split)
    fact_metrics               ROC-AUC / PR-AUC / Brier per model, with 95% CI
    fact_calibration           calibration-curve bins per model
    fact_threshold_table       operating points (chosen on train CV) on test
    fact_subgroup_performance  AUC + calibration by age / sex / race
  dimension tables:
    dim_model, dim_age_group, dim_diag_group

Predicted risk is only populated for the **test** split (out-of-sample); training
rows are left blank so no in-sample optimism leaks into the dashboard.

Run:  python -m src.export.powerbi
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import load_config
from src.data.cohort import build_cohort_table
from src.data.load import build_database, connect
from src.evaluation.bootstrap import cluster_bootstrap_ci
from src.evaluation.metrics import brier, calibration_bins, pr_auc, roc_auc
from src.evaluation.subgroups import subgroup_table
from src.evaluation.thresholds import choose_thresholds, oof_probabilities, threshold_table
from src.features.feature_defs import build_feature_frame
from src.features.icd9_groups import map_icd9
from src.models.splits import cv_splitter, holdout_split, load_cohort_ordered
from src.paths import repo_path

AGE_ORDER = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]
DIAG_GROUPS = [
    "circulatory", "respiratory", "digestive", "diabetes", "injury",
    "musculoskeletal", "genitourinary", "neoplasms", "other", "Missing",
]

_HOME = {1, 6, 8}
_SNF_ICF = {3, 4, 5, 24}
_TRANSFER = {2, 10, 15, 16, 17, 22, 23, 27, 28, 29, 30}


def _discharge_group(i) -> str:
    try:
        v = int(float(i))
    except (TypeError, ValueError):
        return "Other/Unknown"
    if v in _HOME:
        return "Home / home health"
    if v in _SNF_ICF:
        return "SNF / ICF"
    if v in _TRANSFER:
        return "Transfer / other facility"
    return "Other/Unknown"


def _count_bucket(v, cap: int) -> str:
    v = int(v)
    return str(v) if v < cap else f"{cap}+"


def build_fact_encounters(df: pd.DataFrame, test_prob: dict, high_risk_mask) -> pd.DataFrame:
    """Lean, aggregatable fact table: dimensions + target + out-of-sample risk."""
    out = pd.DataFrame(
        {
            "encounter_id": df["encounter_id"],
            "split": df["split"],
            "age_group": df["age"],
            "gender": df["gender"],
            "race": df["race"].replace("?", "Missing"),
            "admission_type_id": pd.to_numeric(df["admission_type_id"], errors="coerce"),
            "discharge_group": df["discharge_disposition_id"].map(_discharge_group),
            "diag_1_group": df["diag_1"].map(map_icd9),
            "prior_inpatient_grp": pd.to_numeric(df["number_inpatient"], errors="coerce")
            .fillna(0).map(lambda v: _count_bucket(v, 3)),
            "a1c_result": df["A1Cresult"],
            "change": df["change"],
            "diabetes_med": df["diabetesMed"],
            "insulin": df["insulin"],
            "time_in_hospital": pd.to_numeric(df["time_in_hospital"], errors="coerce"),
            "number_inpatient": pd.to_numeric(df["number_inpatient"], errors="coerce"),
            "number_emergency": pd.to_numeric(df["number_emergency"], errors="coerce"),
            "number_diagnoses": pd.to_numeric(df["number_diagnoses"], errors="coerce"),
            "readmitted_lt30": df["y"].astype(int),
        }
    )
    out["predicted_risk"] = np.nan
    out["high_risk_top20"] = pd.NA
    test_mask = (df["split"] == "test").to_numpy()
    out.loc[test_mask, "predicted_risk"] = test_prob["Gradient boosting"]
    out.loc[test_mask, "high_risk_top20"] = high_risk_mask.astype(int)
    out["predicted_risk"] = out["predicted_risk"].round(5)
    return out


def _metric_rows(models_preds: dict, y, groups, seed: int) -> pd.DataFrame:
    rows = []
    specs = [("ROC-AUC", roc_auc), ("PR-AUC", pr_auc), ("Brier", brier)]
    for model_name, p in models_preds.items():
        for metric_name, fn in specs:
            point, lo, hi = cluster_bootstrap_ci(fn, y, p, groups, seed=seed)
            rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "value": round(point, 4),
                    "ci_low": round(lo, 4),
                    "ci_high": round(hi, 4),
                    "split": "test",
                }
            )
    return pd.DataFrame(rows)


def _calibration_rows(models_preds: dict, y) -> pd.DataFrame:
    frames = []
    for model_name in ["Logistic regression", "Gradient boosting"]:
        b = calibration_bins(y, models_preds[model_name], n_bins=10)
        b.insert(0, "model", model_name)
        frames.append(b)
    out = pd.concat(frames, ignore_index=True)
    out["mean_predicted"] = out["mean_predicted"].round(5)
    out["observed_rate"] = out["observed_rate"].round(5)
    return out


def _threshold_rows(models, x_train, y_train, g_train, cv, preds, y_test) -> pd.DataFrame:
    frames = []
    for model_name in ["Gradient boosting", "Logistic regression"]:
        oof = oof_probabilities(models[model_name], x_train, y_train, g_train, cv)
        thr = choose_thresholds(oof, y_train)
        t = threshold_table(y_test, preds[model_name], thr)
        t.insert(0, "model", model_name)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def _subgroup_rows(test_df, y_test, p_hgb) -> pd.DataFrame:
    frames = []
    df = test_df.copy()
    df["race"] = df["race"].replace("?", "Missing")
    for col, label in [("age", "age_group"), ("gender", "sex"), ("race", "race")]:
        t = subgroup_table(df, y_test, p_hgb, col)
        t.insert(0, "dimension", label)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def _write(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    group_col = cfg["split"]["group_column"]

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
    df["split"] = "train"
    df.loc[df.index[test_idx], "split"] = "test"
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    x_train, numeric, categorical = build_feature_frame(train_df)
    y_train = train_df["y"].astype(int).to_numpy()
    g_train = train_df[group_col].to_numpy()
    x_test, _, _ = build_feature_frame(test_df)
    y_test = test_df["y"].astype(int).to_numpy()
    g_test = test_df[group_col].to_numpy()

    models_dir = repo_path(cfg["paths"]["processed_dir"], "models")
    if not (models_dir / "hist_gradient_boosting.joblib").exists():
        from src.models import train as train_mod

        train_mod.main()
    model_files = {
        "Baseline: prevalence": "baseline_prevalence.joblib",
        "Baseline: inpatient rule": "baseline_inpatient.joblib",
        "Logistic regression": "logistic_regression.joblib",
        "Gradient boosting": "hist_gradient_boosting.joblib",
    }
    models = {name: joblib.load(models_dir / f) for name, f in model_files.items()}
    preds = {name: m.predict_proba(x_test)[:, 1] for name, m in models.items()}

    cv = cv_splitter(cfg)
    oof_hgb = oof_probabilities(models["Gradient boosting"], x_train, y_train, g_train, cv)
    top20_thr = choose_thresholds(oof_hgb, y_train)["top20 (highest-risk 20%)"]
    high_risk_mask = preds["Gradient boosting"] >= top20_thr

    out_dir = repo_path("dashboard", "data")
    out_dir.mkdir(parents=True, exist_ok=True)

    _write(build_fact_encounters(df, preds, high_risk_mask), out_dir / "fact_encounters.csv")
    _write(_metric_rows(preds, y_test, g_test, seed), out_dir / "fact_metrics.csv")
    _write(_calibration_rows(preds, y_test), out_dir / "fact_calibration.csv")
    _write(
        _threshold_rows(models, x_train, y_train, g_train, cv, preds, y_test),
        out_dir / "fact_threshold_table.csv",
    )
    _write(
        _subgroup_rows(test_df, y_test, preds["Gradient boosting"]),
        out_dir / "fact_subgroup_performance.csv",
    )

    _write(
        pd.DataFrame({"age_group": AGE_ORDER, "sort_order": range(len(AGE_ORDER))}),
        out_dir / "dim_age_group.csv",
    )
    _write(
        pd.DataFrame({"diag_1_group": DIAG_GROUPS, "sort_order": range(len(DIAG_GROUPS))}),
        out_dir / "dim_diag_group.csv",
    )
    _write(
        pd.DataFrame(
            {
                "model": list(model_files.keys()),
                "is_baseline": [True, True, False, False],
            }
        ),
        out_dir / "dim_model.csv",
    )
    print(f"Wrote dashboard CSVs to {out_dir}")


if __name__ == "__main__":
    main()
