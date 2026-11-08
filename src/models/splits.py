"""Stage 4 — patient-level train/test split and cross-validation.

Rows are encounters but one patient can appear in several of them, so every split
is grouped by ``patient_nbr`` to prevent a patient's own encounters from landing
on both sides of a split (which would leak information).

  - Hold-out: a single ``GroupShuffleSplit`` (≈80/20, seed from config). The test
    set is set aside and not touched until final evaluation (Stage 6).
  - Cross-validation inside train: ``StratifiedGroupKFold`` (5 folds), stratified
    on the target, grouped by patient.

Splits are deterministic given the config seed and a stable row order (the cohort
is loaded ordered by ``encounter_id``), so later stages reproduce the same test set
without persisting indices.

Run:  python -m src.models.splits   (writes reports/split_summary.md)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from src.config import load_config
from src.data.cohort import build_cohort_table
from src.data.load import build_database, connect
from src.paths import repo_path


def load_cohort_ordered(con) -> pd.DataFrame:
    """Load the cohort table ordered by encounter_id (stable order for splits)."""
    return con.execute(
        "SELECT * FROM cohort ORDER BY CAST(encounter_id AS BIGINT);"
    ).df()


def holdout_split(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return positional (train_idx, test_idx) from a patient-grouped 80/20 split."""
    group_col = cfg["split"]["group_column"]
    gss = GroupShuffleSplit(
        n_splits=1, test_size=cfg["split"]["test_size"], random_state=cfg["seed"]
    )
    train_idx, test_idx = next(gss.split(df, df["y"], groups=df[group_col]))
    return train_idx, test_idx


def cv_splitter(cfg: dict) -> StratifiedGroupKFold:
    """StratifiedGroupKFold for hyper-parameter / threshold selection inside train."""
    return StratifiedGroupKFold(
        n_splits=cfg["split"]["n_cv_folds"], shuffle=True, random_state=cfg["seed"]
    )


def _stats(df: pd.DataFrame, idx: np.ndarray, group_col: str) -> dict:
    sub = df.iloc[idx]
    return {
        "encounters": len(sub),
        "patients": sub[group_col].nunique(),
        "positives": int(sub["y"].sum()),
        "positive_pct": round(100.0 * sub["y"].mean(), 2),
    }


def _split_row(name: str, s: dict) -> str:
    return (
        f"| {name} | {s['encounters']:,} | {s['patients']:,} | "
        f"{s['positives']:,} | {s['positive_pct']} |"
    )


def build_summary(df: pd.DataFrame, cfg: dict) -> str:
    group_col = cfg["split"]["group_column"]
    train_idx, test_idx = holdout_split(df, cfg)
    tr, te = _stats(df, train_idx, group_col), _stats(df, test_idx, group_col)

    train_patients = set(df.iloc[train_idx][group_col])
    test_patients = set(df.iloc[test_idx][group_col])
    overlap = len(train_patients & test_patients)

    lines = [
        "# Train/test split and cross-validation (Stage 4)",
        "",
        f"Hold-out: `GroupShuffleSplit` by `{group_col}`, "
        f"test_size={cfg['split']['test_size']}, seed={cfg['seed']}. "
        "The test set is not touched until Stage 6.",
        "",
        "| split | encounters | patients | positives | positive % |",
        "| --- | ---: | ---: | ---: | ---: |",
        _split_row("train", tr),
        _split_row("test", te),
        "",
        f"**Patient overlap between train and test: {overlap}** (must be 0).",
        "",
        f"## Cross-validation inside train — StratifiedGroupKFold "
        f"({cfg['split']['n_cv_folds']} folds, grouped by `{group_col}`)",
        "",
        "| fold | train enc | train pat | val enc | val pat | val positive % | group overlap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    skf = cv_splitter(cfg)
    for k, (f_tr, f_va) in enumerate(
        skf.split(train_df, train_df["y"], groups=train_df[group_col]), start=1
    ):
        g_tr = set(train_df.iloc[f_tr][group_col])
        g_va = set(train_df.iloc[f_va][group_col])
        lines.append(
            f"| {k} | {len(f_tr):,} | {len(g_tr):,} | {len(f_va):,} | {len(g_va):,} | "
            f"{round(100.0 * train_df.iloc[f_va]['y'].mean(), 2)} | {len(g_tr & g_va)} |"
        )

    lines += [
        "",
        "All fold group overlaps are 0 — no patient appears in both a fold's train "
        "and validation parts. See `tests/test_leakage.py` for the automated checks.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    cfg = load_config()
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

    summary = build_summary(df, cfg)
    out = repo_path(cfg["paths"]["reports_dir"], "split_summary.md")
    out.write_text(summary, encoding="utf-8", newline="\n")
    train_idx, test_idx = holdout_split(df, cfg)
    gc = cfg["split"]["group_column"]
    overlap = len(set(df.iloc[train_idx][gc]) & set(df.iloc[test_idx][gc]))
    print(f"Wrote {out}")
    print(
        f"train: {len(train_idx):,} enc / {df.iloc[train_idx][gc].nunique():,} pat; "
        f"test: {len(test_idx):,} enc / {df.iloc[test_idx][gc].nunique():,} pat; "
        f"patient overlap: {overlap}"
    )


if __name__ == "__main__":
    main()
