"""Stage 3 — exploratory analysis.

For each patient / treatment characteristic, the share of encounters followed by a
readmission within 30 days (`y == 1`) with a 95% Wilson confidence interval.

These are **descriptive, encounter-level associations, not causal effects**. The
intervals quantify sampling uncertainty only; they do not adjust for confounding
and do not account for repeated encounters of the same patient (that dependence is
handled in the modelling splits, Stage 4).

Outputs:
  - ``reports/eda_report.md``        — one table per factor.
  - ``reports/figures/eda_*.png``    — bar charts with 95% CI for key factors.

Run:  python -m src.eda.descriptive
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pandas as pd

from src.config import load_config
from src.data.cohort import build_cohort_table
from src.data.load import build_database, connect
from src.features.icd9_groups import map_icd9
from src.paths import repo_path

NUMERIC_COLS = [
    "time_in_hospital",
    "number_inpatient",
    "number_emergency",
    "number_outpatient",
    "number_diagnoses",
]

AGE_ORDER = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]
INSULIN_ORDER = ["No", "Down", "Steady", "Up"]
A1C_ORDER = ["None", "Norm", ">7", ">8"]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n (as fractions)."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (center - margin, center + margin)


def _count_bucket(x: int, top: int) -> str:
    """Bucket a non-negative count: 0,1,...,top-1, then 'top+'."""
    return str(x) if x < top else f"{top}+"


def load_cohort_df(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Pull the cohort into pandas with numeric casts and derived EDA columns."""
    df = con.execute("SELECT * FROM cohort;").df()
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Race: show "?" as an explicit Missing category (careful wording in the report).
    df["race"] = df["race"].replace("?", "Missing")

    # Primary diagnosis clinical group.
    df["diag_1_group"] = df["diag_1"].map(map_icd9)

    # Derived bins.
    df["los_group"] = pd.cut(
        df["time_in_hospital"].astype("float"),
        bins=[0, 3, 6, 9, 14],
        labels=["1-3 days", "4-6 days", "7-9 days", "10-14 days"],
    ).astype("object")
    df["number_inpatient_grp"] = df["number_inpatient"].map(lambda v: _count_bucket(int(v), 3))
    df["number_emergency_grp"] = df["number_emergency"].map(lambda v: _count_bucket(int(v), 2))
    df["number_outpatient_grp"] = df["number_outpatient"].map(lambda v: _count_bucket(int(v), 2))
    df["n_diagnoses_grp"] = pd.cut(
        df["number_diagnoses"].astype("float"),
        bins=[0, 4, 7, 8, 9, 16],
        labels=["1-4", "5-7", "8", "9", "10-16"],
    ).astype("object")

    # Admission type description via ids_mapping.
    amap = _mapping_dict(con, "admission_type_id")
    dmap = _mapping_dict(con, "discharge_disposition_id")
    df["admission_type"] = (
        df["admission_type_id"].map(lambda i: amap.get(_to_int(i), "Unknown")).fillna("Unknown")
    )
    df["discharge_disposition"] = (
        df["discharge_disposition_id"]
        .map(lambda i: dmap.get(_to_int(i), "Unknown"))
        .fillna("Unknown")
    )
    return df


def _to_int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _mapping_dict(con: duckdb.DuckDBPyConnection, mapping_type: str) -> dict[int, str]:
    rows = con.execute(
        "SELECT id, description FROM ids_mapping WHERE mapping_type = ?;", [mapping_type]
    ).fetchall()
    return {int(i): str(d) for i, d in rows}


def rate_by(
    df: pd.DataFrame,
    col: str,
    order: list[str] | None = None,
    min_n: int = 1,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Readmission-<30 rate with 95% Wilson CI by category of ``col``."""
    g = df.groupby(col, dropna=False)["y"].agg(n="size", n_pos="sum").reset_index()
    g = g[g["n"] >= min_n]
    rows = []
    for _, r in g.iterrows():
        lo, hi = wilson_ci(int(r["n_pos"]), int(r["n"]))
        rows.append(
            {
                "category": r[col],
                "n": int(r["n"]),
                "n_readmit_lt30": int(r["n_pos"]),
                "rate_pct": round(100.0 * r["n_pos"] / r["n"], 2),
                "ci95_low_pct": round(100.0 * lo, 2),
                "ci95_high_pct": round(100.0 * hi, 2),
            }
        )
    out = pd.DataFrame(rows)
    if order is not None:
        out["__o"] = out["category"].map({c: i for i, c in enumerate(order)})
        out = out.sort_values("__o", na_position="last").drop(columns="__o")
    else:
        out = out.sort_values("n", ascending=False)
    if top_n is not None:
        out = out.head(top_n)
    return out.reset_index(drop=True)


def _table_md(table: pd.DataFrame) -> str:
    head = "| category | n | n(<30) | rate % | 95% CI low | 95% CI high |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: |"
    body = [
        f"| {r.category} | {r.n:,} | {r.n_readmit_lt30:,} | {r.rate_pct} | "
        f"{r.ci95_low_pct} | {r.ci95_high_pct} |"
        for r in table.itertuples(index=False)
    ]
    return "\n".join([head, sep, *body])


def plot_rate(table: pd.DataFrame, title: str, overall_pct: float, out_path: Path) -> None:
    """Horizontal bar chart of rate with asymmetric 95% CI error bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = table.iloc[::-1]  # top category at top of the chart
    y = range(len(t))
    lo_err = (t["rate_pct"] - t["ci95_low_pct"]).clip(lower=0)
    hi_err = (t["ci95_high_pct"] - t["rate_pct"]).clip(lower=0)
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(t) + 1.5))
    ax.barh(list(y), t["rate_pct"], color="#3b5b92", height=0.6)
    ax.errorbar(
        t["rate_pct"], list(y), xerr=[lo_err, hi_err],
        fmt="none", ecolor="#222", elinewidth=1, capsize=3,
    )
    ax.axvline(overall_pct, color="#b23b3b", linestyle="--", linewidth=1,
               label=f"overall {overall_pct:.2f}%")
    ax.set_yticks(list(y))
    ax.set_yticklabels([str(c) for c in t["category"]])
    ax.set_xlabel("30-day readmission rate (%)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# (key, column, display title, order, make_figure)
FACTORS: list[tuple[str, str, str, list[str] | None, bool]] = [
    ("age", "age", "Age group (10-year bands)", AGE_ORDER, True),
    ("gender", "gender", "Sex", None, False),
    ("race", "race", "Race (reported; descriptive only)", None, True),
    ("los", "los_group", "Length of stay",
     ["1-3 days", "4-6 days", "7-9 days", "10-14 days"], False),
    ("number_inpatient", "number_inpatient_grp", "Prior inpatient visits (past year)",
     ["0", "1", "2", "3+"], True),
    ("number_emergency", "number_emergency_grp", "Prior emergency visits (past year)",
     ["0", "1", "2+"], False),
    ("number_outpatient", "number_outpatient_grp", "Prior outpatient visits (past year)",
     ["0", "1", "2+"], False),
    ("n_diagnoses", "n_diagnoses_grp", "Number of diagnoses",
     ["1-4", "5-7", "8", "9", "10-16"], False),
    ("diag_1_group", "diag_1_group", "Primary diagnosis group (ICD-9, Strack scheme)", None, True),
    ("insulin", "insulin", "Insulin", INSULIN_ORDER, False),
    ("change", "change", "Any change in diabetes medication", ["No", "Ch"], False),
    ("diabetesMed", "diabetesMed", "On any diabetes medication", ["No", "Yes"], False),
    ("a1cresult", "A1Cresult", "HbA1c result", A1C_ORDER, True),
    ("admission_type", "admission_type", "Admission type", None, False),
    ("discharge", "discharge_disposition", "Discharge disposition (top 12)", None, False),
]


def build_report(df: pd.DataFrame, figures_dir: Path) -> str:
    overall = 100.0 * df["y"].mean()
    n_enc = len(df)
    lines = [
        "# Exploratory analysis — 30-day readmission (Stage 3)",
        "",
        f"Cohort: **{n_enc:,} encounters** (from the frozen Stage 2 cohort). "
        f"Overall 30-day readmission rate: **{overall:.2f}%**.",
        "",
        "Each table shows, per category, the number of encounters `n`, the number "
        "readmitted within 30 days `n(<30)`, the rate, and a 95% Wilson confidence "
        "interval. **These are descriptive associations, not causal effects**, and "
        "do not adjust for confounding or for repeated encounters of the same patient.",
        "",
    ]
    for key, col, title, order, make_fig in FACTORS:
        top_n = 12 if key == "discharge" else None
        table = rate_by(df, col, order=order, top_n=top_n)
        lines += [f"## {title}", "", _table_md(table), ""]
        if make_fig:
            out = figures_dir / f"eda_{key}.png"
            plot_rate(table, title, overall, out)
            lines += [f"![{title}](figures/{out.name})", ""]
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
        df = load_cohort_df(con)
    finally:
        con.close()

    reports = repo_path(cfg["paths"]["reports_dir"])
    figures_dir = reports / "figures"
    report = build_report(df, figures_dir)
    (reports / "eda_report.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"Wrote {reports / 'eda_report.md'} and EDA figures.")


if __name__ == "__main__":
    main()
