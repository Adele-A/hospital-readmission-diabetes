"""Build a self-contained Plotly preview of the dashboard (no Power BI needed).

Reads the exported CSVs in ``dashboard/data/`` and writes a single, standalone
``reports/dashboard_preview.html`` (Plotly inlined) that opens in any browser on
any OS — the cross-platform stand-in until the ``.pbix`` and screenshots exist.

Run:  python -m src.export.dashboard_preview
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from src.config import load_config
from src.paths import repo_path

PRIMARY = "#3b5b92"
ACCENT = "#b23b3b"
AGE_ORDER = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (center - margin, center + margin)


def _rate(df: pd.DataFrame, col: str, order=None, sort_desc=False) -> pd.DataFrame:
    g = df.groupby(col)["readmitted_lt30"].agg(n="size", pos="sum").reset_index()
    g["rate"] = 100 * g["pos"] / g["n"]
    ci = g.apply(lambda r: _wilson(int(r["pos"]), int(r["n"])), axis=1)
    g["lo"] = [100 * c[0] for c in ci]
    g["hi"] = [100 * c[1] for c in ci]
    if order is not None:
        g["__o"] = g[col].map({c: i for i, c in enumerate(order)})
        g = g.sort_values("__o")
    elif sort_desc:
        g = g.sort_values("rate", ascending=False)
    return g


def _rate_bar(df, col, title, order=None, sort_desc=False, overall=None) -> go.Figure:
    g = _rate(df, col, order=order, sort_desc=sort_desc)
    fig = go.Figure()
    fig.add_bar(
        x=g[col].astype(str),
        y=g["rate"],
        marker_color=PRIMARY,
        error_y=dict(type="data", symmetric=False,
                     array=g["hi"] - g["rate"], arrayminus=g["rate"] - g["lo"]),
    )
    if overall is not None:
        fig.add_hline(y=overall, line_dash="dash", line_color=ACCENT,
                      annotation_text=f"overall {overall:.1f}%")
    fig.update_layout(title=title, yaxis_title="30-day readmission rate (%)",
                      template="plotly_white", height=380, margin=dict(t=50, b=40))
    return fig


def _metrics_fig(fact_metrics: pd.DataFrame) -> go.Figure:
    m = fact_metrics[fact_metrics["metric"] == "ROC-AUC"].copy()
    fig = go.Figure()
    fig.add_bar(
        x=m["model"], y=m["value"], marker_color=PRIMARY,
        error_y=dict(type="data", symmetric=False,
                     array=m["ci_high"] - m["value"], arrayminus=m["value"] - m["ci_low"]),
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#888",
                  annotation_text="no skill 0.5")
    fig.update_layout(title="Test ROC-AUC by model (95% patient-bootstrap CI)",
                      yaxis_title="ROC-AUC", template="plotly_white", height=380,
                      yaxis_range=[0.45, 0.75], margin=dict(t=50, b=80))
    return fig


def _calibration_fig(fact_cal: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#888"),
                    name="perfect")
    for model, color in [("Logistic regression", PRIMARY), ("Gradient boosting", ACCENT)]:
        sub = fact_cal[fact_cal["model"] == model]
        fig.add_scatter(x=sub["mean_predicted"], y=sub["observed_rate"], mode="lines+markers",
                        name=model, line=dict(color=color))
    fig.update_layout(title="Calibration (test set)", xaxis_title="Mean predicted probability",
                      yaxis_title="Observed rate", template="plotly_white", height=420,
                      margin=dict(t=50))
    return fig


def _threshold_fig(fact_thr: pd.DataFrame) -> go.Figure:
    t = fact_thr[fact_thr["model"] == "Gradient boosting"]
    fig = go.Figure()
    fig.add_bar(x=t["operating_point"], y=t["sensitivity"], name="Sensitivity",
                marker_color=PRIMARY)
    fig.add_bar(x=t["operating_point"], y=t["ppv"], name="PPV", marker_color=ACCENT)
    fig.update_layout(title="Gradient boosting — operating points (test)",
                      barmode="group", yaxis_title="rate", template="plotly_white",
                      height=380, margin=dict(t=50))
    return fig


def main() -> None:
    cfg = load_config()
    data = repo_path("dashboard", "data")
    enc = pd.read_csv(data / "fact_encounters.csv")
    metrics = pd.read_csv(data / "fact_metrics.csv")
    cal = pd.read_csv(data / "fact_calibration.csv")
    thr = pd.read_csv(data / "fact_threshold_table.csv")

    overall = 100 * enc["readmitted_lt30"].mean()
    n_enc = len(enc)
    n_pat = enc["encounter_id"].nunique()

    figs = [
        _rate_bar(enc, "prior_inpatient_grp", "Readmission rate by prior inpatient visits",
                  order=["0", "1", "2", "3+"], overall=overall),
        _rate_bar(enc, "age_group", "Readmission rate by age group",
                  order=AGE_ORDER, overall=overall),
        _rate_bar(enc, "diag_1_group", "Readmission rate by primary diagnosis group",
                  sort_desc=True, overall=overall),
        _metrics_fig(metrics),
        _calibration_fig(cal),
        _threshold_fig(thr),
    ]

    blocks = []
    for i, fig in enumerate(figs):
        blocks.append(
            pio.to_html(fig, include_plotlyjs=(i == 0), full_html=False)
        )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Readmission dashboard preview</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 0 auto; max-width: 1000px;
        padding: 16px; color: #1a1a1a; }}
 h1 {{ font-size: 1.5rem; }} .kpis {{ display: flex; gap: 24px; flex-wrap: wrap; }}
 .kpi {{ background: #eef3fb; border-radius: 8px; padding: 12px 20px; }}
 .kpi b {{ font-size: 1.6rem; display: block; color: {PRIMARY}; }}
 .note {{ color: #555; font-size: 0.9rem; }}
</style></head><body>
<h1>30-day readmission — dashboard preview</h1>
<p class="note">Independent portfolio project · UCI Diabetes 130-US (1999–2008) ·
 not a clinical tool. Descriptive charts use the full cohort; model charts use the
 held-out test set. Cross-platform stand-in for the Power BI dashboard.</p>
<div class="kpis">
 <div class="kpi"><b>{n_enc:,}</b>encounters</div>
 <div class="kpi"><b>{n_pat:,}</b>encounter ids</div>
 <div class="kpi"><b>{overall:.2f}%</b>overall readmission rate</div>
</div>
{"".join(f'<section>{b}</section>' for b in blocks)}
</body></html>"""

    out = repo_path(cfg["paths"]["reports_dir"], "dashboard_preview.html")
    out.write_text(html, encoding="utf-8", newline="\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
