"""Stage 5 — Monitoring: report scorer-metric trends across MLflow eval runs.

Stage 1 writes each evaluation's scorer aggregates (e.g. ``is_not_empty/mean``) to MLflow. This
script closes the observability loop by reading that history back out: it queries the run history
for the configured experiment, prints each scorer's value across runs, flags any metric that
dropped more than a tolerance versus the *previous* run, and (optionally) saves a trend plot.

Run it against the PERSISTENT tracking server where your evals accumulate (the default
``http://localhost:6001``), not the throwaway ``sqlite:///mlflow_ci.db`` used by the CI gate --
a time series only exists where runs pile up in one store.

Usage:
    uv run python scripts/report_trends.py

Environment:
    MLFLOW_TRACKING_URI     Overrides the tracking store (default from config: :6001).
    TRENDS_REGRESSION_TOL   Drop (vs previous run) that counts as a regression. Default 0.05.
    TRENDS_PLOT_PATH        Where to save the PNG. Default "reports/trends.png".

Exit code:
    1 if any metric regressed (so a scheduled job surfaces it as a red build), else 0.
"""

import os
import sys

import mlflow

from essential_rag.config import get_settings

METRIC_PREFIX = "metrics."
METRIC_SUFFIX = "/mean"


def load_runs(settings):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    df = mlflow.search_runs(
        experiment_names=[settings.mlflow_experiment],
        order_by=["attributes.start_time ASC"],  # oldest → newest = left-to-right on the plot
    )
    # keep only finished, top-level runs that actually logged scorer metrics:
    df = df[df["status"] == "FINISHED"]
    # child runs have a non-null parentRunId; drop them if the column exists
    if "tags.mlflow.parentRunId" in df.columns:
        df = df[df["tags.mlflow.parentRunId"].isna()]
    return df.reset_index(drop=True)


def metric_columns(df):
    return sorted(c for c in df.columns if c.startswith("metrics.") and c.endswith("/mean"))


def flag_regressions(df, metric_cols, tol):
    flags = {}
    for col in metric_cols:
        series = df[col].dropna()  # ignore runs that didn't log this metric
        if len(series) < 2:
            flags[col] = ("skip", None)  # nothing to compare against yet
            continue
        prev, latest = series.iloc[-2], series.iloc[-1]
        delta = latest - prev
        flags[col] = ("FLAG" if delta < -tol else "ok", delta)
    return flags


def print_report(df, metric_cols, flags):
    print(f"Runs analyzed: {len(df)}  (experiment: ...)")
    for col in metric_cols:
        name = col.removeprefix("metrics.")
        status, delta = flags[col]
        latest = df[col].dropna().iloc[-1] if df[col].notna().any() else float("nan")
        d = f"{delta:+.3f}" if delta is not None else "  n/a"
        print(f"  {status:4}  {name:24} latest={latest:.3f}  Δ={d}")


def plot_trends(df, metric_cols, out_path="reports/trends.png"):
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless backend — works in CI / no display
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — skipping plot; text report above is complete.)")
        return

    x = df["start_time"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in metric_cols:
        ax.plot(x, df[col], marker="o", label=col.removeprefix("metrics."))

    ax.set_title("RAG eval — scorer means across runs")
    ax.set_xlabel("run time")
    ax.set_ylabel("mean score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize="small")
    fig.autofmt_xdate()
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nTrend plot written to {out_path}")


def main() -> int:
    tol = float(os.getenv("TRENDS_REGRESSION_TOL", "0.05"))
    settings = get_settings()
    df = load_runs(settings)
    cols = metric_columns(df)
    if df.empty or not cols:
        print("No eval runs with metrics found — run `uv run essential-rag` a few times first.")
        return 0
    flags = flag_regressions(df, cols, tol)
    print_report(df, cols, flags)
    plot_trends(df, cols)
    return 1 if any(s == "FLAG" for s, _ in flags.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
