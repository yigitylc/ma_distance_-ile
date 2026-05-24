from __future__ import annotations

import pandas as pd


def format_percent_df(df: pd.DataFrame, percent_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in percent_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "—" if pd.isna(x) else f"{x:.2%}")
    return out


def format_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["% dev", "rolling z", "rolling %ile", "rolling tail", "expanding %ile", "expanding tail"]:
        if col in out.columns:
            if "z" in col:
                out[col] = out[col].map(lambda x: "—" if pd.isna(x) else f"{x:+.2f}")
            else:
                out[col] = out[col].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}%")
    return out


def format_event_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    return_cols = ["avg", "median", "worst", "best", "mfe_avg", "mfe_best", "mdd_avg", "mdd_worst"]
    return_cols += [c for c in out.columns if c.startswith("q") and c[1:].isdigit()]
    for col in return_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "—" if pd.isna(x) else f"{x:.2%}")
    for col in ["% positive", "% negative"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}%")
    return out


def format_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format the event-vs-baseline comparison table for display.

    N columns render as integers, return columns (avg/median/edge/worst/best)
    as percentages, and hit-rate columns as `xx.xx%`.
    """
    out = df.copy()
    int_cols = ["Event N", "Normal N"]
    pct_return_cols = [
        "Event Avg",
        "Normal Avg",
        "Avg Edge",
        "Event Median",
        "Normal Median",
        "Median Edge",
        "Event Worst",
        "Normal Worst",
        "Event Best",
        "Normal Best",
    ]
    pct_pp_cols = ["Event % Positive", "Normal % Positive", "Hit-Rate Edge"]
    for col in int_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "0" if pd.isna(x) else f"{int(x)}")
    for col in pct_return_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "—" if pd.isna(x) else f"{x:.2%}")
    for col in pct_pp_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}%")
    return out


def format_event_details(df: pd.DataFrame) -> pd.DataFrame:
    """Format the enriched event-details table for display.

    event_direction is left as text. event_percentile renders with two decimals.
    event_dev_pct renders with sign and two decimals followed by %. fwd_*, mfe_*,
    and mdd_* columns render as percentages.
    """
    out = df.copy()
    if "event_date" in out.columns:
        out["event_date"] = pd.to_datetime(out["event_date"]).dt.strftime("%Y-%m-%d")
    if "event_percentile" in out.columns:
        out["event_percentile"] = out["event_percentile"].map(
            lambda x: "—" if pd.isna(x) else f"{x:.2f}"
        )
    if "event_dev_pct" in out.columns:
        out["event_dev_pct"] = out["event_dev_pct"].map(
            lambda x: "—" if pd.isna(x) else f"{x:+.2f}%"
        )
    for col in out.columns:
        if col.startswith(("fwd_", "mfe_", "mdd_")):
            out[col] = out[col].map(lambda x: "—" if pd.isna(x) else f"{x:.2%}")
    return out
