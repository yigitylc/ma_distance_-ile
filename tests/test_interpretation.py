import numpy as np
import pandas as pd
import pytest

from ma_distance_lab.interpretation import (
    RunMetadata,
    _classify_edge_pp,
    _classify_skew,
    _decimal_edge_to_pp,
    _fmt_pct,
    _fmt_pp,
    _verdict_from_edges,
    build_bottom_report,
    interpret_distance_distribution,
    interpret_edge_bars,
    interpret_event_details,
    interpret_event_summary,
    interpret_event_vs_normal,
    interpret_focus_diagnostics,
    interpret_forward_distributions,
    interpret_latest_snapshot,
    interpret_price_chart,
    interpret_run_summary,
)


def _meta() -> RunMetadata:
    return RunMetadata(
        ticker="QQQ",
        interval="1d",
        period="max",
        bars_loaded=5000,
        first_date="1999-03-10",
        last_date="2026-05-10",
        price_source="adj_close",
        ma_type="EMA",
        focus_length=50,
        rolling_window=500,
    )


def _snapshot(focus_length: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "period": 50,
                "% dev": 8.0,
                "rolling z": 2.4,
                "rolling %ile": 97.0,
                "rolling tail": 3.0,
                "rolling signal": "↑↑",
                "expanding %ile": 80.0,
                "expanding tail": 20.0,
                "expanding signal": "↑",
            },
            {
                "period": 100,
                "% dev": 5.0,
                "rolling z": 1.6,
                "rolling %ile": 92.0,
                "rolling tail": 8.0,
                "rolling signal": "↑",
                "expanding %ile": 70.0,
                "expanding tail": 30.0,
                "expanding signal": "↑",
            },
            {
                "period": 200,
                "% dev": 2.0,
                "rolling z": 0.7,
                "rolling %ile": 90.0,
                "rolling tail": 10.0,
                "rolling signal": "↑",
                "expanding %ile": 65.0,
                "expanding tail": 35.0,
                "expanding signal": "—",
            },
        ]
    )


def _event_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"horizon": 5, "N": 30, "avg": 0.001, "median": 0.0, "% positive": 50.0, "% negative": 50.0, "worst": -0.05, "best": 0.05},
            {"horizon": 10, "N": 30, "avg": 0.003, "median": 0.002, "% positive": 55.0, "% negative": 45.0, "worst": -0.07, "best": 0.08},
            {"horizon": 21, "N": 28, "avg": 0.012, "median": 0.010, "% positive": 60.0, "% negative": 40.0, "worst": -0.10, "best": 0.20},
            {"horizon": 63, "N": 25, "avg": 0.045, "median": 0.040, "% positive": 70.0, "% negative": 30.0, "worst": -0.15, "best": 0.30},
        ]
    )


def _baseline_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"horizon": 5, "N": 4500, "avg": 0.0010, "median": 0.0008, "% positive": 53.0, "% negative": 47.0, "worst": -0.20, "best": 0.18, "std": 0.02, "skew": -0.3, "q05": -0.04, "q25": -0.01, "q75": 0.01, "q95": 0.04},
            {"horizon": 10, "N": 4490, "avg": 0.0020, "median": 0.0015, "% positive": 54.0, "% negative": 46.0, "worst": -0.25, "best": 0.20, "std": 0.03, "skew": -0.4, "q05": -0.06, "q25": -0.02, "q75": 0.02, "q95": 0.06},
            {"horizon": 21, "N": 4470, "avg": 0.0050, "median": 0.0040, "% positive": 56.0, "% negative": 44.0, "worst": -0.35, "best": 0.30, "std": 0.05, "skew": -0.4, "q05": -0.10, "q25": -0.03, "q75": 0.04, "q95": 0.09},
            {"horizon": 63, "N": 4400, "avg": 0.0150, "median": 0.0120, "% positive": 60.0, "% negative": 40.0, "worst": -0.45, "best": 0.50, "std": 0.10, "skew": -0.2, "q05": -0.18, "q25": -0.04, "q75": 0.07, "q95": 0.18},
        ]
    )


def _comparison() -> pd.DataFrame:
    e = _event_summary().set_index("horizon")
    b = _baseline_summary().set_index("horizon")
    rows = []
    for h in (5, 10, 21, 63):
        rows.append(
            {
                "horizon": h,
                "Event N": e.at[h, "N"],
                "Normal N": b.at[h, "N"],
                "Event Avg": e.at[h, "avg"],
                "Normal Avg": b.at[h, "avg"],
                "Avg Edge": e.at[h, "avg"] - b.at[h, "avg"],
                "Event Median": e.at[h, "median"],
                "Normal Median": b.at[h, "median"],
                "Median Edge": e.at[h, "median"] - b.at[h, "median"],
                "Event % Positive": e.at[h, "% positive"],
                "Normal % Positive": b.at[h, "% positive"],
                "Hit-Rate Edge": e.at[h, "% positive"] - b.at[h, "% positive"],
                "Event Worst": e.at[h, "worst"],
                "Normal Worst": b.at[h, "worst"],
                "Event Best": e.at[h, "best"],
                "Normal Best": b.at[h, "best"],
            }
        )
    return pd.DataFrame(rows)


def _event_returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="W")
    return pd.DataFrame(
        {
            "fwd_5": rng.normal(0.001, 0.02, n),
            "fwd_10": rng.normal(0.003, 0.03, n),
            "fwd_21": rng.normal(0.012, 0.05, n),
            "fwd_63": rng.normal(0.045, 0.10, n),
            "event_direction": ["upside"] * 20 + ["downside"] * 10,
            "event_percentile": rng.uniform(95, 100, n),
            "event_dev_pct": rng.uniform(5, 15, n),
        },
        index=pd.Index(idx, name="event_date"),
    )


# ---- formatting + classification primitives ---------------------------------


def test_fmt_pct_returns_signed_string():
    assert _fmt_pct(0.0483) == "+4.83%"
    assert _fmt_pct(-0.0483) == "-4.83%"
    assert _fmt_pct(None) == "—"
    assert _fmt_pct(float("nan")) == "—"


def test_fmt_pp_handles_nan():
    assert _fmt_pp(1.5) == "+1.50 pp"
    assert _fmt_pp(float("nan")) == "—"


def test_classify_edge_thresholds():
    assert _classify_edge_pp(0.05) == "economically small / close to normal"
    assert _classify_edge_pp(-0.09) == "economically small / close to normal"
    assert _classify_edge_pp(0.30) == "mild"
    assert _classify_edge_pp(-1.20) == "meaningful"
    assert _classify_edge_pp(2.50) == "strong"
    assert _classify_edge_pp(float("nan")) == "unknown"


def test_decimal_edge_to_pp():
    assert _decimal_edge_to_pp(0.0048) == pytest.approx(0.48)
    assert pd.isna(_decimal_edge_to_pp(float("nan")))


def test_classify_skew_buckets():
    assert _classify_skew(1.0) == "right-skewed"
    assert _classify_skew(-1.0) == "left-skewed"
    assert _classify_skew(0.1) == "roughly balanced / mild skew"
    assert _classify_skew(float("nan")) == "skew unavailable"


def test_verdict_from_edges_all_branches():
    assert "outperformed" in _verdict_from_edges(1.0, 0.5)
    assert "outliers" in _verdict_from_edges(1.0, -0.5)
    assert "tail" in _verdict_from_edges(-0.5, 1.0)
    assert "underperformed" in _verdict_from_edges(-0.5, -0.5)
    assert "Insufficient" in _verdict_from_edges(float("nan"), 1.0)


# ---- per-section interpretations return strings -----------------------------


def test_interpret_run_summary_returns_string():
    out = interpret_run_summary(_meta())
    assert isinstance(out, str)
    assert "QQQ" in out
    assert "EMA 50" in out


def test_interpret_latest_snapshot_handles_empty():
    assert "No latest-snapshot" in interpret_latest_snapshot(pd.DataFrame(), 50)


def test_interpret_latest_snapshot_with_data():
    out = interpret_latest_snapshot(_snapshot(), 50)
    assert isinstance(out, str)
    assert "upside extension" in out or "downside extension" in out
    assert "broad across the trend ladder" in out  # 3 high-percentile rows


def test_interpret_price_chart_includes_counts():
    out = interpret_price_chart(_meta(), event_count=12, upside_count=8, downside_count=4)
    assert "12" in out and "8" in out and "4" in out


def test_interpret_focus_diagnostics_handles_empty():
    assert "No diagnostic" in interpret_focus_diagnostics(pd.DataFrame(), 50)


def test_interpret_focus_diagnostics_classifies_tail():
    df = pd.DataFrame({"dev_pct": [5, 6], "roll_z": [1.5, 2.4], "roll_pctile": [95.0, 98.0], "roll_tail": [2.0, 1.5]})
    out = interpret_focus_diagnostics(df, 50)
    assert "Very rare" in out


def test_interpret_distance_distribution_handles_empty():
    out = interpret_distance_distribution(pd.Series(dtype=float))
    assert "No deviation" in out


def test_interpret_distance_distribution_basic():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(0, 5, 1000).tolist() + [25.0])  # right-skewed via outlier
    out = interpret_distance_distribution(s)
    assert "Skew" in out
    assert "%" in out


def test_interpret_event_summary_empty():
    assert "No events" in interpret_event_summary(pd.DataFrame(), event_count=0)


def test_interpret_event_summary_full():
    out = interpret_event_summary(_event_summary(), event_count=30)
    assert "5 bars" in out and "63 bars" in out
    assert "%pos" in out


def test_interpret_event_summary_with_optional_mfe_mdd_columns():
    df = _event_summary().copy()
    df["mfe_avg"] = [0.02, 0.03, 0.05, 0.08]
    df["mdd_avg"] = [-0.02, -0.03, -0.04, -0.06]
    out = interpret_event_summary(df, event_count=30)
    assert "MFE avg" in out
    assert "MDD avg" in out


def test_interpret_event_vs_normal_returns_classified_lines():
    out = interpret_event_vs_normal(_comparison())
    assert isinstance(out, str)
    assert "edge" in out
    assert "63 bars" in out


def test_interpret_event_vs_normal_handles_empty():
    assert "unavailable" in interpret_event_vs_normal(pd.DataFrame())


def test_interpret_edge_bars_picks_strongest_horizon():
    out = interpret_edge_bars(_comparison())
    assert "Strongest" in out
    assert "63-bar" in out


def test_interpret_forward_distributions_handles_empty():
    assert "No event-return" in interpret_forward_distributions(pd.DataFrame())


def test_interpret_forward_distributions_lists_horizons():
    out = interpret_forward_distributions(_event_returns())
    for col in ("fwd_5", "fwd_10", "fwd_21", "fwd_63"):
        assert col in out


def test_interpret_event_details_handles_empty():
    assert "No events" in interpret_event_details(pd.DataFrame())


def test_interpret_event_details_summarizes_direction_split():
    out = interpret_event_details(_event_returns())
    assert "upside" in out
    assert "downside" in out
    assert "Largest winner" in out


def test_build_bottom_report_minimal_inputs():
    report = build_bottom_report(
        meta=_meta(),
        snapshot=None,
        focus_dev_series=None,
        event_summary=None,
        comparison=None,
        event_returns=None,
        event_count=0,
        direction="upside",
    )
    for header in (
        "Current State",
        "Rarity / Tail Risk",
        "Event Study Findings",
        "Event vs Normal Baseline Findings",
        "Distribution / Skew Findings",
        "Practical Interpretation",
        "Limitations",
        "Recommended Next Tests",
    ):
        assert header in report


def test_build_bottom_report_full_inputs():
    s = pd.Series(np.random.default_rng(2).normal(0, 5, 1000))
    report = build_bottom_report(
        meta=_meta(),
        snapshot=_snapshot(),
        focus_dev_series=s,
        event_summary=_event_summary(),
        comparison=_comparison(),
        event_returns=_event_returns(),
        event_count=30,
        direction="both",
    )
    assert "Direction = **both**" in report
    assert "Recommended Next Tests" in report
    assert "yfinance" in report  # limitations


def test_no_exceptions_when_only_required_event_columns_present():
    """Optional MFE/MDD/quantile columns absent — should not raise anywhere."""
    summary = _event_summary()
    out_summary = interpret_event_summary(summary, event_count=30)
    out_vs = interpret_event_vs_normal(_comparison())
    out_edge = interpret_edge_bars(_comparison())
    for blob in (out_summary, out_vs, out_edge):
        assert isinstance(blob, str) and len(blob) > 0
