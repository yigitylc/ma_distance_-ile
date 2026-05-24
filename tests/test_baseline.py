import numpy as np
import pandas as pd

from ma_distance_lab.events import (
    EventStudyConfig,
    baseline_forward_returns,
    compare_event_vs_baseline,
    eligibility_mask,
    run_baseline_study,
    run_event_study,
    summarize_baseline_returns,
)
from ma_distance_lab.features import FeatureConfig, build_ma_distance_features


def _synthetic_features(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    price = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.015, n))), index=idx, name="price")
    df = pd.DataFrame({"price": price})
    return build_ma_distance_features(
        df, config=FeatureConfig(ma_type="EMA", lengths=(50,), rolling_window=200)
    )


def test_baseline_forward_returns_correct_for_eligible_bars():
    close = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41])
    eligible = pd.Series([True, True, True, True, True])
    out = baseline_forward_returns(close, eligible, horizons=(1, 2))
    assert round(out["fwd_1"].iloc[0], 4) == 0.1000
    assert round(out["fwd_2"].iloc[0], 4) == 0.2100
    assert round(out["fwd_1"].iloc[3], 4) == 0.1000
    assert pd.isna(out["fwd_1"].iloc[-1])
    assert pd.isna(out["fwd_2"].iloc[-1])
    assert pd.isna(out["fwd_2"].iloc[-2])


def test_baseline_excludes_ineligible_rows():
    close = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41])
    eligible = pd.Series([False, True, False, True, True])
    out = baseline_forward_returns(close, eligible, horizons=(1,))
    assert pd.isna(out["fwd_1"].iloc[0])
    assert pd.isna(out["fwd_1"].iloc[2])
    assert round(out["fwd_1"].iloc[1], 4) == 0.1000
    assert round(out["fwd_1"].iloc[3], 4) == 0.1000


def test_baseline_does_not_filter_only_event_rows():
    """Baseline N must be much larger than event N for a typical run."""
    features = _synthetic_features()
    config = EventStudyConfig(ma_type="EMA", focus_length=50, direction="upside", min_event_gap=5)
    event_summary, _, _ = run_event_study(features, config=config)
    baseline_summary, _ = run_baseline_study(features, config=config)
    for h in (5, 10, 21, 63):
        ev_n = int(event_summary.set_index("horizon").at[h, "N"])
        bl_n = int(baseline_summary.set_index("horizon").at[h, "N"])
        assert bl_n > ev_n
        assert bl_n > 100


def test_baseline_summary_has_extended_stats():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"fwd_5": rng.normal(0.001, 0.02, 500)})
    summary = summarize_baseline_returns(df, horizons=(5,))
    row = summary.iloc[0]
    for col in ("N", "avg", "median", "% positive", "% negative", "worst", "best", "std", "skew", "q05", "q25", "q75", "q95"):
        assert col in summary.columns
    assert row["N"] == 500
    assert round(row["q05"], 6) == round(float(np.quantile(df["fwd_5"], 0.05)), 6)
    assert round(row["q95"], 6) == round(float(np.quantile(df["fwd_5"], 0.95)), 6)
    assert round(row["std"], 6) == round(float(df["fwd_5"].std(ddof=1)), 6)


def test_compare_edge_columns_are_subtraction():
    event_summary = pd.DataFrame(
        [
            {"horizon": 5, "N": 10, "avg": 0.05, "median": 0.04, "% positive": 70.0, "worst": -0.03, "best": 0.08},
            {"horizon": 10, "N": 8, "avg": 0.06, "median": 0.05, "% positive": 65.0, "worst": -0.04, "best": 0.10},
        ]
    )
    baseline_summary = pd.DataFrame(
        [
            {"horizon": 5, "N": 500, "avg": 0.01, "median": 0.008, "% positive": 55.0, "worst": -0.10, "best": 0.12},
            {"horizon": 10, "N": 495, "avg": 0.02, "median": 0.015, "% positive": 56.0, "worst": -0.12, "best": 0.18},
        ]
    )
    cmp = compare_event_vs_baseline(event_summary, baseline_summary)
    row5 = cmp.set_index("horizon").loc[5]
    assert round(row5["Avg Edge"], 8) == round(0.05 - 0.01, 8)
    assert round(row5["Median Edge"], 8) == round(0.04 - 0.008, 8)
    assert round(row5["Hit-Rate Edge"], 8) == round(70.0 - 55.0, 8)
    row10 = cmp.set_index("horizon").loc[10]
    assert row10["Event N"] == 8
    assert row10["Normal N"] == 495


def test_baseline_no_lookahead_uses_only_future_close_per_bar():
    """Baseline forward return at bar i uses only close[i] and close[i+h]."""
    close = pd.Series(np.arange(20, dtype=float) + 100)
    eligible = pd.Series([True] * 20)
    out = baseline_forward_returns(close, eligible, horizons=(3,))
    for i in range(17):
        expected = close.iloc[i + 3] / close.iloc[i] - 1.0
        assert round(out["fwd_3"].iloc[i], 8) == round(expected, 8)
    assert pd.isna(out["fwd_3"].iloc[17])
    assert pd.isna(out["fwd_3"].iloc[18])
    assert pd.isna(out["fwd_3"].iloc[19])


def test_eligibility_mask_requires_both_columns():
    df = pd.DataFrame(
        {
            "ema_50_ma": [np.nan, 1.0, 2.0, np.nan, 3.0],
            "ema_50_roll_pctile": [np.nan, np.nan, 50.0, 60.0, 70.0],
        }
    )
    mask = eligibility_mask(df, "ema_50_ma", "ema_50_roll_pctile")
    assert mask.tolist() == [False, False, True, False, True]
