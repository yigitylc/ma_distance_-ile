import numpy as np
import pandas as pd

from ma_distance_lab.events import (
    EventStudyConfig,
    forward_returns_from_events,
    make_event_mask,
    run_event_study,
    summarize_forward_returns,
)
from ma_distance_lab.features import FeatureConfig, build_ma_distance_features


def test_event_cooldown():
    pct = pd.Series([10, 96, 97, 98, 20, 96, 97], dtype=float)
    events = make_event_mask(pct, direction="upside", upside_threshold=95, min_event_gap=3)
    assert events.tolist() == [False, True, False, False, False, True, False]


def test_forward_returns_from_events():
    close = pd.Series([100, 110, 121, 100], dtype=float)
    events = pd.Series([True, False, False, False])
    out = forward_returns_from_events(close, events, horizons=(1, 2, 3))
    assert round(out.iloc[0]["fwd_1"], 4) == 0.1000
    assert round(out.iloc[0]["fwd_2"], 4) == 0.2100
    assert round(out.iloc[0]["fwd_3"], 4) == 0.0000


def test_summary_stats():
    df = pd.DataFrame({"fwd_1": [0.1, -0.2, 0.3]})
    summary = summarize_forward_returns(df, horizons=(1,))
    row = summary.iloc[0]
    assert row["N"] == 3
    assert round(row["% positive"], 2) == 66.67
    assert round(row["% negative"], 2) == 33.33


def test_downside_event_mask():
    pct = pd.Series([50, 4, 3, 2, 50, 1], dtype=float)
    events = make_event_mask(pct, direction="downside", downside_threshold=5, min_event_gap=2)
    assert events.tolist() == [False, True, False, True, False, True]


def test_both_direction_event_mask():
    pct = pd.Series([50, 96, 50, 4, 50, 97, 50], dtype=float)
    events = make_event_mask(
        pct,
        direction="both",
        upside_threshold=95,
        downside_threshold=5,
        min_event_gap=2,
    )
    assert events.tolist() == [False, True, False, True, False, True, False]


def test_mfe_mdd_correctness():
    close = pd.Series([100.0, 110.0, 90.0, 105.0])
    events = pd.Series([True, False, False, False])
    out = forward_returns_from_events(
        close,
        events,
        horizons=(3,),
        include_mfe=True,
        include_mdd=True,
    )
    assert round(out.iloc[0]["mfe_3"], 4) == 0.1000
    assert round(out.iloc[0]["mdd_3"], 4) == round(90 / 110 - 1.0, 4)


def test_quantile_summary():
    rets = np.linspace(-0.10, 0.10, 21)
    df = pd.DataFrame({"fwd_5": rets})
    summary = summarize_forward_returns(df, horizons=(5,), quantile_levels=(0.05, 0.25, 0.75, 0.95))
    row = summary.iloc[0]
    for q, expected in zip((0.05, 0.25, 0.75, 0.95), np.quantile(rets, (0.05, 0.25, 0.75, 0.95))):
        col = f"q{int(round(q * 100)):02d}"
        assert round(row[col], 6) == round(float(expected), 6)


def test_event_detail_enrichment_columns():
    rng = np.random.default_rng(42)
    n = 800
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    price = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=idx, name="price")
    df = pd.DataFrame({"price": price})
    config_features = FeatureConfig(ma_type="EMA", lengths=(50,), rolling_window=200)
    features = build_ma_distance_features(df, config=config_features)

    config = EventStudyConfig(ma_type="EMA", focus_length=50, direction="both", min_event_gap=5)
    summary, event_returns, events = run_event_study(features, config=config)

    assert "event_direction" in event_returns.columns
    assert "event_percentile" in event_returns.columns
    assert "event_dev_pct" in event_returns.columns
    if not event_returns.empty:
        first_event_date = event_returns.index[0]
        pct_col = "ema_50_roll_pctile"
        assert event_returns.iloc[0]["event_percentile"] == features.loc[first_event_date, pct_col]


def test_backward_compatible_default_columns():
    rng = np.random.default_rng(0)
    n = 600
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    price = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.015, n))), index=idx, name="price")
    df = pd.DataFrame({"price": price})
    features = build_ma_distance_features(
        df, config=FeatureConfig(ma_type="EMA", lengths=(50,), rolling_window=200)
    )
    _, event_returns, _ = run_event_study(features, config=EventStudyConfig(focus_length=50))
    assert event_returns.columns.tolist()[:4] == ["fwd_5", "fwd_10", "fwd_21", "fwd_63"]
