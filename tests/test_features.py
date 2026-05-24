import numpy as np
import pandas as pd

from ma_distance_lab.features import (
    FeatureConfig,
    build_ma_distance_features,
    distance_pct,
    expanding_percentile_rank,
    extension_signal,
    focus_distance_series,
    latest_snapshot_table,
    rolling_percentile_rank,
    rolling_zscore,
    tail_probability,
)


def test_extension_signal_uses_z_or_percentile():
    assert extension_signal(1.4, 98.5) == "↑↑"
    assert extension_signal(1.2, 50.0) == "↑"
    assert extension_signal(-1.4, 5.0) == "↓"
    assert extension_signal(-0.5, 1.5) == "↓↓"
    assert extension_signal(0.0, 50.0) == "—"


def test_tail_probability_directional():
    dev = pd.Series([10.0, -10.0, 0.0])
    pct = pd.Series([98.0, 3.0, 60.0])
    out = tail_probability(dev, pct)
    assert out.iloc[0] == 2.0
    assert out.iloc[1] == 3.0
    assert out.iloc[2] == 40.0


def test_rolling_percentile_rank_last_value():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = rolling_percentile_rank(s, 5)
    assert out.iloc[-1] == 100.0


def test_distance_pct_correctness():
    price = pd.Series([110.0, 90.0, 100.0])
    trend = pd.Series([100.0, 100.0, 100.0])
    out = distance_pct(price, trend)
    assert out.iloc[0] == 10.0
    assert out.iloc[1] == -10.0
    assert out.iloc[2] == 0.0


def test_rolling_zscore_known_values():
    s = pd.Series(np.arange(50, dtype=float))
    out = rolling_zscore(s, 20)
    assert pd.isna(out.iloc[18])
    last = out.iloc[-1]
    window = s.iloc[-20:]
    expected = (window.iloc[-1] - window.mean()) / window.std(ddof=1)
    assert round(last, 8) == round(expected, 8)
    assert last > 0


def test_expanding_percentile_no_lookahead():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = expanding_percentile_rank(s)
    assert out.iloc[2] == 100.0


def test_expanding_percentile_prefix_invariance():
    s = pd.Series([3, 1, 4, 1, 5, 9, 2, 6], dtype=float)
    full = expanding_percentile_rank(s)
    prefix = expanding_percentile_rank(s.iloc[:5])
    for i in range(5):
        assert full.iloc[i] == prefix.iloc[i]


def test_latest_snapshot_table_shape_and_columns():
    rng = np.random.default_rng(3)
    n = 400
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    df = pd.DataFrame({"price": price})
    features = build_ma_distance_features(
        df, config=FeatureConfig(ma_type="EMA", lengths=(20, 50), rolling_window=100)
    )
    snapshot = latest_snapshot_table(features, "EMA", (20, 50))
    assert snapshot.shape == (2, 9)
    assert snapshot.columns.tolist() == [
        "period",
        "% dev",
        "rolling z",
        "rolling %ile",
        "rolling tail",
        "rolling signal",
        "expanding %ile",
        "expanding tail",
        "expanding signal",
    ]


def test_focus_distance_series_returns_dev_pct():
    rng = np.random.default_rng(5)
    n = 300
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    df = pd.DataFrame({"price": price})
    features = build_ma_distance_features(
        df, config=FeatureConfig(ma_type="EMA", lengths=(50,), rolling_window=100)
    )
    out = focus_distance_series(features, "EMA", 50)
    assert out.name == "ema_50_dev_pct" or out.name == "ema_50_dev_pct"
    pd.testing.assert_series_equal(out, features["ema_50_dev_pct"].dropna())
