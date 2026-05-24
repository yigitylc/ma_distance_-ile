import numpy as np
import pandas as pd
import pytest

from ma_distance_lab.ma import ema, moving_average, sma


@pytest.fixture
def sample_series() -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(100 + np.cumsum(rng.normal(0, 1, 300)))


@pytest.mark.parametrize(
    "ma_type",
    ["SMA", "EMA", "WMA", "RMA", "DEMA", "TEMA", "HMA", "ZLEMA", "KAMA", "ALMA"],
)
def test_moving_average_dispatch_returns_series_per_type(sample_series: pd.Series, ma_type: str) -> None:
    out = moving_average(sample_series, length=20, ma_type=ma_type)
    assert isinstance(out, pd.Series)
    assert len(out) == len(sample_series)
    assert out.dropna().size > 0


def test_vwma_returns_series(sample_series: pd.Series) -> None:
    rng = np.random.default_rng(11)
    volume = pd.Series(rng.integers(1_000, 10_000, len(sample_series)).astype(float))
    out = moving_average(sample_series, length=20, ma_type="VWMA", volume=volume)
    assert isinstance(out, pd.Series)
    assert out.dropna().size > 0


def test_sma_matches_manual_rolling_mean(sample_series: pd.Series) -> None:
    expected = sample_series.rolling(20, min_periods=20).mean()
    actual = sma(sample_series, 20)
    pd.testing.assert_series_equal(actual, expected)


def test_ema_matches_manual_ewm(sample_series: pd.Series) -> None:
    expected = sample_series.ewm(span=20, adjust=False, min_periods=20).mean()
    actual = ema(sample_series, 20)
    pd.testing.assert_series_equal(actual, expected)


def test_vwma_requires_volume(sample_series: pd.Series) -> None:
    with pytest.raises(ValueError, match="VWMA"):
        moving_average(sample_series, length=10, ma_type="VWMA", volume=None)


def test_unsupported_type_raises(sample_series: pd.Series) -> None:
    with pytest.raises(ValueError, match="Unsupported ma_type"):
        moving_average(sample_series, length=10, ma_type="FOO")


def test_sma_warmup_nan_count(sample_series: pd.Series) -> None:
    out = sma(sample_series, 20)
    assert out.iloc[:19].isna().all()
    assert not pd.isna(out.iloc[19])
