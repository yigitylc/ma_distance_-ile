import sys
import types

import numpy as np
import pandas as pd
import pytest

from ma_distance_lab.data import MarketDataRequest, fetch_ohlcv


def _ensure_yfinance_module() -> None:
    if "yfinance" not in sys.modules:
        sys.modules["yfinance"] = types.SimpleNamespace(download=lambda **_: pd.DataFrame())


def _make_ohlcv(with_adj_close: bool) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    data = {
        "Open": np.arange(5, dtype=float) + 100,
        "High": np.arange(5, dtype=float) + 101,
        "Low": np.arange(5, dtype=float) + 99,
        "Close": np.arange(5, dtype=float) + 100,
        "Volume": np.full(5, 1_000_000.0),
    }
    if with_adj_close:
        data["Adj Close"] = np.arange(5, dtype=float) + 95
    return pd.DataFrame(data, index=idx)


def test_fetch_ohlcv_picks_adj_close(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()
    fake = _make_ohlcv(with_adj_close=True)
    monkeypatch.setattr("yfinance.download", lambda **_: fake.copy())
    df = fetch_ohlcv(MarketDataRequest(ticker="NVDA"))
    assert df.attrs["price_source"] == "adj_close"
    pd.testing.assert_series_equal(df["price"], df["adj_close"], check_names=False)


def test_fetch_ohlcv_falls_back_to_close(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()
    fake = _make_ohlcv(with_adj_close=False)
    monkeypatch.setattr("yfinance.download", lambda **_: fake.copy())
    df = fetch_ohlcv(MarketDataRequest(ticker="NVDA"))
    assert df.attrs["price_source"] == "close_as_adjusted"
    pd.testing.assert_series_equal(df["price"], df["close"], check_names=False)


def test_fetch_ohlcv_empty_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()
    monkeypatch.setattr("yfinance.download", lambda **_: pd.DataFrame())
    with pytest.raises(ValueError, match="No data returned"):
        fetch_ohlcv(MarketDataRequest(ticker="ZZZZZZ"))


def test_fetch_ohlcv_empty_ticker_raises() -> None:
    with pytest.raises(ValueError, match="Ticker cannot be empty"):
        fetch_ohlcv(MarketDataRequest(ticker=""))
