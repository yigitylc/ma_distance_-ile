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


def test_fetch_ohlcv_rejects_multiple_ticker_input(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()

    def fail_if_called(**_: object) -> pd.DataFrame:
        raise AssertionError("download should not be called for multi-ticker input")

    monkeypatch.setattr("yfinance.download", fail_if_called)
    with pytest.raises(ValueError, match="Enter a single yfinance ticker"):
        fetch_ohlcv(MarketDataRequest(ticker="MSFT AAPL"))


def test_fetch_ohlcv_requires_usable_price_data(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    fake = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [np.nan, np.nan, np.nan],
            "Adj Close": [np.nan, np.nan, np.nan],
            "Volume": [1_000_000.0, 1_000_000.0, 1_000_000.0],
        },
        index=idx,
    )
    monkeypatch.setattr("yfinance.download", lambda **_: fake.copy())

    with pytest.raises(ValueError, match="No usable price data"):
        fetch_ohlcv(MarketDataRequest(ticker="BAD"))


def test_fetch_ohlcv_allows_special_single_ticker_formats(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()
    fake = _make_ohlcv(with_adj_close=True)
    seen_tickers: list[str] = []

    def fake_download(**kwargs: object) -> pd.DataFrame:
        seen_tickers.append(str(kwargs["tickers"]))
        return fake.copy()

    monkeypatch.setattr("yfinance.download", fake_download)
    for ticker in ("BTC-USD", "^GSPC", "BRK-B"):
        df = fetch_ohlcv(MarketDataRequest(ticker=ticker))
        assert not df.empty

    assert seen_tickers == ["BTC-USD", "^GSPC", "BRK-B"]


def test_fetch_ohlcv_converts_yfinance_rate_limit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()

    class YFRateLimitError(Exception):
        pass

    def raise_rate_limit(**_: object) -> pd.DataFrame:
        raise YFRateLimitError("Too Many Requests. Rate limited. Try after a while.")

    monkeypatch.setattr("yfinance.download", raise_rate_limit)
    with pytest.raises(ValueError, match="Yahoo Finance rate-limited"):
        fetch_ohlcv(MarketDataRequest(ticker="NVDA"))


def test_fetch_ohlcv_converts_yfinance_rate_limit_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()
    monkeypatch.setattr("yfinance.download", lambda **_: "YFRateLimitError: Too Many Requests")

    with pytest.raises(ValueError, match="Yahoo Finance rate-limited"):
        fetch_ohlcv(MarketDataRequest(ticker="NVDA"))
