import sys
import types

import numpy as np
import pandas as pd
import pytest

import ma_distance_lab.data as data_module
from ma_distance_lab.data import MarketDataRequest, fetch_ohlcv


def _raise(exc: Exception) -> None:
    raise exc


@pytest.fixture(autouse=True)
def _skip_yfinance_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_module, "YFINANCE_BACKOFF_SECONDS", ())


def _disable_yahoo_chart(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        data_module,
        "_download_yahoo_chart",
        lambda *_args, **_kwargs: _raise(ValueError("Yahoo chart fallback unavailable")),
    )


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


def _make_stooq_ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Date": idx,
            "Open": np.arange(5, dtype=float) + 100,
            "High": np.arange(5, dtype=float) + 101,
            "Low": np.arange(5, dtype=float) + 99,
            "Close": np.arange(5, dtype=float) + 100,
            "Volume": np.full(5, 1_000_000.0),
        }
    )


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
    _disable_yahoo_chart(monkeypatch)
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
    _disable_yahoo_chart(monkeypatch)

    class YFRateLimitError(Exception):
        pass

    def raise_rate_limit(**_: object) -> pd.DataFrame:
        raise YFRateLimitError("Too Many Requests. Rate limited. Try after a while.")

    monkeypatch.setattr("yfinance.download", raise_rate_limit)
    with pytest.raises(ValueError, match="Yahoo Finance rate-limited"):
        fetch_ohlcv(MarketDataRequest(ticker="BTC-USD"))


def test_fetch_ohlcv_converts_yfinance_rate_limit_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()
    _disable_yahoo_chart(monkeypatch)
    monkeypatch.setattr("yfinance.download", lambda **_: "YFRateLimitError: Too Many Requests")

    with pytest.raises(ValueError, match="Yahoo Finance rate-limited"):
        fetch_ohlcv(MarketDataRequest(ticker="BTC-USD"))


def test_fetch_ohlcv_uses_stooq_fallback_for_us_ticker_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()
    _disable_yahoo_chart(monkeypatch)

    class YFRateLimitError(Exception):
        pass

    def raise_rate_limit(**_: object) -> pd.DataFrame:
        raise YFRateLimitError("Too Many Requests. Rate limited. Try after a while.")

    seen_urls: list[str] = []

    def fake_read_csv(url: str) -> pd.DataFrame:
        seen_urls.append(url)
        return _make_stooq_ohlcv()

    monkeypatch.setattr("yfinance.download", raise_rate_limit)
    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    df = fetch_ohlcv(MarketDataRequest(ticker="SPY", period="5y"))

    assert seen_urls == ["https://stooq.com/q/d/l/?s=spy.us&i=d"]
    assert df.attrs["data_source"] == "Stooq fallback"
    assert df.attrs["price_source"] == "close_as_adjusted"
    assert not df.empty


def test_fetch_ohlcv_uses_yahoo_chart_fallback_before_stooq(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()
    fake = _make_ohlcv(with_adj_close=True)

    class YFRateLimitError(Exception):
        pass

    def raise_rate_limit(**_: object) -> pd.DataFrame:
        raise YFRateLimitError("429 Too Many Requests")

    def fail_stooq(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise AssertionError("Stooq should not be called when Yahoo chart fallback succeeds")

    monkeypatch.setattr("yfinance.download", raise_rate_limit)
    monkeypatch.setattr(data_module, "_download_yahoo_chart", lambda *_args, **_kwargs: fake.copy())
    monkeypatch.setattr(data_module, "_download_stooq_daily", fail_stooq)

    df = fetch_ohlcv(MarketDataRequest(ticker="GLD", period="10y"))

    assert df.attrs["data_source"] == "Yahoo Finance chart fallback"
    assert df.attrs["price_source"] == "adj_close"
    assert not df.empty


def test_yahoo_chart_fallback_allows_429_inside_normal_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 200
        text = '{"chart":{"result":[{"regularMarketPrice":429.0}],"error":null}}'

        def json(self) -> dict[str, object]:
            return {
                "chart": {
                    "error": None,
                    "result": [
                        {
                            "timestamp": [1_577_836_800],
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [429.0],
                                        "high": [430.0],
                                        "low": [428.0],
                                        "close": [429.5],
                                        "volume": [1_000_000],
                                    }
                                ],
                                "adjclose": [{"adjclose": [429.5]}],
                            },
                        }
                    ],
                }
            }

    class FakeSession:
        def get(self, *_args: object, **_kwargs: object) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            return None

    monkeypatch.setattr(data_module, "_make_yfinance_session", lambda: FakeSession())

    df = data_module._download_yahoo_chart("GLD", "10y", "1d")

    assert len(df) == 1
    assert float(df["Adj Close"].iloc[0]) == 429.5


def test_fetch_ohlcv_uses_stooq_fallback_for_empty_yfinance_us_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()
    _disable_yahoo_chart(monkeypatch)
    monkeypatch.setattr("yfinance.download", lambda **_: pd.DataFrame())
    monkeypatch.setattr(pd, "read_csv", lambda _: _make_stooq_ohlcv())

    df = fetch_ohlcv(MarketDataRequest(ticker="GLD", period="10y"))

    assert df.attrs["data_source"] == "Stooq fallback"
    assert not df.empty


def test_fetch_ohlcv_passes_threads_false_and_curl_session(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()
    fake = _make_ohlcv(with_adj_close=True)
    sentinel_session = object()
    seen_kwargs: dict[str, object] = {}

    def fake_download(**kwargs: object) -> pd.DataFrame:
        seen_kwargs.update(kwargs)
        return fake.copy()

    monkeypatch.setattr(data_module, "_make_yfinance_session", lambda: sentinel_session)
    monkeypatch.setattr("yfinance.download", fake_download)

    df = fetch_ohlcv(MarketDataRequest(ticker="NVDA", period="10y"))

    assert not df.empty
    assert seen_kwargs["threads"] is False
    assert seen_kwargs["session"] is sentinel_session
    assert seen_kwargs["period"] == "10y"
