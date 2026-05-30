import sys
import types

import numpy as np
import pandas as pd
import pytest

import ma_distance_lab.data as data_module
from ma_distance_lab.data import MarketDataRequest, fetch_ohlcv


@pytest.fixture(autouse=True)
def _skip_yfinance_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    data_module.clear_market_data_cache()
    monkeypatch.setattr(data_module, "YFINANCE_BACKOFF_SECONDS", ())
    yield
    data_module.clear_market_data_cache()


def _ensure_yfinance_module() -> None:
    if "yfinance" not in sys.modules:
        sys.modules["yfinance"] = types.SimpleNamespace(download=lambda **_: pd.DataFrame())


def _make_ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open": np.arange(5, dtype=float) + 100,
            "High": np.arange(5, dtype=float) + 101,
            "Low": np.arange(5, dtype=float) + 99,
            "Close": np.arange(5, dtype=float) + 100,
            "Volume": np.full(5, 1_000_000.0),
        },
        index=idx,
    )


def _make_stooq_history(provider_symbol: str = "spy.us") -> types.SimpleNamespace:
    return types.SimpleNamespace(
        data=_make_ohlcv(),
        provider="stooq",
        provider_symbol=provider_symbol,
        price_basis="stooq_close",
        attempted_providers=(
            types.SimpleNamespace(
                provider="stooq",
                provider_symbol=provider_symbol,
                status="success",
                message="",
            ),
        ),
    )


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


def test_fetch_ohlcv_empty_dataframe_raises_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()
    monkeypatch.setattr("yfinance.download", lambda **_: pd.DataFrame())
    monkeypatch.setattr(
        data_module,
        "get_price_history",
        lambda *_, **__: (_ for _ in ()).throw(data_module.StooqFallbackError("no match")),
    )

    with pytest.raises(data_module.MarketDataError, match="Market data is temporarily unavailable"):
        fetch_ohlcv(MarketDataRequest(ticker="ZZZZZZ"))


def test_fetch_ohlcv_empty_yfinance_response_uses_stooq_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_yfinance_module()

    monkeypatch.setattr("yfinance.download", lambda **_: pd.DataFrame())
    monkeypatch.setattr(data_module, "get_price_history", lambda *_, **__: _make_stooq_history("spy.us"))

    df = fetch_ohlcv(MarketDataRequest(ticker="SPY"))

    assert not df.empty
    assert df.attrs["data_source"] == "Stooq fallback"
    assert df.attrs["provider"] == "stooq"
    assert df.attrs["provider_symbol"] == "spy.us"
    assert df.attrs["price_basis"] == "stooq_close"
    assert df.attrs["attempted_sources"] == ("Yahoo Finance", "Stooq fallback")


def test_fetch_ohlcv_rate_limit_error_raises_clean_error_when_fallback_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_yfinance_module()

    class YFRateLimitError(Exception):
        pass

    def raise_rate_limit(**_: object) -> pd.DataFrame:
        raise YFRateLimitError("Too Many Requests. Rate limited. Try after a while.")

    def raise_fallback_unavailable(*_: object, **__: object) -> object:
        raise data_module.StooqFallbackError("fallback unavailable")

    monkeypatch.setattr("yfinance.download", raise_rate_limit)
    monkeypatch.setattr(data_module, "get_price_history", raise_fallback_unavailable)
    with pytest.raises(data_module.MarketDataError, match="Market data is temporarily unavailable") as exc_info:
        fetch_ohlcv(MarketDataRequest(ticker="NVDA"))

    assert exc_info.value.attempted_sources == ("Yahoo Finance", "Stooq fallback")


def test_fetch_ohlcv_rate_limit_uses_stooq_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()

    class YFRateLimitError(Exception):
        pass

    def raise_rate_limit(**_: object) -> pd.DataFrame:
        raise YFRateLimitError("Too Many Requests. Rate limited. Try after a while.")

    monkeypatch.setattr("yfinance.download", raise_rate_limit)
    monkeypatch.setattr(data_module, "get_price_history", lambda *_, **__: _make_stooq_history("nvda.us"))

    df = fetch_ohlcv(MarketDataRequest(ticker="NVDA"))

    assert not df.empty
    assert df.attrs["data_source"] == "Stooq fallback"
    assert df.attrs["price_source"] == "close_stooq_fallback"
    assert df.attrs["provider"] == "stooq"
    assert df.attrs["provider_symbol"] == "nvda.us"
    assert df.attrs["price_basis"] == "stooq_close"
    assert df.attrs["attempted_sources"] == ("Yahoo Finance", "Stooq fallback")


def test_fetch_ohlcv_non_daily_interval_does_not_call_stooq_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_yfinance_module()

    monkeypatch.setattr("yfinance.download", lambda **_: pd.DataFrame())

    def fail_if_called(*_: object, **__: object) -> object:
        raise AssertionError("Stooq fallback should not be called for non-daily intervals")

    monkeypatch.setattr(data_module, "get_price_history", fail_if_called)

    with pytest.raises(data_module.MarketDataError, match="Market data is temporarily unavailable"):
        fetch_ohlcv(MarketDataRequest(ticker="SPY", interval="1wk"))


def test_fetch_ohlcv_missing_adj_close_uses_auto_adjusted_close(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()
    fake = _make_ohlcv()
    monkeypatch.setattr("yfinance.download", lambda **_: fake.copy())

    df = fetch_ohlcv(MarketDataRequest(ticker="NVDA"))

    assert df.attrs["data_source"] == "Yahoo Finance"
    assert df.attrs["provider"] == "yfinance"
    assert df.attrs["provider_symbol"] == "NVDA"
    assert df.attrs["price_basis"] == "yfinance_adjusted_close"
    assert df.attrs["price_source"] == "close_auto_adjusted"
    assert df.attrs["attempted_sources"] == ("Yahoo Finance",)
    pd.testing.assert_series_equal(df["price"], df["close"], check_names=False)
    pd.testing.assert_series_equal(df["adj_close"], df["close"], check_names=False)


def test_fetch_ohlcv_allows_special_single_ticker_formats(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()
    fake = _make_ohlcv()
    seen_tickers: list[str] = []

    def fake_download(**kwargs: object) -> pd.DataFrame:
        seen_tickers.append(str(kwargs["tickers"]))
        return fake.copy()

    monkeypatch.setattr("yfinance.download", fake_download)
    for ticker in ("BTC-USD", "^GSPC", "BRK-B"):
        df = fetch_ohlcv(MarketDataRequest(ticker=ticker))
        assert not df.empty

    assert seen_tickers == ["BTC-USD", "^GSPC", "BRK-B"]


def test_fetch_ohlcv_uses_simple_yfinance_auto_adjust_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()
    fake = _make_ohlcv()
    seen_kwargs: dict[str, object] = {}

    def fake_download(**kwargs: object) -> pd.DataFrame:
        seen_kwargs.update(kwargs)
        return fake.copy()

    monkeypatch.setattr("yfinance.download", fake_download)
    monkeypatch.setattr(
        data_module,
        "get_price_history",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("fallback should not be called")),
    )

    df = fetch_ohlcv(MarketDataRequest(ticker="SPY", period="10y", interval="1d"))

    assert not df.empty
    assert seen_kwargs["auto_adjust"] is True
    assert seen_kwargs["threads"] is False
    assert seen_kwargs["progress"] is False
    assert seen_kwargs["interval"] == "1d"
    assert "start" in seen_kwargs
    assert "end" in seen_kwargs
    assert "period" not in seen_kwargs


def test_fetch_ohlcv_uses_data_layer_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()
    fake = _make_ohlcv()
    calls = 0

    def fake_download(**_: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return fake.copy()

    monkeypatch.setattr("yfinance.download", fake_download)

    first = fetch_ohlcv(MarketDataRequest(ticker="SPY", period="max", interval="1d"))
    second = fetch_ohlcv(MarketDataRequest(ticker="SPY", period="max", interval="1d"))

    assert calls == 1
    assert not first.empty
    assert not second.empty


def test_fetch_ohlcv_retries_empty_yfinance_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()
    fake = _make_ohlcv()
    calls = 0

    def fake_download(**_: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        if calls == 1:
            return pd.DataFrame()
        return fake.copy()

    monkeypatch.setattr(data_module, "YFINANCE_BACKOFF_SECONDS", (0.0,))
    monkeypatch.setattr("yfinance.download", fake_download)

    df = fetch_ohlcv(MarketDataRequest(ticker="SPY", period="max", interval="1d"))

    assert calls == 2
    assert not df.empty
    assert df.attrs["data_source"] == "Yahoo Finance"


def test_fetch_ohlcv_max_uses_yfinance_period(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_yfinance_module()
    fake = _make_ohlcv()
    seen_kwargs: dict[str, object] = {}

    def fake_download(**kwargs: object) -> pd.DataFrame:
        seen_kwargs.update(kwargs)
        return fake.copy()

    monkeypatch.setattr("yfinance.download", fake_download)

    df = fetch_ohlcv(MarketDataRequest(ticker="SPY", period="max", interval="1d"))

    assert not df.empty
    assert seen_kwargs["period"] == "max"
    assert "start" not in seen_kwargs
    assert "end" not in seen_kwargs
