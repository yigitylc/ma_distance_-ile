from __future__ import annotations

from dataclasses import dataclass
import os
import re
import time
from typing import Optional
from urllib.parse import quote

import pandas as pd


RATE_LIMIT_MESSAGE = (
    "Yahoo Finance rate-limited this Streamlit Cloud session. Wait a few minutes and retry, "
    "or use a fallback data source."
)
YFINANCE_BACKOFF_SECONDS = (1.0, 3.0)
YFINANCE_RATE_LIMIT_MARKERS = (
    "YFRateLimitError",
    "Too Many Requests",
    "Rate limited",
    "rate-limited",
    "429",
)


@dataclass(frozen=True)
class MarketDataRequest:
    ticker: str
    period: str = "max"
    start: Optional[str] = None
    end: Optional[str] = None
    interval: str = "1d"
    auto_adjust: bool = False


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns when a single ticker is downloaded."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance can return either (field, ticker) or (ticker, field) depending on params/version.
        # Keep the level that looks like OHLCV field names.
        field_names = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        level0 = set(map(str, df.columns.get_level_values(0)))
        level1 = set(map(str, df.columns.get_level_values(1)))
        if len(level0 & field_names) >= len(level1 & field_names):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)
    return df


def _normalize_ticker(ticker: str) -> str:
    normalized = ticker.upper().strip()
    if not normalized:
        raise ValueError("Ticker cannot be empty")
    if "," in normalized or ";" in normalized or any(ch.isspace() for ch in normalized):
        raise ValueError("Enter a single yfinance ticker")
    return normalized


def _reject_duplicate_ohlcv_fields(df: pd.DataFrame) -> None:
    columns = pd.Index(df.columns)
    duplicated = columns[columns.duplicated()].unique()
    ohlcv_fields = {"open", "high", "low", "close", "adj_close", "volume"}
    duplicated_ohlcv = sorted(str(col) for col in duplicated if str(col) in ohlcv_fields)
    if duplicated_ohlcv:
        fields = ", ".join(duplicated_ohlcv)
        raise ValueError(
            f"Downloaded data contains ambiguous duplicate OHLCV fields: {fields}. "
            "Enter a single yfinance ticker."
        )


def _is_yfinance_rate_limit(value: object) -> bool:
    text = f"{type(value).__name__} {value!r} {value}"
    return any(marker.lower() in text.lower() for marker in YFINANCE_RATE_LIMIT_MARKERS)


class _YFinanceRateLimited(Exception):
    pass


def _make_yfinance_session() -> object | None:
    try:
        from curl_cffi import requests as curl_requests
    except Exception:
        return None
    return curl_requests.Session(impersonate="chrome")


def _call_yfinance_download(yf: object, download_kwargs: dict[str, object]) -> object:
    try:
        return yf.download(**download_kwargs, raise_errors=True)
    except TypeError as exc:
        if "raise_errors" not in str(exc):
            raise
        return yf.download(**download_kwargs)


def _download_yfinance(
    ticker: str,
    period: str,
    interval: str,
    *,
    start: str | None = None,
    end: str | None = None,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("Install yfinance with `pip install yfinance`.") from exc

    download_kwargs = dict(
        tickers=ticker,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
        threads=False,
    )
    session = _make_yfinance_session()
    if session is not None:
        download_kwargs["session"] = session

    # For dashboard/research default: pull maximum available history.
    # If explicit dates are supplied, use start/end instead.
    if start or end:
        download_kwargs["start"] = start
        download_kwargs["end"] = end
    else:
        download_kwargs["period"] = period or "max"

    try:
        for attempt in range(len(YFINANCE_BACKOFF_SECONDS) + 1):
            try:
                df = _call_yfinance_download(yf, download_kwargs)
                if _is_yfinance_rate_limit(df):
                    raise _YFinanceRateLimited(str(df))
                if not isinstance(df, pd.DataFrame):
                    raise ValueError(f"Unexpected yfinance response for ticker={ticker!r}.")
                return df
            except Exception as exc:
                if not isinstance(exc, _YFinanceRateLimited) and not _is_yfinance_rate_limit(exc):
                    raise
                if attempt >= len(YFINANCE_BACKOFF_SECONDS):
                    raise ValueError(RATE_LIMIT_MESSAGE) from exc
                time.sleep(YFINANCE_BACKOFF_SECONDS[attempt])
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            close()

    raise ValueError(RATE_LIMIT_MESSAGE)


def _stooq_symbol(ticker: str) -> str | None:
    if ticker.startswith("^") or ticker.endswith("-USD"):
        return None
    if not re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z])?", ticker):
        return None
    return f"{ticker.lower()}.us"


def _filter_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    period_lower = (period or "max").lower()
    if period_lower == "max":
        return df
    if period_lower.endswith("y"):
        try:
            years = int(period_lower[:-1])
        except ValueError:
            return df
        cutoff = df.index.max() - pd.DateOffset(years=years)
        return df[df.index >= cutoff]
    return df


def _download_stooq_daily(
    ticker: str,
    period: str,
    interval: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    if interval != "1d":
        raise ValueError("Stooq fallback only supports daily data.")
    symbol = _stooq_symbol(ticker)
    if symbol is None:
        raise ValueError(f"Stooq fallback is not available for ticker={ticker!r}.")

    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    stooq_api_key = os.environ.get("STOOQ_API_KEY", "").strip()
    if stooq_api_key:
        url = f"{url}&apikey={stooq_api_key}"
    try:
        df = pd.read_csv(url)
    except Exception as exc:
        raise ValueError(f"No fallback CSV data returned for ticker={ticker!r}.") from exc
    if df.empty:
        raise ValueError(f"No fallback data returned for ticker={ticker!r}.")

    df.columns = [str(col).strip() for col in df.columns]
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(f"No fallback OHLCV data returned for ticker={ticker!r}.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].set_index("Date").sort_index()
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    df = _filter_period(df, period)
    if df.empty:
        raise ValueError(f"No fallback data returned for ticker={ticker!r}.")
    return df


def _chart_period_params(period: str, start: str | None, end: str | None) -> dict[str, object]:
    if start or end:
        period1 = int(pd.Timestamp(start or "1900-01-01").timestamp())
        period2 = int(pd.Timestamp(end).timestamp()) if end else int(pd.Timestamp.utcnow().timestamp())
        return {"period1": period1, "period2": period2}
    return {"range": period or "max"}


def _download_yahoo_chart(
    ticker: str,
    period: str,
    interval: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    session = _make_yfinance_session()
    if session is None:
        raise ValueError("Yahoo chart fallback requires curl_cffi.")

    params = {
        **_chart_period_params(period, start, end),
        "interval": interval,
        "events": "history",
        "includeAdjustedClose": "true",
    }
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker, safe='')}"

    try:
        response = session.get(url, params=params, timeout=30)
        text = response.text
        has_rate_limit_text = any(
            marker.lower() in text.lower()
            for marker in YFINANCE_RATE_LIMIT_MARKERS
            if marker != "429"
        )
        if response.status_code == 429 or has_rate_limit_text:
            raise ValueError(RATE_LIMIT_MESSAGE)
        if response.status_code != 200:
            raise ValueError(f"Yahoo chart fallback returned HTTP {response.status_code}.")
        payload = response.json()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"No Yahoo chart fallback data returned for ticker={ticker!r}.") from exc
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            close()

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise ValueError(f"No Yahoo chart fallback data returned for ticker={ticker!r}.")
    results = chart.get("result") or []
    if not results:
        raise ValueError(f"No Yahoo chart fallback data returned for ticker={ticker!r}.")

    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quote_blocks = indicators.get("quote") or []
    if not timestamps or not quote_blocks:
        raise ValueError(f"No Yahoo chart fallback data returned for ticker={ticker!r}.")

    quote_data = quote_blocks[0]
    adj_blocks = indicators.get("adjclose") or []
    adj_close = adj_blocks[0].get("adjclose") if adj_blocks else None
    index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None)
    data = {
        "Open": quote_data.get("open"),
        "High": quote_data.get("high"),
        "Low": quote_data.get("low"),
        "Close": quote_data.get("close"),
        "Volume": quote_data.get("volume"),
    }
    if adj_close is not None:
        data["Adj Close"] = adj_close
    return pd.DataFrame(data, index=index)


def _prepare_ohlcv_frame(
    df: pd.DataFrame,
    *,
    ticker: str,
    period: str,
    auto_adjust: bool,
    data_source: str,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"No data returned for ticker={ticker!r}.")

    df = _flatten_yfinance_columns(df).copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    if df.empty:
        raise ValueError(f"No dated data returned for ticker={ticker!r}.")
    df = df.sort_index()

    rename_map = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=rename_map)
    _reject_duplicate_ohlcv_fields(df)

    if "adj_close" in df.columns:
        df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
        df["price"] = df["adj_close"]
        price_source = "adj_close"
    elif "close" in df.columns:
        # This happens when auto_adjust=True or for assets where Adj Close is absent.
        # With auto_adjust=True, close is already adjusted by yfinance.
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["adj_close"] = df["close"]
        df["price"] = df["adj_close"]
        price_source = "close_as_adjusted"
    else:
        raise ValueError("Downloaded data does not contain Close or Adj Close.")

    df = df[df["price"].notna()].copy()
    if df.empty:
        raise ValueError(f"No usable price data returned for ticker={ticker!r}.")

    df.attrs["ticker"] = ticker
    df.attrs["period"] = period
    df.attrs["auto_adjust"] = auto_adjust
    df.attrs["price_col"] = "price"
    df.attrs["price_source"] = price_source
    df.attrs["data_source"] = data_source
    return df


def fetch_ohlcv(req: MarketDataRequest) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance, with a daily Stooq fallback for US symbols.

    Default behavior uses ``period='max'`` and ``auto_adjust=False`` so the research
    price series is explicitly based on yfinance's ``Adj Close`` column when it is
    available. If a start/end date is provided, yfinance uses those dates instead
    of the period argument.
    """
    ticker = _normalize_ticker(req.ticker)
    period = req.period or "max"

    yfinance_error: ValueError | None = None
    try:
        df = _download_yfinance(
            ticker,
            period,
            req.interval,
            start=req.start,
            end=req.end,
            auto_adjust=req.auto_adjust,
        )
        if df.empty:
            raise ValueError(f"No data returned for ticker={ticker!r}.")
        data_source = "Yahoo Finance"
    except ValueError as exc:
        yfinance_error = exc
        should_try_stooq = _is_yfinance_rate_limit(exc) or "No data returned" in str(exc)
        if not should_try_stooq:
            raise
        try:
            df = _download_yahoo_chart(
                ticker,
                period,
                req.interval,
                start=req.start,
                end=req.end,
            )
            data_source = "Yahoo Finance chart fallback"
        except Exception:
            df = None

    if yfinance_error is not None and df is None:
        try:
            df = _download_stooq_daily(
                ticker,
                period,
                req.interval,
                start=req.start,
                end=req.end,
            )
            data_source = "Stooq fallback"
        except Exception:
            raise yfinance_error

    return _prepare_ohlcv_frame(
        df,
        ticker=ticker,
        period=period,
        auto_adjust=req.auto_adjust,
        data_source=data_source,
    )
