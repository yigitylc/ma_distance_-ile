from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Optional

import pandas as pd


YFINANCE_ERROR_MESSAGE = (
    "Yahoo Finance returned no data or rate-limited this public cloud session. Try again later."
)
MARKET_DATA_ERROR_MESSAGE = (
    "Market data is temporarily unavailable. Yahoo Finance returned no data or rate-limited this public cloud session, "
    "and no fallback data source could complete the request. Try again later or use Refresh / Clear Cache."
)
YFINANCE_BACKOFF_SECONDS = (1.0, 3.0)
MARKET_DATA_CACHE_TTL_SECONDS = 6 * 60 * 60
YFINANCE_RATE_LIMIT_MARKERS = (
    "YFRateLimitError",
    "Too Many Requests",
    "Rate limited",
    "rate-limited",
    "429",
)
STOOQ_FALLBACK_SOURCE = "Stooq fallback"
YAHOO_SOURCE = "Yahoo Finance"
_COMMON_US_TICKER_RE = re.compile(r"[A-Z]{1,5}(\.[A-Z])?")
_MARKET_DATA_CACHE: dict[tuple[object, ...], tuple[float, pd.DataFrame]] = {}


class MarketDataError(ValueError):
    """Expected provider failure that Streamlit can render without a stack trace."""

    def __init__(
        self,
        message: str,
        *,
        ticker: str | None = None,
        attempted_sources: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.ticker = ticker
        self.attempted_sources = attempted_sources


@dataclass(frozen=True)
class MarketDataResult:
    frame: pd.DataFrame
    data_source: str
    price_source: str
    attempted_sources: tuple[str, ...]


@dataclass(frozen=True)
class MarketDataRequest:
    ticker: str
    period: str = "max"
    start: Optional[str] = None
    end: Optional[str] = None
    interval: str = "1d"
    auto_adjust: bool = True


def clear_market_data_cache() -> None:
    _MARKET_DATA_CACHE.clear()


def _cache_key(
    source: str,
    ticker: str,
    period: str,
    interval: str,
    *,
    start: str | None,
    end: str | None,
    auto_adjust: bool,
) -> tuple[object, ...]:
    return (source, ticker, period, interval, start, end, auto_adjust)


def _from_cache(key: tuple[object, ...]) -> pd.DataFrame | None:
    cached = _MARKET_DATA_CACHE.get(key)
    if cached is None:
        return None

    cached_at, df = cached
    if time.monotonic() - cached_at > MARKET_DATA_CACHE_TTL_SECONDS:
        _MARKET_DATA_CACHE.pop(key, None)
        return None
    return df.copy()


def _save_cache(key: tuple[object, ...], df: pd.DataFrame) -> pd.DataFrame:
    _MARKET_DATA_CACHE[key] = (time.monotonic(), df.copy())
    return df


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns when a single ticker is downloaded."""
    if isinstance(df.columns, pd.MultiIndex):
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


def _raise_market_data_error(
    message: str = YFINANCE_ERROR_MESSAGE,
    *,
    ticker: str | None = None,
    attempted_sources: tuple[str, ...] = (),
    cause: BaseException | None = None,
) -> None:
    error = MarketDataError(message, ticker=ticker, attempted_sources=attempted_sources)
    if cause is not None:
        raise error from cause
    raise error


def _period_dates(period: str) -> tuple[str | None, str | None]:
    period_lower = (period or "max").lower()
    if period_lower not in {"5y", "10y"}:
        return None, None

    years = int(period_lower[:-1])
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    start = today - pd.DateOffset(years=years)
    end = today + pd.Timedelta(days=1)
    return start.date().isoformat(), end.date().isoformat()


def _stooq_symbol(ticker: str) -> str | None:
    normalized = ticker.upper().replace("-", ".")
    if not _COMMON_US_TICKER_RE.fullmatch(normalized):
        return None
    return f"{normalized}.US".lower()


def _read_stooq_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


def _download_stooq(
    ticker: str,
    period: str,
    interval: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    if interval != "1d":
        _raise_market_data_error(f"{STOOQ_FALLBACK_SOURCE} only supports daily data.")

    symbol = _stooq_symbol(ticker)
    if symbol is None:
        _raise_market_data_error(
            f"{STOOQ_FALLBACK_SOURCE} supports common US equity and ETF tickers only."
        )

    query_parts = [f"https://stooq.com/q/d/l/?s={symbol}", "i=d"]
    stooq_start = start
    stooq_end = end
    if not (start or end):
        stooq_start, stooq_end = _period_dates(period)
    if stooq_start:
        query_parts.append(f"d1={stooq_start.replace('-', '')}")
    if stooq_end:
        query_parts.append(f"d2={stooq_end.replace('-', '')}")
    url = "&".join(query_parts)

    try:
        df = _read_stooq_csv(url)
    except Exception as exc:
        _raise_market_data_error(f"{STOOQ_FALLBACK_SOURCE} request failed.", cause=exc)

    if not isinstance(df, pd.DataFrame) or df.empty or "Date" not in df.columns:
        _raise_market_data_error(f"{STOOQ_FALLBACK_SOURCE} returned no data.")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()]
    if df.empty:
        _raise_market_data_error(f"{STOOQ_FALLBACK_SOURCE} returned no dated rows.")
    return df.set_index("Date").sort_index()


def _download_yfinance_uncached(
    ticker: str,
    period: str,
    interval: str,
    *,
    start: str | None = None,
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("Install yfinance with `pip install yfinance`.") from exc

    download_kwargs: dict[str, object] = dict(
        tickers=ticker,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
    )

    if start or end:
        download_kwargs["start"] = start
        download_kwargs["end"] = end
    else:
        derived_start, derived_end = _period_dates(period)
        if derived_start and derived_end:
            download_kwargs["start"] = derived_start
            download_kwargs["end"] = derived_end
        else:
            download_kwargs["period"] = period or "max"

    last_error: BaseException | None = None
    for attempt in range(len(YFINANCE_BACKOFF_SECONDS) + 1):
        try:
            df = yf.download(**download_kwargs)
        except Exception as exc:
            last_error = exc
            if attempt >= len(YFINANCE_BACKOFF_SECONDS):
                if _is_yfinance_rate_limit(exc) or isinstance(exc, ValueError):
                    _raise_market_data_error(YFINANCE_ERROR_MESSAGE, cause=exc)
                _raise_market_data_error(YFINANCE_ERROR_MESSAGE, cause=exc)
            time.sleep(YFINANCE_BACKOFF_SECONDS[attempt])
            continue

        if not isinstance(df, pd.DataFrame):
            last_error = TypeError(f"Expected DataFrame, got {type(df).__name__}.")
        elif df.empty:
            last_error = ValueError("Yahoo Finance returned an empty DataFrame.")
        else:
            return df

        if attempt >= len(YFINANCE_BACKOFF_SECONDS):
            _raise_market_data_error(YFINANCE_ERROR_MESSAGE, cause=last_error)
        time.sleep(YFINANCE_BACKOFF_SECONDS[attempt])

    _raise_market_data_error(YFINANCE_ERROR_MESSAGE, cause=last_error)


def _download_provider(
    source: str,
    ticker: str,
    period: str,
    interval: str,
    *,
    start: str | None = None,
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    key = _cache_key(
        source,
        ticker,
        period,
        interval,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
    )
    cached = _from_cache(key)
    if cached is not None:
        return cached

    if source == YAHOO_SOURCE:
        df = _download_yfinance_uncached(
            ticker,
            period,
            interval,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
        )
    elif source == STOOQ_FALLBACK_SOURCE:
        df = _download_stooq(
            ticker,
            period,
            interval,
            start=start,
            end=end,
        )
    else:  # pragma: no cover - defensive guard
        _raise_market_data_error(f"Unsupported data source: {source}.")

    return _save_cache(key, df)


def _download_market_data(
    ticker: str,
    period: str,
    interval: str,
    *,
    start: str | None = None,
    end: str | None = None,
    auto_adjust: bool = True,
) -> MarketDataResult:
    attempted_sources: list[str] = []
    errors: list[BaseException] = []

    for source, price_source in (
        (YAHOO_SOURCE, "close_auto_adjusted"),
        (STOOQ_FALLBACK_SOURCE, "close_stooq_fallback"),
    ):
        attempted_sources.append(source)
        try:
            df = _download_provider(
                source,
                ticker,
                period,
                interval,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
            )
        except MarketDataError as exc:
            errors.append(exc)
            continue
        except Exception as exc:
            errors.append(exc)
            continue

        if df.empty:
            errors.append(MarketDataError(f"{source} returned no data."))
            continue
        return MarketDataResult(
            frame=df,
            data_source=source,
            price_source=price_source,
            attempted_sources=tuple(attempted_sources),
        )

    last_error = errors[-1] if errors else None
    _raise_market_data_error(
        MARKET_DATA_ERROR_MESSAGE,
        ticker=ticker,
        attempted_sources=tuple(attempted_sources),
        cause=last_error,
    )


def _prepare_ohlcv_frame(
    df: pd.DataFrame,
    *,
    ticker: str,
    period: str,
    auto_adjust: bool,
    data_source: str = YAHOO_SOURCE,
    price_source: str = "close_auto_adjusted",
    attempted_sources: tuple[str, ...] = (YAHOO_SOURCE,),
) -> pd.DataFrame:
    if df.empty:
        raise MarketDataError(YFINANCE_ERROR_MESSAGE)

    df = _flatten_yfinance_columns(df).copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    if df.empty:
        raise MarketDataError(YFINANCE_ERROR_MESSAGE)
    df = df.sort_index()

    rename_map = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=rename_map)
    _reject_duplicate_ohlcv_fields(df)

    if "close" not in df.columns:
        raise MarketDataError(YFINANCE_ERROR_MESSAGE)

    price_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in price_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["adj_close"] = df["close"]
    df["price"] = df["close"]
    df = df[df["price"].notna()].copy()
    if df.empty:
        raise MarketDataError(YFINANCE_ERROR_MESSAGE)

    df.attrs["ticker"] = ticker
    df.attrs["period"] = period
    df.attrs["auto_adjust"] = auto_adjust
    df.attrs["price_col"] = "price"
    df.attrs["price_source"] = price_source
    df.attrs["data_source"] = data_source
    df.attrs["attempted_sources"] = attempted_sources
    df.attrs["market_data_cache_ttl_seconds"] = MARKET_DATA_CACHE_TTL_SECONDS
    return df


def fetch_ohlcv(req: MarketDataRequest) -> pd.DataFrame:
    """Fetch one ticker from yfinance using auto-adjusted Close as research price."""
    ticker = _normalize_ticker(req.ticker)
    period = req.period or "max"
    auto_adjust = True

    try:
        result = _download_market_data(
            ticker,
            period,
            req.interval,
            start=req.start,
            end=req.end,
            auto_adjust=auto_adjust,
        )
        return _prepare_ohlcv_frame(
            result.frame,
            ticker=ticker,
            period=period,
            auto_adjust=auto_adjust,
            data_source=result.data_source,
            price_source=result.price_source,
            attempted_sources=result.attempted_sources,
        )
    except MarketDataError:
        raise
    except ValueError:
        raise
    except Exception as exc:
        raise MarketDataError(MARKET_DATA_ERROR_MESSAGE, ticker=ticker) from exc
