from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

import pandas as pd


YFINANCE_ERROR_MESSAGE = (
    "Yahoo Finance returned no data or rate-limited this public cloud session. Try again later."
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
    auto_adjust: bool = True


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


def _period_dates(period: str) -> tuple[str | None, str | None]:
    period_lower = (period or "max").lower()
    if period_lower not in {"5y", "10y"}:
        return None, None

    years = int(period_lower[:-1])
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    start = today - pd.DateOffset(years=years)
    end = today + pd.Timedelta(days=1)
    return start.date().isoformat(), end.date().isoformat()


def _download_yfinance(
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

    for attempt in range(len(YFINANCE_BACKOFF_SECONDS) + 1):
        try:
            df = yf.download(**download_kwargs)
        except Exception as exc:
            if attempt >= len(YFINANCE_BACKOFF_SECONDS):
                if _is_yfinance_rate_limit(exc) or isinstance(exc, ValueError):
                    raise ValueError(YFINANCE_ERROR_MESSAGE) from exc
                raise ValueError(YFINANCE_ERROR_MESSAGE) from exc
            time.sleep(YFINANCE_BACKOFF_SECONDS[attempt])
            continue

        if not isinstance(df, pd.DataFrame):
            raise ValueError(YFINANCE_ERROR_MESSAGE)
        if df.empty:
            raise ValueError(YFINANCE_ERROR_MESSAGE)
        return df

    raise ValueError(YFINANCE_ERROR_MESSAGE)


def _prepare_ohlcv_frame(
    df: pd.DataFrame,
    *,
    ticker: str,
    period: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError(YFINANCE_ERROR_MESSAGE)

    df = _flatten_yfinance_columns(df).copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    if df.empty:
        raise ValueError(YFINANCE_ERROR_MESSAGE)
    df = df.sort_index()

    rename_map = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=rename_map)
    _reject_duplicate_ohlcv_fields(df)

    if "close" not in df.columns:
        raise ValueError(YFINANCE_ERROR_MESSAGE)

    price_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in price_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["adj_close"] = df["close"]
    df["price"] = df["close"]
    df = df[df["price"].notna()].copy()
    if df.empty:
        raise ValueError(YFINANCE_ERROR_MESSAGE)

    df.attrs["ticker"] = ticker
    df.attrs["period"] = period
    df.attrs["auto_adjust"] = auto_adjust
    df.attrs["price_col"] = "price"
    df.attrs["price_source"] = "close_auto_adjusted"
    df.attrs["data_source"] = "Yahoo Finance"
    return df


def fetch_ohlcv(req: MarketDataRequest) -> pd.DataFrame:
    """Fetch one ticker from yfinance using auto-adjusted Close as research price."""
    ticker = _normalize_ticker(req.ticker)
    period = req.period or "max"
    auto_adjust = True

    try:
        df = _download_yfinance(
            ticker,
            period,
            req.interval,
            start=req.start,
            end=req.end,
            auto_adjust=auto_adjust,
        )
        return _prepare_ohlcv_frame(
            df,
            ticker=ticker,
            period=period,
            auto_adjust=auto_adjust,
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(YFINANCE_ERROR_MESSAGE) from exc
