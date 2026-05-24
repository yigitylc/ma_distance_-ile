from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


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


def fetch_ohlcv(req: MarketDataRequest) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance.

    Default behavior uses ``period='max'`` and ``auto_adjust=False`` so the research
    price series is explicitly based on yfinance's ``Adj Close`` column when it is
    available. If a start/end date is provided, yfinance uses those dates instead
    of the period argument.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("Install yfinance with `pip install yfinance`.") from exc

    ticker = req.ticker.upper().strip()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    download_kwargs = dict(
        tickers=ticker,
        interval=req.interval,
        auto_adjust=req.auto_adjust,
        progress=False,
        group_by="column",
    )

    # For dashboard/research default: pull maximum available history.
    # If explicit dates are supplied, use start/end instead.
    if req.start or req.end:
        download_kwargs["start"] = req.start
        download_kwargs["end"] = req.end
    else:
        download_kwargs["period"] = req.period or "max"

    df = yf.download(**download_kwargs)

    if df.empty:
        raise ValueError(f"No data returned for ticker={ticker!r}.")

    df = _flatten_yfinance_columns(df).copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    rename_map = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=rename_map)

    if "adj_close" in df.columns:
        df["price"] = df["adj_close"]
        price_source = "adj_close"
    elif "close" in df.columns:
        # This happens when auto_adjust=True or for assets where Adj Close is absent.
        # With auto_adjust=True, close is already adjusted by yfinance.
        df["adj_close"] = df["close"]
        df["price"] = df["adj_close"]
        price_source = "close_as_adjusted"
    else:
        raise ValueError("Downloaded data does not contain Close or Adj Close.")

    df.attrs["ticker"] = ticker
    df.attrs["period"] = req.period
    df.attrs["auto_adjust"] = req.auto_adjust
    df.attrs["price_col"] = "price"
    df.attrs["price_source"] = price_source
    return df
