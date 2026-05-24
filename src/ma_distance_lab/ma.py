from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, length: int) -> pd.Series:
    return s.rolling(length, min_periods=length).mean()


def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False, min_periods=length).mean()


def rma(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def wma(s: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1, dtype=float)
    return s.rolling(length, min_periods=length).apply(lambda x: float(np.dot(x, weights) / weights.sum()), raw=True)


def vwma(price: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    pv = price * volume
    return pv.rolling(length, min_periods=length).sum() / volume.rolling(length, min_periods=length).sum()


def dema(s: pd.Series, length: int) -> pd.Series:
    e1 = ema(s, length)
    e2 = ema(e1, length)
    return 2 * e1 - e2


def tema(s: pd.Series, length: int) -> pd.Series:
    e1 = ema(s, length)
    e2 = ema(e1, length)
    e3 = ema(e2, length)
    return 3 * e1 - 3 * e2 + e3


def hma(s: pd.Series, length: int) -> pd.Series:
    half = max(1, int(length / 2))
    sqrt_len = max(1, int(np.sqrt(length)))
    raw = 2 * wma(s, half) - wma(s, length)
    return wma(raw, sqrt_len)


def zlema(s: pd.Series, length: int) -> pd.Series:
    lag = max(1, int((length - 1) / 2))
    adjusted = s + (s - s.shift(lag))
    return ema(adjusted, length)


def kama(s: pd.Series, length: int, fast: int = 2, slow: int = 30) -> pd.Series:
    change = (s - s.shift(length)).abs()
    volatility = s.diff().abs().rolling(length, min_periods=length).sum()
    er = (change / volatility).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    out = pd.Series(index=s.index, dtype=float)
    for i, value in enumerate(s):
        if pd.isna(value):
            out.iloc[i] = np.nan
        elif i == 0 or pd.isna(out.iloc[i - 1]):
            out.iloc[i] = value
        else:
            out.iloc[i] = out.iloc[i - 1] + sc.iloc[i] * (value - out.iloc[i - 1])
    return out


def alma(s: pd.Series, length: int, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    m = offset * (length - 1)
    denom = length / sigma
    weights = np.exp(-((np.arange(length) - m) ** 2) / (2 * denom * denom))
    weights = weights / weights.sum()
    return s.rolling(length, min_periods=length).apply(lambda x: float(np.dot(x, weights)), raw=True)


def moving_average(
    price: pd.Series,
    length: int,
    ma_type: str = "EMA",
    volume: pd.Series | None = None,
) -> pd.Series:
    """Compute a moving average/trend metric."""
    key = ma_type.upper().replace("/", "_").replace(" ", "_")

    if key in {"EMA", "EWMA", "EMA_EWMA"}:
        return ema(price, length)
    if key == "SMA":
        return sma(price, length)
    if key == "WMA":
        return wma(price, length)
    if key in {"RMA", "RMA_WILDER", "WILDER"}:
        return rma(price, length)
    if key == "DEMA":
        return dema(price, length)
    if key == "TEMA":
        return tema(price, length)
    if key == "HMA":
        return hma(price, length)
    if key == "ZLEMA":
        return zlema(price, length)
    if key == "KAMA":
        return kama(price, length)
    if key == "ALMA":
        return alma(price, length)
    if key == "VWMA":
        if volume is None:
            raise ValueError("VWMA requires a volume series.")
        return vwma(price, volume, length)

    raise ValueError(f"Unsupported ma_type: {ma_type!r}")
