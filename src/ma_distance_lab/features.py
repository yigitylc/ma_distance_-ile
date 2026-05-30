from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .ma import moving_average


@dataclass(frozen=True)
class FeatureConfig:
    ma_type: str = "EMA"
    lengths: tuple[int, ...] = (10, 20, 21, 50, 100, 200)
    rolling_window: int = 500


def distance_pct(price: pd.Series, trend: pd.Series) -> pd.Series:
    return ((price - trend) / trend) * 100.0


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=1)
    return (s - mu) / sd


def _last_percent_rank(values: np.ndarray) -> float:
    current = values[-1]
    if np.isnan(current):
        return np.nan
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return np.nan
    return float((valid <= current).sum() / valid.size * 100.0)


def rolling_percentile_rank(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).apply(_last_percent_rank, raw=True)


def expanding_percentile_rank(s: pd.Series) -> pd.Series:
    vals: list[float] = []
    out: list[float] = []
    for value in s.to_numpy(dtype=float):
        if np.isnan(value):
            out.append(np.nan)
            continue
        vals.append(float(value))
        arr = np.asarray(vals, dtype=float)
        out.append(float((arr <= value).sum() / arr.size * 100.0))
    return pd.Series(out, index=s.index, name=s.name)


def tail_probability(dev: pd.Series, pctile: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(dev >= 0, 100.0 - pctile, pctile),
        index=dev.index,
        name="tail_probability",
    )


def extension_signal(z: float | None, pctile: float | None) -> str:
    if z is None or pctile is None or pd.isna(z) or pd.isna(pctile):
        return "—"
    if z > 2.0 or pctile > 98.0:
        return "↑↑"
    if z > 1.0 or pctile > 90.0:
        return "↑"
    if z < -2.0 or pctile < 2.0:
        return "↓↓"
    if z < -1.0 or pctile < 10.0:
        return "↓"
    return "—"


def build_ma_distance_features(
    df: pd.DataFrame,
    price_col: str = "price",
    config: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    out = df.copy()
    out.attrs.update(df.attrs)
    price = out[price_col]
    volume = out["volume"] if "volume" in out.columns else None

    for length in config.lengths:
        prefix = f"{config.ma_type.lower()}_{length}"
        ma = moving_average(price, length, config.ma_type, volume=volume)
        dev = distance_pct(price, ma)
        rz = rolling_zscore(dev, config.rolling_window)
        rp = rolling_percentile_rank(dev, config.rolling_window)
        ep = expanding_percentile_rank(dev)

        out[f"{prefix}_ma"] = ma
        out[f"{prefix}_dev_pct"] = dev
        out[f"{prefix}_roll_z"] = rz
        out[f"{prefix}_roll_pctile"] = rp
        out[f"{prefix}_exp_pctile"] = ep
        out[f"{prefix}_roll_tail"] = tail_probability(dev, rp)
        out[f"{prefix}_exp_tail"] = tail_probability(dev, ep)
        out[f"{prefix}_roll_signal"] = [extension_signal(z, p) for z, p in zip(rz, rp)]
        out[f"{prefix}_exp_signal"] = [extension_signal(z, p) for z, p in zip(rz, ep)]

    return out


def focus_distance_series(features: pd.DataFrame, ma_type: str, focus_length: int) -> pd.Series:
    """Return the dev_pct series for the chosen focus MA, dropna'd."""
    col = f"{ma_type.lower()}_{int(focus_length)}_dev_pct"
    if col not in features.columns:
        raise KeyError(f"Missing focus deviation column: {col}")
    return features[col].dropna()


def latest_snapshot_table(features: pd.DataFrame, ma_type: str, lengths: Iterable[int]) -> pd.DataFrame:
    rows = []
    last = features.iloc[-1]
    for length in lengths:
        prefix = f"{ma_type.lower()}_{length}"
        rows.append(
            {
                "period": length,
                "% dev": last.get(f"{prefix}_dev_pct"),
                "rolling z": last.get(f"{prefix}_roll_z"),
                "rolling %ile": last.get(f"{prefix}_roll_pctile"),
                "rolling tail": last.get(f"{prefix}_roll_tail"),
                "rolling signal": last.get(f"{prefix}_roll_signal"),
                "expanding %ile": last.get(f"{prefix}_exp_pctile"),
                "expanding tail": last.get(f"{prefix}_exp_tail"),
                "expanding signal": last.get(f"{prefix}_exp_signal"),
            }
        )
    return pd.DataFrame(rows)
