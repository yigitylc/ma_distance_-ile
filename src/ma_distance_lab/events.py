from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

EventDirection = Literal["upside", "downside", "both"]


@dataclass(frozen=True)
class EventStudyConfig:
    ma_type: str = "EMA"
    focus_length: int = 50
    percentile_col_kind: str = "roll_pctile"
    direction: EventDirection = "upside"
    upside_threshold: float = 95.0
    downside_threshold: float = 5.0
    min_event_gap: int = 10
    horizons: tuple[int, ...] = (5, 10, 21, 63)
    include_mfe: bool = False
    include_mdd: bool = False
    include_quantiles: bool = False
    quantile_levels: tuple[float, ...] = (0.05, 0.25, 0.75, 0.95)


def make_event_mask(
    pctile: pd.Series,
    direction: EventDirection = "upside",
    upside_threshold: float = 95.0,
    downside_threshold: float = 5.0,
    min_event_gap: int = 10,
) -> pd.Series:
    """Create a de-duplicated event mask from percentile thresholds.

    Uses only the percentile available at the current timestamp. This is designed
    for rolling percentile triggers and avoids using final full-history ranks.
    """
    if direction == "upside":
        raw = pctile >= upside_threshold
    elif direction == "downside":
        raw = pctile <= downside_threshold
    elif direction == "both":
        raw = (pctile >= upside_threshold) | (pctile <= downside_threshold)
    else:
        raise ValueError(f"Unsupported direction: {direction!r}")

    accepted = pd.Series(False, index=pctile.index)
    last_event_pos: int | None = None
    raw_values = raw.fillna(False).to_numpy(dtype=bool)

    for i, is_event in enumerate(raw_values):
        if not is_event:
            continue
        if last_event_pos is None or i - last_event_pos >= min_event_gap:
            accepted.iloc[i] = True
            last_event_pos = i

    return accepted


def _forward_path_stats(values: np.ndarray, pos: int, h: int) -> tuple[float, float]:
    """Return (mfe, mdd) over the inclusive forward window pos+1..pos+h.

    mfe = max(window) / values[pos] - 1
    mdd = peak-to-trough drawdown over the window, returned as a non-positive float.
    Returns (nan, nan) when the window is incomplete.
    """
    if pos + h >= len(values):
        return (float("nan"), float("nan"))
    base = values[pos]
    window = values[pos + 1 : pos + h + 1]
    if window.size == 0 or not np.isfinite(base) or base == 0.0:
        return (float("nan"), float("nan"))
    mfe = float(window.max()) / float(base) - 1.0
    running_max = np.maximum.accumulate(window)
    drawdowns = window / running_max - 1.0
    mdd = float(drawdowns.min())
    return (mfe, mdd)


def forward_returns_from_events(
    close: pd.Series,
    events: pd.Series,
    horizons: Iterable[int] = (5, 10, 21, 63),
    include_mfe: bool = False,
    include_mdd: bool = False,
    extra_event_cols: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Return one row per event with forward returns for each horizon.

    Column ordering (always): fwd_{h} for each horizon, then optional mfe_{h},
    then optional mdd_{h}, then any extra_event_cols values sampled at the
    event bar by integer position (look-ahead safe).
    """
    horizons = tuple(horizons)
    fwd_cols = [f"fwd_{h}" for h in horizons]
    mfe_cols = [f"mfe_{h}" for h in horizons] if include_mfe else []
    mdd_cols = [f"mdd_{h}" for h in horizons] if include_mdd else []
    extra_names = list(extra_event_cols.keys()) if extra_event_cols else []
    all_cols = fwd_cols + mfe_cols + mdd_cols + extra_names

    close = close.dropna()
    events = events.reindex(close.index).fillna(False)
    idx = close.index
    values = close.to_numpy(dtype=float)
    event_positions = np.flatnonzero(events.to_numpy(dtype=bool))

    aligned_extras: dict[str, np.ndarray] = {}
    if extra_event_cols:
        for name, series in extra_event_cols.items():
            aligned_extras[name] = series.reindex(idx).to_numpy()

    rows: list[dict[str, object]] = []
    for pos in event_positions:
        base = values[pos]
        row: dict[str, object] = {"event_date": idx[pos]}
        for h in horizons:
            if pos + h < len(values) and base != 0.0:
                row[f"fwd_{h}"] = values[pos + h] / base - 1.0
            else:
                row[f"fwd_{h}"] = np.nan
        if include_mfe or include_mdd:
            for h in horizons:
                mfe, mdd = _forward_path_stats(values, pos, h)
                if include_mfe:
                    row[f"mfe_{h}"] = mfe
                if include_mdd:
                    row[f"mdd_{h}"] = mdd
        for name, arr in aligned_extras.items():
            row[name] = arr[pos]
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=all_cols)
    df = pd.DataFrame(rows).set_index("event_date")
    return df[[c for c in all_cols if c in df.columns]]


def _summary_one(
    s: pd.Series,
    quantile_levels: tuple[float, ...] | None = None,
) -> dict[str, float | int]:
    clean = s.dropna()
    n = int(clean.size)
    if n == 0:
        out: dict[str, float | int] = {
            "N": 0,
            "avg": np.nan,
            "median": np.nan,
            "% positive": np.nan,
            "% negative": np.nan,
            "worst": np.nan,
            "best": np.nan,
        }
        if quantile_levels:
            for q in quantile_levels:
                out[_quantile_label(q)] = np.nan
        return out
    out = {
        "N": n,
        "avg": float(clean.mean()),
        "median": float(clean.median()),
        "% positive": float((clean > 0).mean() * 100.0),
        "% negative": float((clean < 0).mean() * 100.0),
        "worst": float(clean.min()),
        "best": float(clean.max()),
    }
    if quantile_levels:
        for q in quantile_levels:
            out[_quantile_label(q)] = float(np.quantile(clean.to_numpy(), q))
    return out


def _quantile_label(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def summarize_forward_returns(
    event_returns: pd.DataFrame,
    horizons: Iterable[int] = (5, 10, 21, 63),
    include_mfe: bool = False,
    include_mdd: bool = False,
    quantile_levels: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    rows = []
    for h in horizons:
        col = f"fwd_{h}"
        series = event_returns[col] if col in event_returns else pd.Series(dtype=float)
        stats = _summary_one(series, quantile_levels=quantile_levels)
        stats = {"horizon": h, **stats}
        if include_mfe:
            mfe_col = f"mfe_{h}"
            mfe_series = event_returns[mfe_col].dropna() if mfe_col in event_returns else pd.Series(dtype=float)
            stats["mfe_avg"] = float(mfe_series.mean()) if mfe_series.size else np.nan
            stats["mfe_best"] = float(mfe_series.max()) if mfe_series.size else np.nan
        if include_mdd:
            mdd_col = f"mdd_{h}"
            mdd_series = event_returns[mdd_col].dropna() if mdd_col in event_returns else pd.Series(dtype=float)
            stats["mdd_avg"] = float(mdd_series.mean()) if mdd_series.size else np.nan
            stats["mdd_worst"] = float(mdd_series.min()) if mdd_series.size else np.nan
        rows.append(stats)
    return pd.DataFrame(rows)


def _label_event_directions(
    pctile_series: pd.Series,
    upside_threshold: float,
    downside_threshold: float,
    fallback: str,
) -> pd.Series:
    """Per-row direction label sampled at each bar's percentile."""
    arr = pctile_series.to_numpy(dtype=float)
    labels = np.where(
        arr >= upside_threshold,
        "upside",
        np.where(arr <= downside_threshold, "downside", fallback),
    )
    return pd.Series(labels, index=pctile_series.index, name="event_direction")


def run_event_study(
    features: pd.DataFrame,
    close_col: str = "price",
    config: EventStudyConfig = EventStudyConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    prefix = f"{config.ma_type.lower()}_{config.focus_length}"
    pct_col = f"{prefix}_{config.percentile_col_kind}"
    dev_col = f"{prefix}_dev_pct"
    if pct_col not in features.columns:
        raise KeyError(f"Missing percentile column: {pct_col}")

    events = make_event_mask(
        features[pct_col],
        direction=config.direction,
        upside_threshold=config.upside_threshold,
        downside_threshold=config.downside_threshold,
        min_event_gap=config.min_event_gap,
    )

    fallback = config.direction if config.direction in ("upside", "downside") else "upside"
    direction_series = _label_event_directions(
        features[pct_col],
        upside_threshold=config.upside_threshold,
        downside_threshold=config.downside_threshold,
        fallback=fallback,
    )
    extra_event_cols: dict[str, pd.Series] = {
        "event_direction": direction_series,
        "event_percentile": features[pct_col],
        "event_dev_pct": features[dev_col] if dev_col in features.columns else pd.Series(np.nan, index=features.index),
    }

    event_returns = forward_returns_from_events(
        features[close_col],
        events,
        config.horizons,
        include_mfe=config.include_mfe,
        include_mdd=config.include_mdd,
        extra_event_cols=extra_event_cols,
    )
    summary = summarize_forward_returns(
        event_returns,
        config.horizons,
        include_mfe=config.include_mfe,
        include_mdd=config.include_mdd,
        quantile_levels=config.quantile_levels if config.include_quantiles else None,
    )
    return summary, event_returns, events


def eligibility_mask(features: pd.DataFrame, ma_col: str, pct_col: str) -> pd.Series:
    """Eligibility for the unconditional baseline: focus MA and rolling percentile both available.

    The forward-return horizon eligibility is handled separately by `baseline_forward_returns`,
    which yields NaN at the tail of the series where a horizon does not fit.
    """
    return features[ma_col].notna() & features[pct_col].notna()


def baseline_forward_returns(
    close: pd.Series,
    eligibility: pd.Series,
    horizons: Iterable[int] = (5, 10, 21, 63),
) -> pd.DataFrame:
    """Forward returns for ALL eligible bars (no event filter).

    For each horizon `h`, returns `close.shift(-h) / close - 1` masked to eligible bars.
    Bars in the last `h` positions yield NaN, which is correct: that horizon is not yet
    observable, so the bar is ineligible for that horizon.
    """
    horizons = tuple(horizons)
    close = close.dropna()
    eligible = eligibility.reindex(close.index).fillna(False).astype(bool)
    out = pd.DataFrame(index=close.index)
    for h in horizons:
        fwd = close.shift(-h) / close - 1.0
        out[f"fwd_{h}"] = fwd.where(eligible)
    return out


def _baseline_stats_one(s: pd.Series) -> dict[str, float | int]:
    clean = s.dropna()
    n = int(clean.size)
    if n == 0:
        return {
            "N": 0,
            "avg": np.nan,
            "median": np.nan,
            "% positive": np.nan,
            "% negative": np.nan,
            "worst": np.nan,
            "best": np.nan,
            "std": np.nan,
            "skew": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q95": np.nan,
        }
    arr = clean.to_numpy(dtype=float)
    return {
        "N": n,
        "avg": float(clean.mean()),
        "median": float(clean.median()),
        "% positive": float((clean > 0).mean() * 100.0),
        "% negative": float((clean < 0).mean() * 100.0),
        "worst": float(clean.min()),
        "best": float(clean.max()),
        "std": float(clean.std(ddof=1)) if n > 1 else np.nan,
        "skew": float(clean.skew()) if n > 2 else np.nan,
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
    }


def summarize_baseline_returns(
    baseline_returns: pd.DataFrame,
    horizons: Iterable[int] = (5, 10, 21, 63),
) -> pd.DataFrame:
    """Per-horizon summary of unconditional forward returns."""
    rows = []
    for h in horizons:
        col = f"fwd_{h}"
        series = baseline_returns[col] if col in baseline_returns else pd.Series(dtype=float)
        rows.append({"horizon": h, **_baseline_stats_one(series)})
    return pd.DataFrame(rows)


def run_baseline_study(
    features: pd.DataFrame,
    close_col: str = "price",
    config: EventStudyConfig = EventStudyConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the unconditional baseline for the same focus MA / percentile / horizons.

    Returns (baseline_summary, baseline_returns) where baseline_returns is a per-bar
    DataFrame with one fwd_{h} column per horizon (NaN where the bar is not eligible).
    """
    prefix = f"{config.ma_type.lower()}_{config.focus_length}"
    ma_col = f"{prefix}_ma"
    pct_col = f"{prefix}_{config.percentile_col_kind}"
    if ma_col not in features.columns:
        raise KeyError(f"Missing focus MA column: {ma_col}")
    if pct_col not in features.columns:
        raise KeyError(f"Missing percentile column: {pct_col}")

    eligible = eligibility_mask(features, ma_col, pct_col)
    baseline_returns = baseline_forward_returns(features[close_col], eligible, config.horizons)
    baseline_summary = summarize_baseline_returns(baseline_returns, config.horizons)
    return baseline_summary, baseline_returns


def compare_event_vs_baseline(
    event_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Side-by-side event vs unconditional-baseline forward-return comparison."""
    e = event_summary.set_index("horizon")
    b = baseline_summary.set_index("horizon")
    horizons = sorted(set(e.index).union(b.index))

    def _get(df: pd.DataFrame, h: int, col: str) -> float:
        if h in df.index and col in df.columns:
            return df.at[h, col]
        return np.nan

    rows = []
    for h in horizons:
        ev_avg = _get(e, h, "avg")
        bl_avg = _get(b, h, "avg")
        ev_med = _get(e, h, "median")
        bl_med = _get(b, h, "median")
        ev_pos = _get(e, h, "% positive")
        bl_pos = _get(b, h, "% positive")
        rows.append(
            {
                "horizon": h,
                "Event N": int(_get(e, h, "N")) if h in e.index else 0,
                "Normal N": int(_get(b, h, "N")) if h in b.index else 0,
                "Event Avg": ev_avg,
                "Normal Avg": bl_avg,
                "Avg Edge": ev_avg - bl_avg if pd.notna(ev_avg) and pd.notna(bl_avg) else np.nan,
                "Event Median": ev_med,
                "Normal Median": bl_med,
                "Median Edge": ev_med - bl_med if pd.notna(ev_med) and pd.notna(bl_med) else np.nan,
                "Event % Positive": ev_pos,
                "Normal % Positive": bl_pos,
                "Hit-Rate Edge": ev_pos - bl_pos if pd.notna(ev_pos) and pd.notna(bl_pos) else np.nan,
                "Event Worst": _get(e, h, "worst"),
                "Normal Worst": _get(b, h, "worst"),
                "Event Best": _get(e, h, "best"),
                "Normal Best": _get(b, h, "best"),
            }
        )
    return pd.DataFrame(rows)
