"""Deterministic, data-driven commentary generated from the dashboard's
already-computed tables. No LLM, no external calls. All text is derived from
the actual values in the snapshot, event-study, baseline-comparison, and
event-detail DataFrames produced upstream.

Each interpret_* function takes computed objects and returns a markdown string.
Functions degrade gracefully when inputs are empty, when optional columns
(MFE/MDD/quantiles) are missing, or when a horizon has zero events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Formatting + classification helpers
# ---------------------------------------------------------------------------

def _fmt_pct(value: Any, signed: bool = True) -> str:
    """Format a decimal return (e.g. 0.0483) as a percentage string ('+4.83%')."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:+.2f}%" if signed else f"{value * 100:.2f}%"


def _fmt_pp(value: Any, signed: bool = True) -> str:
    """Format a value already expressed in percentage points (e.g. 12.34) as '12.34 pp'."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.2f} pp" if signed else f"{value:.2f} pp"


def _fmt_int(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{int(value):,}"


def _classify_edge_pp(edge_pp: float) -> str:
    """Classify an edge expressed in percentage points (positive or negative)."""
    if edge_pp is None or pd.isna(edge_pp):
        return "unknown"
    mag = abs(edge_pp)
    if mag < 0.10:
        return "economically small / close to normal"
    if mag < 0.50:
        return "mild"
    if mag < 1.50:
        return "meaningful"
    return "strong"


def _decimal_edge_to_pp(edge_decimal: float) -> float:
    """Convert a return-edge (e.g. 0.0048 = 48 bps) to percentage points (0.48 pp)."""
    if edge_decimal is None or pd.isna(edge_decimal):
        return float("nan")
    return float(edge_decimal) * 100.0


def _classify_skew(skew: float) -> str:
    if skew is None or pd.isna(skew):
        return "skew unavailable"
    if skew > 0.5:
        return "right-skewed"
    if skew < -0.5:
        return "left-skewed"
    return "roughly balanced / mild skew"


def _safe_get(row: pd.Series, key: str, default: Any = float("nan")) -> Any:
    if key in row.index:
        return row[key]
    return default


# ---------------------------------------------------------------------------
# Per-section interpretations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunMetadata:
    ticker: str
    interval: str
    period: str
    bars_loaded: int
    first_date: str
    last_date: str
    price_source: str
    ma_type: str
    focus_length: int
    rolling_window: int


def interpret_run_summary(meta: RunMetadata) -> str:
    return "\n".join(
        [
            f"**Run summary.** {meta.ticker} on `{meta.interval}` data, period=`{meta.period}`, "
            f"{meta.bars_loaded:,} bars from {meta.first_date} to {meta.last_date}.",
            f"Research price = `{meta.price_source}`. Focus MA = **{meta.ma_type} {meta.focus_length}** "
            f"with rolling window = **{meta.rolling_window}** bars.",
            "",
            "_The rolling window controls both the rolling-regime percentile and the rolling Z-score; "
            "longer windows produce smoother regime context but slower reaction to new conditions. "
            "All event triggers below use the rolling-regime percentile to avoid look-ahead bias._",
        ]
    )


def interpret_latest_snapshot(snapshot_df: pd.DataFrame, focus_length: int) -> str:
    if snapshot_df is None or snapshot_df.empty:
        return "_No latest-snapshot data available yet._"

    parts: list[str] = ["**Latest snapshot interpretation.**"]
    df = snapshot_df.copy()
    most_stretched = df.iloc[df["% dev"].abs().fillna(-1).argmax()]
    period = int(most_stretched["period"])
    dev = float(most_stretched["% dev"])
    direction = "upside extension" if dev > 0 else "downside extension" if dev < 0 else "flat"
    parts.append(
        f"- Most stretched MA in the ladder: **{period}-period** at "
        f"`{dev:+.2f}%` deviation ({direction})."
    )

    focus_row = df[df["period"] == focus_length]
    if not focus_row.empty:
        row = focus_row.iloc[0]
        roll_pctile = row.get("rolling %ile")
        roll_tail = row.get("rolling tail")
        exp_pctile = row.get("expanding %ile")
        focus_dev = row.get("% dev")
        if pd.notna(roll_pctile):
            if roll_pctile >= 95 and pd.notna(focus_dev) and focus_dev > 0:
                parts.append(
                    f"- Focus MA ({focus_length}) sits at the **upper tail of the recent rolling regime** "
                    f"(rolling %ile = `{roll_pctile:.2f}`)."
                )
            elif roll_pctile <= 5 and pd.notna(focus_dev) and focus_dev < 0:
                parts.append(
                    f"- Focus MA ({focus_length}) sits at the **lower tail of the recent rolling regime** "
                    f"(rolling %ile = `{roll_pctile:.2f}`)."
                )
            else:
                parts.append(
                    f"- Focus MA ({focus_length}) rolling %ile = `{roll_pctile:.2f}`, "
                    f"deviation = `{focus_dev:+.2f}%` (no tail extreme)."
                )
        if pd.notna(roll_tail):
            if roll_tail <= 2:
                parts.append(f"- Tail probability `{roll_tail:.2f}%` — **very rare** in the recent regime.")
            elif roll_tail <= 5:
                parts.append(f"- Tail probability `{roll_tail:.2f}%` — **rare** in the recent regime.")
            elif roll_tail <= 10:
                parts.append(
                    f"- Tail probability `{roll_tail:.2f}%` — elevated but not extremely rare."
                )
            else:
                parts.append(f"- Tail probability `{roll_tail:.2f}%` — within normal range.")
        if pd.notna(roll_pctile) and pd.notna(exp_pctile):
            gap = float(roll_pctile) - float(exp_pctile)
            if gap > 15:
                parts.append(
                    f"- Rolling regime is **far more extreme** than long-run expanding history "
                    f"(rolling `{roll_pctile:.1f}` vs expanding `{exp_pctile:.1f}`)."
                )
            elif gap < -15:
                parts.append(
                    f"- Long-run expanding history is more extreme than the recent rolling regime "
                    f"(rolling `{roll_pctile:.1f}` vs expanding `{exp_pctile:.1f}`); "
                    "current state is unusual only in a longer-history sense."
                )

    high_pctile_rows = df[df["rolling %ile"].fillna(0) >= 90]
    if len(high_pctile_rows) >= 3:
        periods = ", ".join(str(int(p)) for p in high_pctile_rows["period"])
        parts.append(
            f"- Extension is **broad across the trend ladder** (high rolling %ile at: {periods}), "
            "not just a single short-term spike."
        )

    parts.append("")
    parts.append(
        "_This is a regime signal, not an entry trigger. Percentile-based readings are more robust "
        "than raw Z-scores when MA-distance distributions are non-normal._"
    )
    return "\n".join(parts)


def interpret_price_chart(meta: RunMetadata, event_count: int, upside_count: int, downside_count: int) -> str:
    return "\n".join(
        [
            "**Price chart with focus MA and event markers.**",
            f"- {event_count:,} de-duplicated events plotted on the price line "
            f"({upside_count:,} upside, {downside_count:,} downside).",
            "- Markers identify the bar at which the rolling-regime percentile crossed the configured threshold "
            "and survived the cooldown window.",
            "",
            "_Cluster patterns matter: tight clusters of upside markers can flag euphoric extension regimes; "
            "tight clusters of downside markers can flag capitulation regimes. Isolated markers are weaker signals._",
        ]
    )


def interpret_focus_diagnostics(focus_series_df: pd.DataFrame, focus_length: int) -> str:
    """focus_series_df should expose the latest values keyed by:
    'dev_pct', 'roll_z', 'roll_pctile', 'roll_tail'.
    """
    if focus_series_df is None or focus_series_df.empty:
        return "_No diagnostic series available yet._"

    last = focus_series_df.iloc[-1]
    dev = last.get("dev_pct")
    z = last.get("roll_z")
    pctile = last.get("roll_pctile")
    tail = last.get("roll_tail")

    parts = [
        "**Four-panel focus diagnostics.**",
        f"- **% Deviation** — current `{_safe_fmt_signed_pct(dev)}`. Raw distance from the "
        f"{focus_length}-period MA. Positive = price stretched above MA, negative = below.",
        f"- **Rolling Z-score** — current `{z:+.2f}`." if pd.notna(z) else "- **Rolling Z-score** — unavailable.",
        "  Useful but distorted by non-normal / fat-tailed distributions; treat extreme |Z| > 2 with care.",
        f"- **Rolling Percentile** — current `{pctile:.2f}` (empirical rank vs the rolling window). "
        "More robust than Z when the distribution is skewed."
        if pd.notna(pctile)
        else "- **Rolling Percentile** — unavailable.",
        f"- **Tail Probability** — current `{tail:.2f}%` (upside: 100 - %ile; downside: %ile)."
        if pd.notna(tail)
        else "- **Tail Probability** — unavailable.",
    ]

    if pd.notna(tail):
        if tail <= 2:
            parts.append("  → **Very rare** state in the recent rolling regime.")
        elif tail <= 5:
            parts.append("  → **Rare** state in the recent rolling regime.")
        elif tail <= 10:
            parts.append("  → Elevated but not extremely rare.")
        else:
            parts.append("  → Within normal range; not a tail event in the recent regime.")
    parts.append("")
    parts.append(
        "_A low tail probability means the condition is rare, not automatically bearish or bullish._"
    )
    return "\n".join(parts)


def _safe_fmt_signed_pct(x: Any) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):+.2f}%"


def interpret_distance_distribution(focus_dev_series: pd.Series) -> str:
    if focus_dev_series is None or focus_dev_series.dropna().empty:
        return "_No deviation data available._"
    s = focus_dev_series.dropna()
    current = float(s.iloc[-1])
    mean = float(s.mean())
    median = float(s.median())
    std = float(s.std(ddof=1)) if s.size > 1 else float("nan")
    skew = float(s.skew()) if s.size > 2 else float("nan")
    smin = float(s.min())
    smax = float(s.max())
    rank = float((s <= current).mean() * 100.0)

    parts = [
        "**Focus MA-distance distribution.**",
        f"- Current deviation: `{current:+.2f}%`, sample percentile = `{rank:.2f}`.",
        f"- Mean = `{mean:+.2f}%`, Median = `{median:+.2f}%`, Std = `{std:.2f}%`, "
        f"min = `{smin:+.2f}%`, max = `{smax:+.2f}%`.",
        f"- Skew = `{skew:+.2f}` → **{_classify_skew(skew)}**.",
    ]
    if abs(mean - median) > max(0.25, 0.5 * std):
        parts.append(
            "- Mean and median diverge — average reading is being pulled by outliers; "
            "median is more representative of the typical state."
        )
    if rank >= 95:
        parts.append("- Current value sits in the **right tail** of the historical distribution.")
    elif rank <= 5:
        parts.append("- Current value sits in the **left tail** of the historical distribution.")
    parts.append("")
    parts.append(
        "_Percentile is preferred over Z-score for non-normal distributions. "
        "If the distribution has a fat left tail, downside crash regimes have produced larger negative deviations than normal conditions._"
    )
    return "\n".join(parts)


def interpret_event_summary(event_summary: pd.DataFrame, event_count: int) -> str:
    if event_summary is None or event_summary.empty:
        return "_No events met the trigger criteria. Consider loosening thresholds or shrinking the rolling window._"

    parts = [f"**Forward-return event study** — {event_count:,} de-duplicated events."]
    has_mfe = "mfe_avg" in event_summary.columns
    has_mdd = "mdd_avg" in event_summary.columns

    for _, row in event_summary.iterrows():
        h = int(row["horizon"])
        n = int(row.get("N", 0))
        if n == 0:
            parts.append(f"- **{h} bars**: no completed forward windows.")
            continue
        avg = row.get("avg")
        med = row.get("median")
        pos = row.get("% positive")
        neg = row.get("% negative")
        worst = row.get("worst")
        best = row.get("best")
        line = (
            f"- **{h} bars** (N={n:,}): avg `{_fmt_pct(avg)}`, median `{_fmt_pct(med)}`, "
            f"%pos `{pos:.2f}%` / %neg `{neg:.2f}%`, worst `{_fmt_pct(worst)}`, best `{_fmt_pct(best)}`."
        )
        if has_mfe and pd.notna(row.get("mfe_avg")):
            line += f" MFE avg `{_fmt_pct(row['mfe_avg'])}`."
        if has_mdd and pd.notna(row.get("mdd_avg")):
            line += f" MDD avg `{_fmt_pct(row['mdd_avg'])}`."
        agreement = ""
        if pd.notna(avg) and pd.notna(med):
            if avg > 0 and med > 0:
                agreement = " Avg and median agree (positive expectancy)."
            elif avg < 0 and med < 0:
                agreement = " Avg and median agree (negative expectancy)."
            elif avg > 0 >= med:
                agreement = " Avg positive but median ≤ 0 — outliers dominate; not a typical-case win."
            elif med > 0 >= avg:
                agreement = " Median positive but avg ≤ 0 — typical case improves but adverse tail drags the mean."
        parts.append(line + agreement)

    parts.append("")
    parts.append(
        "_Positive event returns alone are not enough; they must be compared against normal QQQ drift "
        "(see the Event vs Normal table). Worst-case returns describe **path risk** even when expectancy is positive._"
    )
    return "\n".join(parts)


def interpret_event_vs_normal(comparison: pd.DataFrame) -> str:
    if comparison is None or comparison.empty:
        return "_Comparison table unavailable._"

    parts = ["**Event Returns vs Normal Forward Returns.**"]
    parts.append(
        "_Event returns are conditional on the trigger; normal returns are the unconditional baseline "
        "over all eligible bars (focus MA + rolling percentile available)._"
    )
    for _, row in comparison.iterrows():
        h = int(row["horizon"])
        ev_avg = row.get("Event Avg")
        bl_avg = row.get("Normal Avg")
        ev_med = row.get("Event Median")
        bl_med = row.get("Normal Median")
        ev_pos = row.get("Event % Positive")
        bl_pos = row.get("Normal % Positive")
        avg_edge_pp = _decimal_edge_to_pp(row.get("Avg Edge"))
        med_edge_pp = _decimal_edge_to_pp(row.get("Median Edge"))
        hit_edge = row.get("Hit-Rate Edge")  # already in pp

        verdict = _verdict_from_edges(avg_edge_pp, med_edge_pp)
        hit_msg = _hit_rate_msg(hit_edge)
        avg_class = _classify_edge_pp(avg_edge_pp)
        med_class = _classify_edge_pp(med_edge_pp)
        hit_class = _classify_edge_pp(hit_edge)

        parts.append(
            f"- **{h} bars**: event avg `{_fmt_pct(ev_avg)}` vs normal `{_fmt_pct(bl_avg)}` "
            f"(edge `{avg_edge_pp:+.2f} pp`, {avg_class}); "
            f"event median `{_fmt_pct(ev_med)}` vs normal `{_fmt_pct(bl_med)}` "
            f"(edge `{med_edge_pp:+.2f} pp`, {med_class}); "
            f"event %pos `{ev_pos:.2f}%` vs normal `{bl_pos:.2f}%` "
            f"(hit-rate edge `{hit_edge:+.2f} pp`, {hit_class})."
        )
        parts.append(f"  → {verdict} {hit_msg}".strip())

    parts.append("")
    parts.append(
        "_Edge classification: `< 0.10 pp` = economically small, `0.10–0.50 pp` = mild, "
        "`0.50–1.50 pp` = meaningful, `> 1.50 pp` = strong._"
    )
    return "\n".join(parts)


def _verdict_from_edges(avg_edge_pp: float, med_edge_pp: float) -> str:
    if pd.isna(avg_edge_pp) or pd.isna(med_edge_pp):
        return "Insufficient data for a verdict."
    if avg_edge_pp > 0 and med_edge_pp > 0:
        return "Event condition outperformed normal behavior on both average and median."
    if avg_edge_pp > 0 and med_edge_pp <= 0:
        return "Average outperformance may be driven by outliers; median does not confirm typical improvement."
    if avg_edge_pp <= 0 and med_edge_pp > 0:
        return "Typical event outcome improved, but average is dragged by adverse tail events."
    return "Event condition underperformed normal behavior."


def _hit_rate_msg(hit_edge_pp: float) -> str:
    if pd.isna(hit_edge_pp):
        return ""
    if hit_edge_pp > 0:
        return "Higher probability of a positive outcome than the normal baseline."
    if hit_edge_pp < 0:
        return "Lower probability of a positive outcome than the normal baseline."
    return "Hit rate matches the normal baseline."


def interpret_edge_bars(comparison: pd.DataFrame) -> str:
    if comparison is None or comparison.empty:
        return "_No comparison data to summarize._"

    df = comparison.copy()
    df["avg_edge_pp"] = df["Avg Edge"].apply(_decimal_edge_to_pp)
    df["med_edge_pp"] = df["Median Edge"].apply(_decimal_edge_to_pp)
    df["hit_edge_pp"] = df["Hit-Rate Edge"]

    def _peak(col: str) -> tuple[int | None, float]:
        valid = df.dropna(subset=[col])
        if valid.empty:
            return None, float("nan")
        idx = valid[col].abs().idxmax()
        return int(valid.loc[idx, "horizon"]), float(valid.loc[idx, col])

    avg_h, avg_v = _peak("avg_edge_pp")
    med_h, med_v = _peak("med_edge_pp")
    hit_h, hit_v = _peak("hit_edge_pp")

    parts = ["**Edge bar charts — summary.**"]
    if avg_h is not None:
        parts.append(
            f"- Strongest **average** edge at the **{avg_h}-bar** horizon "
            f"(`{avg_v:+.2f} pp`, {_classify_edge_pp(avg_v)})."
        )
    if med_h is not None:
        parts.append(
            f"- Strongest **median** edge at the **{med_h}-bar** horizon "
            f"(`{med_v:+.2f} pp`, {_classify_edge_pp(med_v)})."
        )
    if hit_h is not None:
        parts.append(
            f"- Strongest **hit-rate** edge at the **{hit_h}-bar** horizon "
            f"(`{hit_v:+.2f} pp`, {_classify_edge_pp(hit_v)})."
        )

    short_horizons = df[df["horizon"].isin([5, 10, 21])]
    if not short_horizons.empty:
        weak_or_negative = (
            (short_horizons["avg_edge_pp"].fillna(0) <= 0.10).all()
            and (short_horizons["med_edge_pp"].fillna(0) <= 0.10).all()
        )
        long_row = df[df["horizon"] == 63]
        long_strong = (
            not long_row.empty
            and pd.notna(long_row["avg_edge_pp"].iloc[0])
            and abs(long_row["avg_edge_pp"].iloc[0]) >= 0.50
        )
        if weak_or_negative and long_strong:
            parts.append(
                "- Edge is **concentrated at the 63-bar horizon**; short-term horizons do not show robust "
                "outperformance versus normal behavior. The signal is more useful for **regime classification** "
                "than short-term timing."
            )
    parts.append("")
    parts.append(
        "_A single horizon dominating the edge is more believable than a flat profile across all horizons "
        "if there is a coherent reason — e.g. mean-reversion timeframe matches the focus MA span._"
    )
    return "\n".join(parts)


def interpret_forward_distributions(event_returns: pd.DataFrame) -> str:
    if event_returns is None or event_returns.empty:
        return "_No event-return distribution to interpret._"

    fwd_cols = [c for c in event_returns.columns if c.startswith("fwd_")]
    if not fwd_cols:
        return "_No forward-return columns present._"

    parts = ["**Forward-return distributions (box plots + histograms).**"]
    for col in fwd_cols:
        s = event_returns[col].dropna()
        if s.empty:
            parts.append(f"- `{col}`: no completed observations.")
            continue
        skew = float(s.skew()) if s.size > 2 else float("nan")
        std = float(s.std(ddof=1)) if s.size > 1 else float("nan")
        mean = float(s.mean())
        median = float(s.median())
        line = (
            f"- `{col}` (N={s.size:,}): mean `{_fmt_pct(mean)}`, median `{_fmt_pct(median)}`, "
            f"std `{std * 100:.2f}%`, skew `{skew:+.2f}` → {_classify_skew(skew)}."
        )
        if pd.notna(std) and abs(mean) > 0 and std > 3 * abs(mean):
            line += " Dispersion is high relative to the mean; do not interpret the average alone."
        parts.append(line)

    parts.append("")
    parts.append(
        "_Box plots show the median (typical event outcome) and interquartile range; whiskers and outliers "
        "expose tail risk and extreme winners/losers. Histograms show the full shape — right skew points to "
        "large positive outliers, left skew points to crash-tail risk._"
    )
    return "\n".join(parts)


def interpret_event_details(event_returns: pd.DataFrame) -> str:
    if event_returns is None or event_returns.empty:
        return "_No events to detail._"

    parts = [f"**Event details** — {len(event_returns):,} events."]
    if "event_direction" in event_returns.columns:
        counts = event_returns["event_direction"].value_counts()
        ups = int(counts.get("upside", 0))
        downs = int(counts.get("downside", 0))
        parts.append(f"- Direction split: **{ups:,} upside**, **{downs:,} downside**.")

    long_horizon_cols = [c for c in event_returns.columns if c.startswith("fwd_")]
    if long_horizon_cols:
        # Pick the longest horizon present for "winner / loser" attribution.
        h = max(int(c.split("_")[1]) for c in long_horizon_cols)
        col = f"fwd_{h}"
        s = event_returns[col].dropna()
        if not s.empty:
            best_idx = s.idxmax()
            worst_idx = s.idxmin()
            parts.append(
                f"- Largest winner ({h} bars): `{_fmt_pct(s.loc[best_idx])}` on "
                f"{pd.Timestamp(best_idx).date()}."
            )
            parts.append(
                f"- Worst loser ({h} bars): `{_fmt_pct(s.loc[worst_idx])}` on "
                f"{pd.Timestamp(worst_idx).date()}."
            )

    most_recent = event_returns.index.max()
    if pd.notna(most_recent):
        parts.append(f"- Most recent event: **{pd.Timestamp(most_recent).date()}**.")
        if "event_direction" in event_returns.columns:
            try:
                parts.append(
                    f"  Direction: **{event_returns.loc[most_recent, 'event_direction']}**."
                )
            except KeyError:
                pass

    parts.append("")
    parts.append(
        "_Direction matters. Upside extension and downside extension should not be blindly combined "
        "because they represent different regimes._"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Bottom combined report
# ---------------------------------------------------------------------------

def build_bottom_report(
    *,
    meta: RunMetadata,
    snapshot: pd.DataFrame | None,
    focus_dev_series: pd.Series | None,
    event_summary: pd.DataFrame | None,
    comparison: pd.DataFrame | None,
    event_returns: pd.DataFrame | None,
    event_count: int,
    direction: str,
) -> str:
    """Aggregated end-of-page research report with 8 standard sections."""
    sections: list[str] = ["## Research Report"]

    # 1. Current State
    sections.append("### 1. Current State")
    if snapshot is not None and not snapshot.empty:
        focus_row = snapshot[snapshot["period"] == meta.focus_length]
        if not focus_row.empty:
            r = focus_row.iloc[0]
            dev = r.get("% dev")
            pctile = r.get("rolling %ile")
            tail = r.get("rolling tail")
            extension = "upside" if pd.notna(dev) and dev > 0 else "downside" if pd.notna(dev) and dev < 0 else "flat"
            sections.append(
                f"{meta.ma_type} {meta.focus_length} on {meta.ticker}: deviation `{_safe_fmt_signed_pct(dev)}`, "
                f"rolling %ile `{pctile:.2f}` (rolling-window={meta.rolling_window}), "
                f"tail probability `{tail:.2f}%`. Current state = **{extension} extension**."
            )
        else:
            sections.append("Focus MA not in the snapshot ladder.")
    else:
        sections.append("No snapshot available.")

    # 2. Rarity / Tail Risk
    sections.append("### 2. Rarity / Tail Risk")
    if snapshot is not None and not snapshot.empty:
        focus_row = snapshot[snapshot["period"] == meta.focus_length]
        if not focus_row.empty:
            tail = float(focus_row.iloc[0].get("rolling tail", float("nan")))
            if pd.isna(tail):
                sections.append("Tail probability unavailable.")
            elif tail <= 2:
                sections.append(
                    f"Tail probability `{tail:.2f}%` — **very rare** in the recent rolling regime."
                )
            elif tail <= 5:
                sections.append(f"Tail probability `{tail:.2f}%` — **rare** in the recent rolling regime.")
            elif tail <= 10:
                sections.append(f"Tail probability `{tail:.2f}%` — elevated but not extremely rare.")
            else:
                sections.append(f"Tail probability `{tail:.2f}%` — within normal range.")

    # 3. Event Study Findings
    sections.append("### 3. Event Study Findings")
    sections.append(interpret_event_summary(event_summary, event_count))

    # 4. Event vs Normal Baseline
    sections.append("### 4. Event vs Normal Baseline Findings")
    sections.append(interpret_event_vs_normal(comparison))

    # 5. Distribution / Skew Findings
    sections.append("### 5. Distribution / Skew Findings")
    if focus_dev_series is not None and not focus_dev_series.dropna().empty:
        sections.append(interpret_distance_distribution(focus_dev_series))
    if event_returns is not None and not event_returns.empty:
        sections.append(interpret_forward_distributions(event_returns))

    # 6. Practical Interpretation
    sections.append("### 6. Practical Interpretation")
    sections.append(_practical_interpretation(comparison, direction))

    # 7. Limitations
    sections.append("### 7. Limitations")
    sections.append(_limitations_block(direction))

    # 8. Recommended Next Tests
    sections.append("### 8. Recommended Next Tests")
    sections.append(_recommended_next_tests(meta))

    return "\n\n".join(sections)


def _practical_interpretation(comparison: pd.DataFrame | None, direction: str) -> str:
    if comparison is None or comparison.empty:
        return "Not enough data to draw a practical interpretation."

    horizons = comparison.set_index("horizon")
    short_edges_pp = []
    long_edge_pp = float("nan")
    for h in (5, 10, 21):
        if h in horizons.index:
            short_edges_pp.append(_decimal_edge_to_pp(horizons.at[h, "Avg Edge"]))
    if 63 in horizons.index:
        long_edge_pp = _decimal_edge_to_pp(horizons.at[63, "Avg Edge"])

    bullets: list[str] = []
    if short_edges_pp and all(pd.notna(x) for x in short_edges_pp):
        avg_short = float(np.mean(short_edges_pp))
        if avg_short < 0.10:
            bullets.append(
                "Short-horizon (5/10/21) average edges are economically small or negative — "
                "the event condition does not provide a clear short-term timing edge."
            )
    if pd.notna(long_edge_pp):
        if abs(long_edge_pp) >= 0.50:
            verb = "exceeds" if long_edge_pp > 0 else "underperforms"
            bullets.append(
                f"At 63 bars the event condition {verb} the unconditional baseline by `{long_edge_pp:+.2f} pp`. "
                "Treat this as a **medium-term regime signal**, not a short-term entry trigger."
            )
        else:
            bullets.append(
                f"63-bar edge `{long_edge_pp:+.2f} pp` is small — no obvious medium-term edge versus baseline."
            )
    if direction == "both":
        bullets.append(
            "Direction = **both** mixes upside and downside regimes; rerun separately to avoid masking opposite effects."
        )
    if not bullets:
        bullets.append("No standout edges across horizons; treat as a neutral regime read.")
    return "\n".join(f"- {b}" for b in bullets)


def _limitations_block(direction: str) -> str:
    items = [
        "This is historical conditional analysis, not a guarantee of future returns.",
        "Event-study output is sensitive to the rolling window and percentile thresholds.",
        "Event de-duplication via the minimum event gap directly affects sample size and the comparison's statistical weight.",
        "Normal-baseline comparison is necessary because QQQ has positive long-term drift; raw event returns can look profitable while underperforming the baseline.",
        "yfinance data quality and adjusted-close methodology can shift historical readings; backfills change.",
        "Current-state similarity is **not** the same as a broad percentile event study unless explicitly filtered to a deviation/percentile bucket near the current value.",
    ]
    if direction == "both":
        items.insert(2, "`Direction = both` mixes upside and downside regimes into a single bucket.")
    return "\n".join(f"- {x}" for x in items)


def _recommended_next_tests(meta: RunMetadata) -> str:
    items = [
        "Run **upside-only** events separately, then **downside-only** events separately, and compare.",
        "Sweep rolling windows: 252 / 500 / 1000 — confirm the edge is not a window artifact.",
        f"Sweep focus MAs: 50 / 100 / 200 around current focus = {meta.focus_length}.",
        "Add a **current-state similarity** filter: only include historical bars whose deviation or percentile is near the current value.",
        "Layer optional volatility/breadth filters (e.g., realized-vol bucket, advance-decline regime) before declaring the edge robust.",
    ]
    return "\n".join(f"- {x}" for x in items)


# ---------------------------------------------------------------------------
# Convenience renderer used by the dashboard
# ---------------------------------------------------------------------------

def render_section(title: str, body: str, *, expanded: bool = True) -> tuple[str, str, bool]:
    """Return a tuple the streamlit layer can unpack into st.expander.

    Kept as a pure function so the interpretation module never imports streamlit.
    """
    return title, body, expanded
