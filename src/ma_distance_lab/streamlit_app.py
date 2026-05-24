from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ma_distance_lab.data import MarketDataRequest, fetch_ohlcv
from ma_distance_lab.events import (
    EventStudyConfig,
    compare_event_vs_baseline,
    run_baseline_study,
    run_event_study,
)
from ma_distance_lab.features import (
    FeatureConfig,
    build_ma_distance_features,
    focus_distance_series,
    latest_snapshot_table,
)
from ma_distance_lab.interpretation import (
    RunMetadata,
    build_bottom_report,
    interpret_distance_distribution,
    interpret_edge_bars,
    interpret_event_details,
    interpret_event_summary,
    interpret_event_vs_normal,
    interpret_focus_diagnostics,
    interpret_forward_distributions,
    interpret_latest_snapshot,
    interpret_price_chart,
    interpret_run_summary,
)
from ma_distance_lab.reporting import (
    format_comparison_table,
    format_event_details,
    format_event_summary,
    format_snapshot,
)


APP_VERSION = "no-autofetch-rate-limit-safe-2026-05-24"

if "active_ticker" not in st.session_state:
    st.session_state["active_ticker"] = None
if "has_submitted" not in st.session_state:
    st.session_state["has_submitted"] = False

st.set_page_config(page_title="MA Distance Lab", layout="wide")
st.title("MA Distance Lab")
st.caption(
    "Moving-average distance, rolling percentiles, tail probability, and forward-return event studies. "
    "Data is fetched from yfinance only after Run Analysis and uses adjusted close as the research price series."
)

with st.expander("EWMA reference (formula)", expanded=False):
    st.markdown(
        "**Exponentially weighted moving average (EWMA)** — a moving average where more recent "
        "observations are weighted more heavily. If `P_t` is the most recent data point then the "
        "EWMA is:\n\n"
        "`EWMA_t = (A × P_t) + (A × (1 - A) × P_{t-1}) + (A × (1 - A)^2 × P_{t-2}) "
        "+ (A × (1 - A)^3 × P_{t-3}) + ...`\n\n"
        "You can also calculate this recursively. If you have yesterday's EWMA `E_{t-1}`, then "
        "today's EWMA is:\n\n"
        "`EWMA_t = (A × P_t) + (E_{t-1} × (1 - A))`\n\n"
        "Used in the EWMAC trading rule and in the estimation of price volatility. In this "
        "dashboard, selecting **EMA** (or `EWMA`) under MA Type computes exactly this recursion "
        "via `pandas.Series.ewm(span=length, adjust=False)`, which sets `A = 2 / (length + 1)`."
    )

with st.sidebar:
    st.header("Data")
    with st.sidebar.form("ticker_form"):
        ticker_input = st.text_input(
            "YFinance Ticker",
            value=st.session_state.get("ticker_draft", "NVDA"),
            help="Enter one yfinance-compatible ticker, e.g. NVDA, SPY, QQQ, GLD, BTC-USD, ^GSPC, BRK-B.",
            key="ticker_draft",
        )
        submitted = st.form_submit_button("Run Analysis")
    if submitted:
        cleaned = ticker_input.strip().upper()
        if not cleaned:
            st.error("Ticker cannot be empty. Enter one ticker and click Run Analysis.")
            st.stop()
        st.session_state["active_ticker"] = cleaned
        st.session_state["has_submitted"] = True
        st.rerun()
    st.sidebar.caption(f"App version: {APP_VERSION}")
    if st.session_state.get("has_submitted", False):
        st.sidebar.caption(f"Current submitted ticker: {st.session_state['active_ticker']}")
    st.sidebar.caption("Data source: Yahoo Finance via yfinance. Public cloud apps may occasionally be rate-limited.")
    period = st.selectbox("Historical period", ["5y", "10y", "max"], index=1)
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    if st.sidebar.button("Refresh / Clear Cache"):
        st.cache_data.clear()
        st.rerun()

    st.header("MA / Stats")
    ma_type = st.selectbox(
        "MA Type",
        ["EMA", "SMA", "WMA", "RMA", "DEMA", "TEMA", "HMA", "ZLEMA", "KAMA", "ALMA", "VWMA"],
        index=0,
    )
    lengths_text = st.text_input("MA Lengths", value="10,20,21,50,100,200")
    rolling_window = st.slider(
        "Rolling Window",
        min_value=50,
        max_value=5000,
        value=500,
        step=50,
        help="Used for rolling Z-score, rolling percentile, tail probability, and event triggers.",
    )
    focus_length = st.number_input("Focus Length", min_value=1, value=50, step=1)

    st.header("Event Study")
    direction = st.selectbox("Direction", ["upside", "downside", "both"], index=0)
    upside_threshold = st.number_input("Upside Percentile Threshold", min_value=50.0, max_value=100.0, value=95.0, step=1.0)
    downside_threshold = st.number_input("Downside Percentile Threshold", min_value=0.0, max_value=50.0, value=5.0, step=1.0)
    min_event_gap = st.number_input("Minimum Event Gap", min_value=1, value=10, step=1)
    include_mfe_mdd = st.checkbox("Include MFE / MDD per horizon", value=False)
    include_quantiles_toggle = st.checkbox("Include forward-return quantiles", value=False)

    st.header("Commentary")
    show_commentary = st.checkbox("Show Commentary", value=True)
    commentary_placement = st.selectbox(
        "Commentary Placement",
        ["After each section", "Bottom report only", "Both"],
        index=0,
    )

show_inline = show_commentary and commentary_placement in ("After each section", "Both")
show_bottom = show_commentary and commentary_placement in ("Bottom report only", "Both")

if not st.session_state.get("has_submitted", False):
    st.info("Enter a ticker and click Run Analysis to start.")
    st.stop()

ticker = st.session_state["active_ticker"]


def _render_inline(title: str, body: str, *, expanded: bool = True) -> None:
    if not show_inline:
        return
    with st.expander(title, expanded=expanded):
        st.markdown(body)

try:
    lengths = tuple(int(x.strip()) for x in lengths_text.split(",") if x.strip())
except ValueError:
    st.error("MA Lengths must be comma-separated integers, e.g. 10,20,21,50,100,200.")
    st.stop()

focus_length = int(focus_length)
if not lengths:
    st.error("MA Lengths must include at least one positive integer.")
    st.stop()
if any(length <= 0 for length in lengths):
    st.error("MA Lengths must be positive integers.")
    st.stop()
focus_length_auto_added = focus_length not in lengths
if focus_length not in lengths:
    lengths = tuple(sorted(set(lengths + (focus_length,))))
if focus_length_auto_added:
    st.sidebar.info(f"Added focus length {focus_length} to MA Lengths for this run.")

if not ticker.strip():
    st.error("Ticker cannot be empty. Enter one ticker and click Run Analysis.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_and_compute(
    ticker: str,
    period: str,
    interval: str,
    ma_type: str,
    lengths: tuple[int, ...],
    rolling_window: int,
) -> pd.DataFrame:
    raw = fetch_ohlcv(
        MarketDataRequest(
            ticker=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
        )
    )
    return build_ma_distance_features(
        raw,
        config=FeatureConfig(ma_type=ma_type, lengths=lengths, rolling_window=rolling_window),
    )


try:
    features = load_and_compute(ticker, period, interval, ma_type, lengths, int(rolling_window))
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error("Unexpected error while loading ticker data.")
    with st.expander("Technical details"):
        st.exception(exc)
    st.stop()

st.sidebar.success(f"Loaded {ticker}")

prefix = f"{ma_type.lower()}_{focus_length}"
ma_col = f"{prefix}_ma"
dev_col = f"{prefix}_dev_pct"
pct_col = f"{prefix}_roll_pctile"
z_col = f"{prefix}_roll_z"
tail_col = f"{prefix}_roll_tail"

snapshot = latest_snapshot_table(features, ma_type, lengths)
valid_rows = features["price"].dropna().shape[0]
price_source = features.attrs.get("price_source", "—")
first_date = features.index.min()
last_date = features.index.max()

if features[pct_col].dropna().empty:
    st.warning(
        f"Not enough history for {ma_type} {focus_length} with rolling window {int(rolling_window)}. "
        f"Loaded {valid_rows:,} usable bars; reduce the rolling window or choose a ticker with more history."
    )

event_config = EventStudyConfig(
    ma_type=ma_type,
    focus_length=focus_length,
    direction=direction,
    upside_threshold=float(upside_threshold),
    downside_threshold=float(downside_threshold),
    min_event_gap=int(min_event_gap),
    include_mfe=bool(include_mfe_mdd),
    include_mdd=bool(include_mfe_mdd),
    include_quantiles=bool(include_quantiles_toggle),
)
summary, event_returns, events = run_event_study(features, config=event_config)
baseline_summary, _baseline_returns = run_baseline_study(features, config=event_config)
comparison = compare_event_vs_baseline(summary, baseline_summary)

st.subheader("Run summary")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Ticker", ticker)
m2.metric("Interval", interval)
m3.metric("Period", period)
m4.metric("Bars loaded", f"{valid_rows:,}")
m5.metric("Price source", price_source)
m6, m7, m8, m9, m10 = st.columns(5)
m6.metric("First date", str(pd.Timestamp(first_date).date()) if pd.notna(first_date) else "—")
m7.metric("Last date", str(pd.Timestamp(last_date).date()) if pd.notna(last_date) else "—")
m8.metric("MA type", ma_type)
m9.metric("Focus MA", str(focus_length))
m10.metric("Rolling window", str(int(rolling_window)))

run_meta = RunMetadata(
    ticker=ticker,
    interval=interval,
    period=period,
    bars_loaded=valid_rows,
    first_date=str(pd.Timestamp(first_date).date()) if pd.notna(first_date) else "—",
    last_date=str(pd.Timestamp(last_date).date()) if pd.notna(last_date) else "—",
    price_source=price_source,
    ma_type=ma_type,
    focus_length=focus_length,
    rolling_window=int(rolling_window),
)
_render_inline("Interpretation - Run summary", interpret_run_summary(run_meta))

c1, c2 = st.columns([1.25, 1])
with c1:
    st.subheader("Price and Focus MA")
    fig = go.Figure()

    have_ohlc = all(col in features.columns for col in ("open", "high", "low", "close"))
    if have_ohlc:
        raw_close = features["close"].replace(0, pd.NA)
        adj_ratio = (features["adj_close"] / raw_close).astype(float)
        adj_open = features["open"] * adj_ratio
        adj_high = features["high"] * adj_ratio
        adj_low = features["low"] * adj_ratio
        adj_close_series = features["adj_close"]
        fig.add_trace(
            go.Candlestick(
                x=features.index,
                open=adj_open,
                high=adj_high,
                low=adj_low,
                close=adj_close_series,
                name="OHLC (adjusted)",
                increasing_line_color="#2ca02c",
                decreasing_line_color="#d62728",
                increasing_fillcolor="#2ca02c",
                decreasing_fillcolor="#d62728",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=features.index,
                y=features["price"],
                name="Adjusted Close / Price",
                line=dict(color="#1f77b4"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=features.index,
            y=features[ma_col],
            name=f"{ma_type} {focus_length}",
            line=dict(color="#ff7f0e"),
        )
    )

    event_dates = events[events].index
    upside_dates = downside_dates = pd.Index([])
    if len(event_dates) > 0:
        pct_at_events = features.loc[event_dates, pct_col]
        upside_dates = pct_at_events[pct_at_events >= float(upside_threshold)].index
        downside_dates = pct_at_events[pct_at_events <= float(downside_threshold)].index
        if len(upside_dates) > 0:
            fig.add_trace(
                go.Scatter(
                    x=upside_dates,
                    y=features.loc[upside_dates, "price"],
                    mode="markers",
                    name="Upside extension (reversal warning)",
                    marker=dict(symbol="triangle-up", size=11, color="#d62728"),
                )
            )
        if len(downside_dates) > 0:
            fig.add_trace(
                go.Scatter(
                    x=downside_dates,
                    y=features.loc[downside_dates, "price"],
                    mode="markers",
                    name="Downside extension (reversal opportunity)",
                    marker=dict(symbol="triangle-down", size=11, color="#2ca02c"),
                )
            )
    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    _render_inline(
        "Interpretation - Price chart",
        interpret_price_chart(run_meta, len(event_dates), len(upside_dates), len(downside_dates)),
    )

with c2:
    st.subheader("Latest Snapshot")
    st.dataframe(format_snapshot(snapshot), use_container_width=True, hide_index=True)
    _render_inline(
        "Interpretation - Latest snapshot",
        interpret_latest_snapshot(snapshot, focus_length),
    )

st.subheader("Focus MA-Distance Diagnostics")
fig2 = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=("% Deviation", "Rolling Z-score", "Rolling Percentile", "Tail Probability"),
)
fig2.add_trace(go.Scatter(x=features.index, y=features[dev_col], line=dict(color="#1f77b4")), row=1, col=1)
fig2.add_trace(go.Scatter(x=features.index, y=features[z_col], line=dict(color="#ff7f0e")), row=2, col=1)
fig2.add_trace(go.Scatter(x=features.index, y=features[pct_col], line=dict(color="#2ca02c")), row=3, col=1)
for y in (95, 98, 5, 2):
    fig2.add_hline(y=y, line_dash="dot", line_color="gray", row=3, col=1)
if tail_col in features.columns:
    fig2.add_trace(go.Scatter(x=features.index, y=features[tail_col], line=dict(color="#9467bd")), row=4, col=1)
fig2.update_layout(height=900, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig2, use_container_width=True)

focus_diag_df = pd.DataFrame(
    {
        "dev_pct": features[dev_col],
        "roll_z": features[z_col],
        "roll_pctile": features[pct_col],
        "roll_tail": features[tail_col] if tail_col in features.columns else float("nan"),
    }
)
_render_inline(
    "Interpretation - Focus diagnostics",
    interpret_focus_diagnostics(focus_diag_df, focus_length),
)

st.subheader("Focus MA-distance distribution")
dev_series = focus_distance_series(features, ma_type, focus_length)
if dev_series.empty:
    st.info("No deviation data yet — increase data history or reduce rolling window.")
else:
    hist = px.histogram(dev_series, nbins=80, labels={"value": "% deviation"})
    current = float(dev_series.iloc[-1])
    hist.add_vline(
        x=current,
        line_color="red",
        annotation_text=f"now: {current:+.2f}%",
        annotation_position="top",
    )
    hist.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    st.plotly_chart(hist, use_container_width=True)
    _render_inline(
        "Interpretation - Distance distribution",
        interpret_distance_distribution(dev_series),
    )

st.subheader("Forward-Return Event Study")
st.caption(
    "Events are triggered by rolling percentile to reduce look-ahead bias. "
    "Toggle MFE / MDD or quantile checkboxes in the sidebar for additional columns."
)
st.dataframe(format_event_summary(summary), use_container_width=True, hide_index=True)
_render_inline(
    "Interpretation - Event study",
    interpret_event_summary(summary, int(events.sum())),
)

st.subheader("Event Returns vs Normal Forward Returns")
st.caption(
    "Normal = unconditional forward returns over all eligible bars (focus MA + rolling percentile available). "
    "Edge columns are event metric minus normal metric."
)
st.dataframe(format_comparison_table(comparison), use_container_width=True, hide_index=True)
_render_inline(
    "Interpretation - Event vs Normal",
    interpret_event_vs_normal(comparison),
)

with st.expander("Edge bar charts"):
    if comparison.empty:
        st.info("Not enough data to render comparison charts.")
    else:
        chart_long_avg = pd.melt(
            comparison[["horizon", "Event Avg", "Normal Avg"]],
            id_vars="horizon",
            var_name="Series",
            value_name="Return",
        )
        chart_long_med = pd.melt(
            comparison[["horizon", "Event Median", "Normal Median"]],
            id_vars="horizon",
            var_name="Series",
            value_name="Return",
        )
        chart_long_pos = pd.melt(
            comparison[["horizon", "Event % Positive", "Normal % Positive"]],
            id_vars="horizon",
            var_name="Series",
            value_name="Pct Positive",
        )
        cc1, cc2, cc3 = st.columns(3)
        for col, frame, ycol, title in (
            (cc1, chart_long_avg, "Return", "Avg return by horizon"),
            (cc2, chart_long_med, "Return", "Median return by horizon"),
            (cc3, chart_long_pos, "Pct Positive", "% Positive by horizon"),
        ):
            bar = px.bar(
                frame,
                x="horizon",
                y=ycol,
                color="Series",
                barmode="group",
                title=title,
            )
            bar.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            col.plotly_chart(bar, use_container_width=True)
        if show_inline:
            st.markdown(interpret_edge_bars(comparison))

st.subheader("Event details")
if event_returns.empty:
    st.info("No events met the trigger criteria. Loosen thresholds or shrink the rolling window.")
else:
    details = event_returns.reset_index()
    ordered = ["event_date", "event_direction", "event_percentile", "event_dev_pct"] + [
        c for c in event_returns.columns if c.startswith(("fwd_", "mfe_", "mdd_"))
    ]
    details = details[[c for c in ordered if c in details.columns]]
    st.dataframe(format_event_details(details), use_container_width=True, hide_index=True)
    _render_inline(
        "Interpretation - Event details",
        interpret_event_details(event_returns),
    )

    st.subheader("Forward-return distribution per horizon")
    horizons_present = [int(c.split("_")[1]) for c in event_returns.columns if c.startswith("fwd_")]
    if horizons_present:
        tab_box, tab_hist = st.tabs(["Box plots", "Histograms"])
        with tab_box:
            long = (
                event_returns[[f"fwd_{h}" for h in horizons_present]]
                .melt(var_name="horizon", value_name="ret")
                .dropna()
            )
            if not long.empty:
                box = px.box(long, x="horizon", y="ret", points="suspectedoutliers")
                box.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(box, use_container_width=True)
            else:
                st.info("Not enough completed forward windows to plot distributions yet.")
        with tab_hist:
            cols = st.columns(len(horizons_present))
            for col, h in zip(cols, horizons_present):
                series = event_returns[f"fwd_{h}"].dropna()
                if series.empty:
                    col.info(f"No data for fwd_{h}")
                    continue
                hfig = px.histogram(series, nbins=30, title=f"fwd_{h}")
                hfig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
                col.plotly_chart(hfig, use_container_width=True)
        _render_inline(
            "Interpretation - Forward-return distributions",
            interpret_forward_distributions(event_returns),
        )

if show_bottom:
    st.markdown("---")
    report_md = build_bottom_report(
        meta=run_meta,
        snapshot=snapshot,
        focus_dev_series=dev_series if not dev_series.empty else None,
        event_summary=summary,
        comparison=comparison,
        event_returns=event_returns,
        event_count=int(events.sum()),
        direction=direction,
    )
    st.markdown(report_md)
