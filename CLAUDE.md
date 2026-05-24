# CLAUDE.md — MA Distance Lab

This file gives Claude Code the persistent project context for working in this repository.

## Project Purpose

MA Distance Lab is the Python research companion to the user's TradingView moving-average distance indicator.

TradingView is the live monitoring layer. This project is the deeper research and validation engine.

The project studies:

1. How far price is from selected moving averages.
2. How rare that distance is versus rolling and expanding/full-history distributions.
3. Whether extreme distance-from-MA conditions historically lead to mean reversion, sideways consolidation, or momentum continuation.

The project should be treated as a research dashboard, not just a simple technical-indicator app.

## Core User Requirements

- The ticker must be manually selectable from the Streamlit left sidebar by entering any valid yfinance ticker, such as `NVDA`, `SPY`, `QQQ`, `GLD`, `BTC-USD`, or `^GSPC`.
- Historical data should default to `period="max"` in yfinance.
- Use adjusted close as the research price series when available.
- Fall back to close only when adjusted close is unavailable.
- Rolling window must be manually adjustable from the Streamlit left sidebar.
- Event-study triggers should use rolling percentile by default to avoid look-ahead bias.
- The project should support single-symbol deep research first. Multi-symbol research can be added later.

## Data Rules

Use yfinance for now.

Default data call behavior:

```python
yf.download(ticker, period="max", interval=interval, auto_adjust=False)
```

Research price priority:

1. `Adj Close` if available.
2. `Close` if `Adj Close` is unavailable.

Always make the selected price source visible in the dashboard metadata.

## Look-Ahead Bias Rules

These rules are critical.

- Rolling percentile can be used for historical event triggers.
- Expanding percentile can be used only if calculated up to each historical bar.
- Do not use final full-history percentile to classify old events.
- Full-history/expanding diagnostics may be displayed for the latest snapshot.
- Event-study results should record forward returns only after enough bars have passed.
- Do not let future values leak into signal/event classification.

## Moving Average Types

The project should support at least:

- SMA
- EMA / EWMA
- WMA
- RMA / Wilder
- DEMA
- TEMA
- HMA
- ZLEMA
- KAMA
- ALMA
- VWMA when volume data is available

## Feature Definitions

### Percent Distance From Moving Average

```text
% deviation = (price - moving_average) / moving_average * 100
```

### Rolling Z-Score

```text
rolling_z = (current_deviation - rolling_mean_deviation) / rolling_stdev_deviation
```

### Rolling Percentile

Percentile rank of the current MA-distance relative to the rolling window.

### Tail Probability

If current % deviation is positive:

```text
tail = 100 - percentile
```

If current % deviation is negative:

```text
tail = percentile
```

Interpretation:

- Positive deviation tail tells how rare it is to be more stretched to the upside.
- Negative deviation tail tells how rare it is to be more stretched to the downside.

### Extension Signals

Signals are not buy/sell signals. They are extension diagnostics.

Use both Z-score and percentile:

```text
↑↑ = Z > +2 OR percentile > 98
↑  = Z > +1 OR percentile > 90
↓↓ = Z < -2 OR percentile < 2
↓  = Z < -1 OR percentile < 10
—  = otherwise
```

## Event Study Logic

Event study should answer:

```text
When this MA-distance condition became extreme, what usually happened afterward?
```

Default trigger basis:

```text
Rolling percentile
```

Upside event:

```text
focus rolling percentile >= upside threshold
```

Downside event:

```text
focus rolling percentile <= downside threshold
```

Use a minimum event gap/cooldown so one stretched regime is not counted every bar.

Forward horizons:

- 5 bars
- 10 bars
- 21 bars
- 63 bars

For each horizon, compute:

- N
- Average forward return
- Median forward return
- % positive
- % negative
- Worst return
- Best return
- Optional forward max drawdown
- Optional forward max favorable excursion
- Optional return quantiles: 5%, 25%, 75%, 95%

## Dashboard Expectations

The Streamlit dashboard should have a professional research layout.

### Sidebar Controls

- Manual ticker input
- Interval selector: `1d`, `1wk`, `1mo`
- MA type selector
- MA length list
- Focus MA length
- Rolling window slider/input
- Event direction
- Upside percentile threshold
- Downside percentile threshold
- Minimum event gap

### Main Sections

1. Header summary
   - ticker
   - interval
   - period, default `max`
   - number of bars loaded
   - first date
   - last date
   - selected price source
   - selected MA type
   - focus MA length
   - rolling window

2. Price chart
   - adjusted close / selected research price
   - focus moving average
   - event markers

3. Latest snapshot table
   - MA period
   - % deviation
   - rolling Z
   - rolling percentile
   - rolling tail
   - rolling signal
   - expanding percentile
   - expanding tail
   - expanding signal

4. Focus MA diagnostics
   - % deviation over time
   - rolling Z-score
   - rolling percentile
   - tail probability

5. Event study table
   - Horizon
   - N
   - Avg
   - Median
   - % Pos
   - % Neg
   - Worst
   - Best

6. Event details table
   - event date
   - event direction
   - event percentile
   - event % deviation
   - forward returns by horizon

### Useful Visuals

Add these when feasible:

- Histogram of focus MA-distance with current value marker.
- Rolling percentile chart with 95, 98, 5, and 2 reference lines.
- Tail probability chart.
- Forward return distribution chart.
- Event markers on the price chart.

## Code Organization

Keep logic modular.

Do not dump everything into Streamlit.

Use:

- `src/ma_distance_lab/data.py` for data loading.
- `src/ma_distance_lab/ma.py` for moving averages.
- `src/ma_distance_lab/features.py` for features.
- `src/ma_distance_lab/events.py` for event studies.
- `src/ma_distance_lab/reporting.py` for formatting and reporting helpers.
- `src/ma_distance_lab/streamlit_app.py` for UI only.
- `src/ma_distance_lab/cli.py` for command-line runs.

## Quality Checks

Before finishing a coding task, run:

```bash
pytest -q
python -m py_compile src/ma_distance_lab/*.py
```

If Streamlit was modified, make sure this command is still valid:

```bash
streamlit run src/ma_distance_lab/streamlit_app.py
```

## Development Priorities

Priority order:

1. Correctness and no look-ahead bias.
2. Robust yfinance loading using `period="max"`.
3. Clean Streamlit controls and dashboard layout.
4. Event-study accuracy.
5. Useful visual diagnostics.
6. Code quality and tests.
7. UI polish.

## Communication Style for Claude Code

When summarizing changes, include:

- What changed.
- Why it changed.
- Tests run.
- Any assumptions.
- Any limitations.
- Recommended next steps.
