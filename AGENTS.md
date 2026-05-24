# AGENTS.md — Claude Code Instructions

You are working on a trading research project called **MA Distance Lab**.

## Role
Act as a senior trader and Python research engineer. Be precise, avoid look-ahead bias, and write clean, testable code.

## Project Objective
Build a robust Python companion to a TradingView indicator that studies price distance from moving averages and historical forward returns after extreme extension events.

The project should answer:

1. How far is price from multiple moving averages?
2. How rare is that distance versus rolling and expanding/full-history distributions?
3. What happened after similar extreme conditions historically?
4. Does the extension imply mean reversion, momentum continuation, or regime change?

## Data Rules
- Use `yfinance` for now.
- Use adjusted close as the default price series.
- Preserve OHLCV columns if available.
- Do not silently mix adjusted and unadjusted prices.
- Make the adjusted-price logic explicit in code and documentation.

## Statistical Rules
- MA distance: `(price - moving_average) / moving_average * 100`.
- Rolling Z-score uses rolling mean and rolling sample stdev.
- Rolling percentile rank must use only the current rolling window.
- Expanding percentile rank must use only data available up to that date.
- Tail probability:
  - if distance >= 0: `100 - percentile`
  - if distance < 0: `percentile`
- Signals are extension signals, not buy/sell recommendations.

## Signal Rules
Use both Z-score and percentile:

- `↑↑` if `Z > +2` OR `percentile > 98`
- `↑`  if `Z > +1` OR `percentile > 90`
- `↓↓` if `Z < -2` OR `percentile < 2`
- `↓`  if `Z < -1` OR `percentile < 10`
- `—` otherwise

## Event Study Rules
- Use rolling percentile for historical event triggers by default.
- Do not use final full-history percentile as a historical event trigger because that can introduce look-ahead bias.
- Apply event de-duplication using `min_event_gap`.
- Forward returns should be calculated only when the horizon is available:
  - 5 bars
  - 10 bars
  - 21 bars
  - 63 bars
- Event-study table should report:
  - N
  - average forward return
  - median forward return
  - % positive
  - % negative
  - worst return
  - best return

## Engineering Rules
- Prefer modular functions over notebook-only logic.
- Add tests when changing statistical behavior.
- Keep UI/dashboard code separate from research logic.
- Avoid hidden assumptions. If a choice is made, document it.
- Do not add excessive dependencies without reason.
- Use type hints where practical.

## Documentation Rules
Update docs when you change behavior:
- `docs/CURRENT_STATUS.md`
- `docs/NEXT_TASKS.md`
- `docs/METHODOLOGY.md`

## Initial Milestones
1. Verify yfinance adjusted-close fetch works.
2. Validate all MA functions on synthetic data.
3. Validate rolling percentile and tail probability.
4. Validate event study on synthetic data.
5. Build Streamlit dashboard for one ticker.
6. Add multi-ticker batch research after single-ticker logic is stable.
