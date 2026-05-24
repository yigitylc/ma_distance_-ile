# MA Distance Lab

A Python research scaffold for studying how far price is from moving averages and what historically happens after statistically extreme MA-distance conditions.

This project is designed as the Python research companion to a TradingView Pine indicator. TradingView is useful for live chart monitoring; this Python project is for deeper, more precise research using yfinance adjusted-close data.

## Core Questions

1. How far is price from selected moving averages?
2. How rare is the current distance versus rolling and expanding/full-history distributions?
3. After similar extension events happened historically, what were the forward returns?
4. Does a high MA-distance condition behave more like mean reversion or momentum continuation?

## Current Scope

- Uses yfinance OHLCV data.
- Dashboard defaults to `period='max'` so it pulls the maximum available yfinance history for the manually entered ticker.
- Uses adjusted close as the research price series when available.
- Lets the ticker be manually entered from the left Streamlit sidebar, e.g. `NVDA`, `SPY`, `QQQ`, `GLD`, `BTC-USD`, `^GSPC`.
- Lets the rolling window be manually adjusted from the left Streamlit sidebar.
- Computes MA distance in percent.
- Supports common trend metrics: SMA, EMA/EWMA, WMA, RMA/Wilder, DEMA, TEMA, HMA, ZLEMA, KAMA, ALMA, and VWMA when volume is available.
- Computes rolling Z-score, rolling percentile, expanding percentile, tail probability, and extension signal.
- Runs forward-return event studies for selected MA-distance percentile thresholds.
- Includes a simple Streamlit app and CLI starter.

## Quick Start

```bash
# from the project root
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

Run a quick CLI event study using maximum yfinance history:

```bash
python -m ma_distance_lab.cli --ticker NVDA --period max --ma-type EMA --focus-length 50 --rolling-window 500
```

Run the dashboard:

```bash
streamlit run src/ma_distance_lab/streamlit_app.py
```

Run tests:

```bash
pytest -q
```

## Dashboard Controls

The left sidebar controls the main research variables:

- **YFinance Ticker:** manually enter any compatible ticker.
- **Historical Period:** fixed to `max` by default for maximum yfinance history.
- **Interval:** daily, weekly, or monthly.
- **MA Type:** EMA, SMA, WMA, RMA, DEMA, TEMA, HMA, ZLEMA, KAMA, ALMA, VWMA.
- **MA Lengths:** comma-separated lengths such as `10,20,21,50,100,200`.
- **Rolling Window:** manually adjustable; used for rolling Z-score, rolling percentile, tail probability, and event triggers.
- **Focus Length:** the specific MA length used for diagnostics and event study.

## Suggested Workflow

1. Start with one ticker, one MA type, and one focus length.
2. Compare rolling percentile versus expanding percentile.
3. Use event studies to check forward 5/10/21/63-bar returns after extension events.
4. Expand to a ticker universe only after single-symbol logic is validated.
5. Treat the TradingView indicator as the live monitoring layer and this Python project as the research/validation layer.

## Important Research Notes

- Rolling percentile is preferred for event triggers because it avoids using future data.
- Expanding/full-history percentile is useful for current diagnosis, but should be used carefully in historical event testing.
- High percentile does not automatically mean short. It means price is statistically extended.
- Event studies should distinguish between mean reversion and momentum continuation.

## Claude Code Context

This scaffold includes `CLAUDE.md` at the project root. Claude Code should use it as the persistent project context and implementation guide. `AGENTS.md` is kept as a broader agent/project instruction file.

