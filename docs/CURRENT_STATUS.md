# Current Status

The scaffold is a working first pass for the MA Distance Lab.

## Implemented

- yfinance loader using `period='max'` by default.
- Adjusted-close research price series when available.
- Manual ticker entry from Streamlit left sidebar.
- Manually adjustable rolling window from Streamlit left sidebar.
- Moving average distance features.
- Rolling Z-score and rolling percentile.
- Expanding percentile and tail probability.
- Z + percentile extension signals.
- Forward-return event study.
- Streamlit dashboard starter.
- CLI runner.
- Basic tests using synthetic data.

## Next Focus

- Validate event study output across tickers and regimes.
- Add richer distribution charts.
- Add parameter sweeps.
- Add ticker universe support after single-symbol logic is stable.
