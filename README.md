# MA Distance Lab

A Python research for studying how far price is from moving averages and what historically happens after statistically extreme MA-distance conditions.

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



## Important Notes

- Rolling percentile is preferred for event triggers because it avoids using future data.
- Expanding/full-history percentile is useful for current diagnosis, but should be used carefully in historical event testing.
- High percentile does not automatically mean short. It means price is statistically extended.
- Event studies should distinguish between mean reversion and momentum continuation.


