# Methodology

## Price Series

The project uses yfinance data. The default request uses `period='max'`, `interval='1d'`, and `auto_adjust=False`.

The research price series is:

1. `Adj Close` when yfinance provides it.
2. `Close` only when `Adj Close` is unavailable. If `auto_adjust=True` is later enabled, yfinance's `Close` is treated as adjusted.

## MA Distance

For each moving average:

```text
MA distance % = (price - MA) / MA * 100
```

Positive distance means price is above the moving average. Negative distance means price is below the moving average.

## Rolling Statistics

Rolling statistics use the manually selected rolling window from the Streamlit sidebar.

- Rolling Z-score compares current MA distance to the rolling mean/stdev.
- Rolling percentile ranks the current MA distance against the rolling window.
- Rolling tail probability converts percentile into directional rarity.

## Expanding / Full-History Statistics

Expanding percentile uses all observations available up to the current bar. This avoids look-ahead bias.

For current diagnosis, the final expanding percentile can be interpreted as the current reading versus all loaded history up to the latest bar.

## Event Study

Events are triggered by rolling percentile, not final full-history percentile.

This is intentional because using final full-history percentile to label old events would leak future information.

Forward returns are calculated after an accepted event over fixed horizons:

- 5 bars
- 10 bars
- 21 bars
- 63 bars

Cooldown logic prevents counting the same extension regime every bar.

## Normal QQQ Forward Returns (Unconditional Baseline)

Event-study returns by themselves answer "what happens after extreme MA-distance bars?" but they do not say whether those returns differ from ordinary drift. The unconditional baseline answers that.

**Definition.** Normal forward returns are the unconditional forward returns over all eligible historical bars in the same adjusted-close dataset used by the event study, computed without any event filter.

**Eligibility for a bar at index `i` and horizon `h`:**

1. The selected focus moving average is available at bar `i` (warmup complete).
2. The rolling percentile is available at bar `i` (rolling-window warmup complete).
3. The forward return at horizon `h` is observable, i.e. bar `i + h` exists in the loaded series.

The first two conditions match the warmup universe used by the event-study trigger so the baseline and the event study are drawn from the same eligible bars; the only difference is the event filter. The third condition is per-horizon, so a bar can contribute to `fwd_5` but not to `fwd_63` if it sits in the tail of the loaded series.

**Per-horizon statistics.** For each horizon `h` the baseline reports `N`, average, median, % positive, % negative, worst, best, standard deviation, skew, and the 5th / 25th / 75th / 95th percentiles of `(price[i + h] / price[i]) - 1`.

**Comparison table — Event Returns vs Normal Forward Returns.** The dashboard renders a side-by-side table with the columns `Horizon`, `Event N`, `Normal N`, `Event Avg`, `Normal Avg`, `Avg Edge`, `Event Median`, `Normal Median`, `Median Edge`, `Event % Positive`, `Normal % Positive`, `Hit-Rate Edge`, `Event Worst`, `Normal Worst`, `Event Best`, `Normal Best`. Edge columns are simple subtractions:

- `Avg Edge       = Event Avg       − Normal Avg`
- `Median Edge    = Event Median    − Normal Median`
- `Hit-Rate Edge  = Event % Positive − Normal % Positive`

A positive edge means the event-conditioned distribution outperformed unconditional drift on that metric for that horizon; a negative edge means it underperformed.

**Look-ahead bias.** The baseline uses the same kind of forward-return calculation as the event study (`price[i + h] / price[i] - 1`) and the same warmup universe (rolling percentile available). It introduces no additional look-ahead beyond the standard horizon shift, which is the realised return an investor would have observed `h` bars after entry.
