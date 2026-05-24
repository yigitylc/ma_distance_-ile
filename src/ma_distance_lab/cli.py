from __future__ import annotations

import argparse

from .data import MarketDataRequest, fetch_ohlcv
from .events import EventStudyConfig, run_event_study
from .features import FeatureConfig, build_ma_distance_features, latest_snapshot_table
from .reporting import format_event_summary, format_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MA-distance percentile and forward-return event study.")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--period", default="max", help="yfinance period. Default is max. Ignored when --start/--end is supplied.")
    parser.add_argument("--start", default=None, help="Optional explicit start date. If omitted, yfinance period='max' is used.")
    parser.add_argument("--end", default=None)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--ma-type", default="EMA")
    parser.add_argument("--lengths", default="10,20,21,50,100,200")
    parser.add_argument("--focus-length", type=int, default=50)
    parser.add_argument("--rolling-window", type=int, default=500)
    parser.add_argument("--direction", choices=["upside", "downside", "both"], default="upside")
    parser.add_argument("--upside-threshold", type=float, default=95.0)
    parser.add_argument("--downside-threshold", type=float, default=5.0)
    parser.add_argument("--min-event-gap", type=int, default=10)
    parser.add_argument("--include-mfe", action="store_true", help="Include max favorable excursion per horizon.")
    parser.add_argument("--include-mdd", action="store_true", help="Include max drawdown per horizon.")
    parser.add_argument("--include-quantiles", action="store_true", help="Include forward-return quantile columns (5/25/75/95).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lengths = tuple(int(x.strip()) for x in args.lengths.split(",") if x.strip())

    df = fetch_ohlcv(
        MarketDataRequest(
            ticker=args.ticker,
            period=args.period,
            start=args.start,
            end=args.end,
            interval=args.interval,
            auto_adjust=False,
        )
    )
    features = build_ma_distance_features(
        df,
        config=FeatureConfig(ma_type=args.ma_type, lengths=lengths, rolling_window=args.rolling_window),
    )

    snapshot = latest_snapshot_table(features, args.ma_type, lengths)
    event_summary, event_returns, _events = run_event_study(
        features,
        config=EventStudyConfig(
            ma_type=args.ma_type,
            focus_length=args.focus_length,
            direction=args.direction,
            upside_threshold=args.upside_threshold,
            downside_threshold=args.downside_threshold,
            min_event_gap=args.min_event_gap,
            include_mfe=args.include_mfe,
            include_mdd=args.include_mdd,
            include_quantiles=args.include_quantiles,
        ),
    )

    print(
        f"\nTicker: {args.ticker.upper()} | Period: {args.period} | Interval: {args.interval} "
        f"| MA: {args.ma_type} | Rolling window: {args.rolling_window}"
    )
    print("Research price: adjusted close when available.\n")
    print("Latest MA-distance snapshot:")
    print(format_snapshot(snapshot).to_string(index=False))

    print(f"\nEvent study: focus {args.focus_length}, direction={args.direction}, events={len(event_returns)}")
    print(format_event_summary(event_summary).to_string(index=False))


if __name__ == "__main__":
    main()
