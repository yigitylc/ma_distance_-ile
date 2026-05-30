from __future__ import annotations


def display_price_source(price_source: str) -> str:
    if price_source == "stooq_close":
        return "Stooq daily Close"
    if price_source == "close_auto_adjusted":
        return "Auto-adjusted Close"
    if price_source == "close_stooq_fallback":
        return "Stooq daily Close"
    if price_source.startswith("close_"):
        return "Close"
    return "Price"


def display_price_basis(price_basis: str) -> str:
    if price_basis == "yfinance_adjusted_close":
        return "Yahoo adjusted close"
    if price_basis == "yfinance_close":
        return "Yahoo close"
    if price_basis == "stooq_close":
        return "Stooq close"
    return price_basis or "Unknown"


def display_provider_symbol(ticker: str, provider_symbol: str) -> str:
    if not provider_symbol or provider_symbol.upper() == ticker.upper():
        return "Same as ticker"
    return provider_symbol
