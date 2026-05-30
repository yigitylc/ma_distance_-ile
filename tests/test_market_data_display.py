from ma_distance_lab.market_data_display import (
    display_price_basis,
    display_price_source,
    display_provider_symbol,
)


def test_display_price_basis_labels_stooq_close_clearly():
    assert display_price_basis("stooq_close") == "Stooq close"
    assert display_price_basis("yfinance_adjusted_close") == "Yahoo adjusted close"


def test_display_price_source_labels_stooq_fallback_clearly():
    assert display_price_source("close_stooq_fallback") == "Stooq daily Close"
    assert display_price_source("close_auto_adjusted") == "Auto-adjusted Close"


def test_display_provider_symbol_hides_duplicate_input_symbol():
    assert display_provider_symbol("NVDA", "NVDA") == "Same as ticker"
    assert display_provider_symbol("NVDA", "nvda.us") == "nvda.us"
