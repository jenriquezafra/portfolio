from __future__ import annotations

import pandas as pd

from src.model_xgb import _apply_live_market_context_fallback


def test_live_market_context_fallback_extends_recent_dates_within_limit() -> None:
    clean_prices = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-02-20"), pd.Timestamp("2026-02-23")],
            "ticker": ["AAA", "AAA"],
            "adj_close": [100.0, 101.0],
        }
    )
    market_context = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-02-20")],
            "mkt_nasdaq_proxy_lag1": [0.01],
            "mkt_growth_proxy_lag1": [0.02],
        }
    )

    extended, meta = _apply_live_market_context_fallback(
        clean_prices_df=clean_prices,
        market_context_df=market_context,
        allow_fallback=True,
        max_stale_business_days=3,
    )

    assert extended is not None
    assert pd.Timestamp(extended["date"].max()) == pd.Timestamp("2026-02-23")
    row = extended[extended["date"] == pd.Timestamp("2026-02-23")].iloc[0]
    assert float(row["mkt_nasdaq_proxy_lag1"]) == 0.01
    assert float(row["mkt_growth_proxy_lag1"]) == 0.02

    assert meta["market_context_stale_business_days"] == 1
    assert meta["market_context_fallback_applied"] is True
