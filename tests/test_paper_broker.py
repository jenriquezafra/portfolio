from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.execution.paper import PaperBroker


def test_paper_broker_uses_avg_cost_fallback_for_missing_price_symbols(tmp_path: Path) -> None:
    prices_path = tmp_path / "prices.parquet"
    state_path = tmp_path / "paper_state.json"

    prices = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-05")],
            "ticker": ["AAA", "BBB"],
            "adj_close": [100.0, 50.0],
        }
    )
    prices.to_parquet(prices_path, index=False)

    state = {
        "cash": 100000.0,
        "positions": {
            "OLD": {"quantity": 10.0, "avg_cost": 42.5},
            "AAA": {"quantity": 5.0, "avg_cost": 95.0},
        },
        "last_updated": None,
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    broker = PaperBroker(
        state_path=state_path,
        prices_path=prices_path,
        initial_cash=100000.0,
        as_of_date=pd.Timestamp("2024-01-05"),
    )
    broker.connect()
    try:
        out = broker.get_last_prices(["AAA", "OLD"])
    finally:
        broker.disconnect()

    assert "AAA" in out
    assert out["AAA"] == 100.0
    assert "OLD" in out
    assert out["OLD"] == 42.5
