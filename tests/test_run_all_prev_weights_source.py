from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_run_all_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "06_run_all.py"
    spec = importlib.util.spec_from_file_location("run_all_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_clean_prices() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2026-03-01", "ticker": "AAPL", "adj_close": 49.0},
            {"date": "2026-03-02", "ticker": "AAPL", "adj_close": 50.0},
            {"date": "2026-03-01", "ticker": "MSFT", "adj_close": 99.0},
            {"date": "2026-03-02", "ticker": "MSFT", "adj_close": 100.0},
        ]
    )


def test_load_prev_weights_ignores_paper_state_when_broker_is_not_paper(tmp_path: Path) -> None:
    module = _load_run_all_module()

    state_path = tmp_path / "paper_state.json"
    state_path.write_text(
        json.dumps(
            {
                "cash": 100.0,
                "positions": {
                    "AAPL": {"quantity": 2.0, "avg_cost": 50.0},
                    "MSFT": {"quantity": 1.0, "avg_cost": 100.0},
                },
            }
        ),
        encoding="utf-8",
    )
    clean_prices = _make_clean_prices()
    execution_cfg = {
        "execution": {"mode": "paper", "broker": "ibkr"},
        "paper": {"state_path": str(state_path)},
    }

    weights, has_positions = module._load_prev_weights_from_paper_state(
        execution_cfg=execution_cfg,
        clean_prices=clean_prices,
        as_of_date=pd.Timestamp("2026-03-02"),
    )

    assert has_positions is False
    assert weights.empty


def test_load_prev_weights_uses_paper_state_when_paper_broker_active(tmp_path: Path) -> None:
    module = _load_run_all_module()

    state_path = tmp_path / "paper_state.json"
    state_path.write_text(
        json.dumps(
            {
                "cash": 100.0,
                "positions": {
                    "AAPL": {"quantity": 2.0, "avg_cost": 50.0},
                    "MSFT": {"quantity": 1.0, "avg_cost": 100.0},
                },
            }
        ),
        encoding="utf-8",
    )
    clean_prices = _make_clean_prices()
    execution_cfg = {
        "execution": {"mode": "paper", "broker": "paper"},
        "paper": {"state_path": str(state_path)},
    }

    weights, has_positions = module._load_prev_weights_from_paper_state(
        execution_cfg=execution_cfg,
        clean_prices=clean_prices,
        as_of_date=pd.Timestamp("2026-03-02"),
    )

    assert has_positions is True
    assert list(weights.sort_index().index) == ["AAPL", "MSFT"]
    assert abs(float(weights.loc["AAPL"]) - (100.0 / 300.0)) < 1e-12
    assert abs(float(weights.loc["MSFT"]) - (100.0 / 300.0)) < 1e-12
