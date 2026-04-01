from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_history_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "17_execution_history.py"
    spec = importlib.util.spec_from_file_location("execution_history_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_history_row_computes_exposure_and_cash_estimate(tmp_path: Path) -> None:
    module = _load_history_module()

    weights_csv = tmp_path / "weights.csv"
    pd.DataFrame(
        [
            {"ticker": "AAPL", "weight": 0.60},
            {"ticker": "MSFT", "weight": 0.20},
        ]
    ).to_csv(weights_csv, index=False)

    orders_csv = tmp_path / "orders.csv"
    pd.DataFrame(
        [
            {"ticker": "AAPL", "delta_qty": 100, "price": 10.0},
            {"ticker": "MSFT", "delta_qty": -20, "price": 5.0},
        ]
    ).to_csv(orders_csv, index=False)

    recommendation_path = tmp_path / "recommendation.json"
    recommendation_path.write_text(
        json.dumps(
            {
                "live_weights_summary": {
                    "signal_gate_multiplier": 0.6,
                    "signal_quality_gate_enabled": True,
                    "turnover_cap_applied": True,
                    "used_existing_positions": True,
                }
            }
        ),
        encoding="utf-8",
    )

    summary = {
        "timestamp_utc": "2026-03-02T23:15:01+00:00",
        "rebalance_date": "2026-03-02",
        "broker": "ibkr",
        "mode": "paper",
        "apply": True,
        "portfolio_mode": "long_only",
        "n_orders": 2,
        "portfolio_turnover_estimate": 0.25,
        "account_equity": 100000.0,
        "account_cash": 100000.0,
        "min_cash_buffer": 0.02,
        "orders_file": str(orders_csv),
        "weights_source_meta": {
            "weights_source": "run_all",
            "weights_csv": str(weights_csv),
            "recommendation_path": str(recommendation_path),
        },
    }

    row = module._build_history_row(
        summary_path=tmp_path / "summary.json",
        summary=summary,
        recommendation_cache={},
    )

    assert abs(float(row["target_weight_gross"]) - 0.80) < 1e-12
    assert abs(float(row["target_weight_net"]) - 0.80) < 1e-12
    assert abs(float(row["planned_net_notional"]) - 900.0) < 1e-12
    assert abs(float(row["estimated_cash_after_orders"]) - 99100.0) < 1e-12
    assert abs(float(row["estimated_cash_after_fraction"]) - 0.991) < 1e-12
    assert abs(float(row["target_under_investment_fraction"]) - 0.18) < 1e-12
    assert float(row["run_all_signal_gate_multiplier"]) == 0.6
    assert row["run_all_used_existing_positions"] is True


def test_build_aggregate_summary_reports_reasons() -> None:
    module = _load_history_module()
    history = pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-03-01T00:00:00Z",
                "rebalance_date": "2026-03-01",
                "account_equity_pre": 100000.0,
                "apply": True,
                "n_orders": 10,
                "target_weight_gross": 0.95,
                "pre_cash_fraction": 0.05,
                "estimated_cash_after_fraction": 0.06,
                "target_under_investment_fraction": 0.01,
                "run_all_signal_gate_multiplier": 1.0,
                "run_all_turnover_cap_applied": False,
                "run_all_used_existing_positions": False,
            },
            {
                "timestamp_utc": "2026-03-02T00:00:00Z",
                "rebalance_date": "2026-03-02",
                "account_equity_pre": 102000.0,
                "apply": True,
                "n_orders": 12,
                "target_weight_gross": 0.75,
                "pre_cash_fraction": 0.25,
                "estimated_cash_after_fraction": 0.24,
                "target_under_investment_fraction": 0.23,
                "run_all_signal_gate_multiplier": 0.6,
                "run_all_turnover_cap_applied": True,
                "run_all_used_existing_positions": True,
            },
        ]
    )

    summary = module._build_aggregate_summary(history_df=history)

    reasons = summary.get("latest_possible_underinvestment_reasons", [])
    assert "target_weights_below_investable_fraction" in reasons
    assert "signal_quality_gate_de_risking" in reasons
    assert "live_recommendation_used_existing_positions" in reasons


def test_list_rebalance_summary_paths_filters_non_rebalance_files(tmp_path: Path) -> None:
    module = _load_history_module()

    keep = tmp_path / "rebalance_2026-03-02_20260302_231456_summary.json"
    keep.write_text("{}", encoding="utf-8")
    (tmp_path / "rebalance_latest_summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "rebalance_history_summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "other_summary.json").write_text("{}", encoding="utf-8")

    paths = module._list_rebalance_summary_paths(tmp_path)

    assert [p.name for p in paths] == ["rebalance_2026-03-02_20260302_231456_summary.json"]
