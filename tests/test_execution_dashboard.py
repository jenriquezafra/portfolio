from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_dashboard_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "18_execution_dashboard.py"
    spec = importlib.util.spec_from_file_location("execution_dashboard_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_daily_table_uses_last_run_per_day() -> None:
    module = _load_dashboard_module()
    history = pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-03-01T10:00:00Z",
                "account_equity_pre": 100000.0,
                "pre_cash_fraction": 0.20,
                "estimated_cash_after_fraction": 0.18,
                "target_weight_gross": 0.80,
                "investable_target_fraction": 0.98,
                "target_under_investment_fraction": 0.18,
                "n_orders": 10,
                "apply": True,
            },
            {
                "timestamp_utc": "2026-03-01T16:00:00Z",
                "account_equity_pre": 100500.0,
                "pre_cash_fraction": 0.15,
                "estimated_cash_after_fraction": 0.14,
                "target_weight_gross": 0.85,
                "investable_target_fraction": 0.98,
                "target_under_investment_fraction": 0.13,
                "n_orders": 12,
                "apply": False,
            },
            {
                "timestamp_utc": "2026-03-02T16:00:00Z",
                "account_equity_pre": 101000.0,
                "pre_cash_fraction": 0.10,
                "estimated_cash_after_fraction": 0.09,
                "target_weight_gross": 0.90,
                "investable_target_fraction": 0.98,
                "target_under_investment_fraction": 0.08,
                "n_orders": 8,
                "apply": True,
            },
        ]
    )

    daily = module._build_daily_table(history)

    assert len(daily) == 2
    assert str(daily.iloc[0]["run_date"]) == "2026-03-01"
    assert float(daily.iloc[0]["account_equity_pre"]) == 100500.0
    assert bool(daily.iloc[0]["apply"]) is False
    assert abs(float(daily.iloc[0]["equity_index"]) - 1.0) < 1e-12
    assert abs(float(daily.iloc[1]["equity_index"]) - (101000.0 / 100500.0)) < 1e-12


def test_plot_dashboard_writes_png(tmp_path: Path) -> None:
    module = _load_dashboard_module()
    daily = pd.DataFrame(
        [
            {
                "run_date": "2026-03-01",
                "timestamp_utc": "2026-03-01T16:00:00Z",
                "apply": False,
                "n_orders": 12,
                "account_equity_pre": 100500.0,
                "equity_index": 1.0,
                "cash_pre_pct": 15.0,
                "cash_after_orders_pct": 14.0,
                "target_gross_pct": 85.0,
                "investable_target_pct": 98.0,
                "target_under_investment_pct": 13.0,
            },
            {
                "run_date": "2026-03-02",
                "timestamp_utc": "2026-03-02T16:00:00Z",
                "apply": True,
                "n_orders": 8,
                "account_equity_pre": 101000.0,
                "equity_index": 1.0049751243781095,
                "cash_pre_pct": 10.0,
                "cash_after_orders_pct": 9.0,
                "target_gross_pct": 90.0,
                "investable_target_pct": 98.0,
                "target_under_investment_pct": 8.0,
            },
        ]
    )

    output_png = tmp_path / "dashboard.png"
    module._plot_dashboard(daily_df=daily, output_png=output_png)

    assert output_png.exists()
    assert output_png.stat().st_size > 0
