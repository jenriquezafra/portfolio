from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "19_auto_rebalance_if_due.py"
    spec = importlib.util.spec_from_file_location("auto_rebalance_if_due_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluate_schedule_no_previous_is_due() -> None:
    module = _load_module()
    out = module.evaluate_schedule(
        latest_clean_date=pd.Timestamp("2026-03-03"),
        last_applied_rebalance_date=None,
        rebalance_every_n_days=20,
        force=False,
    )
    assert out["due"] is True
    assert out["reason"] == "no_previous_applied_rebalance"


def test_evaluate_schedule_not_due_with_remaining_days() -> None:
    module = _load_module()
    out = module.evaluate_schedule(
        latest_clean_date=pd.Timestamp("2026-03-06"),
        last_applied_rebalance_date=pd.Timestamp("2026-03-02"),
        rebalance_every_n_days=5,
        force=False,
    )
    assert out["due"] is False
    assert out["reason"] == "cadence_not_due"
    assert out["business_days_since_last_applied"] == 4
    assert out["remaining_business_days"] == 1
    assert out["next_due_date"] == "2026-03-09"


def test_evaluate_schedule_due_when_elapsed_reaches_cadence() -> None:
    module = _load_module()
    out = module.evaluate_schedule(
        latest_clean_date=pd.Timestamp("2026-03-09"),
        last_applied_rebalance_date=pd.Timestamp("2026-03-02"),
        rebalance_every_n_days=5,
        force=False,
    )
    assert out["due"] is True
    assert out["reason"] == "cadence_due"
    assert out["business_days_since_last_applied"] == 5
    assert out["remaining_business_days"] == 0
    assert out["next_due_date"] == "2026-03-09"


def test_list_rebalance_summary_paths_filters_latest_and_history(tmp_path: Path) -> None:
    module = _load_module()
    keep = tmp_path / "rebalance_2026-03-02_20260302_231456_summary.json"
    keep.write_text("{}", encoding="utf-8")
    (tmp_path / "rebalance_latest_summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "rebalance_history_summary.json").write_text("{}", encoding="utf-8")

    paths = module._list_rebalance_summary_paths(tmp_path)
    assert [p.name for p in paths] == ["rebalance_2026-03-02_20260302_231456_summary.json"]


def test_load_last_applied_rebalance_picks_latest_timestamp(tmp_path: Path) -> None:
    module = _load_module()
    p1 = tmp_path / "rebalance_2026-02-23_20260223_231450_summary.json"
    p2 = tmp_path / "rebalance_2026-03-02_20260302_231456_summary.json"
    p3 = tmp_path / "rebalance_2026-03-01_20260301_120000_summary.json"

    p1.write_text(
        json.dumps(
            {
                "apply": True,
                "timestamp_utc": "2026-02-23T23:14:50+00:00",
                "rebalance_date": "2026-02-23",
                "broker": "ibkr",
                "n_orders": 10,
            }
        ),
        encoding="utf-8",
    )
    p2.write_text(
        json.dumps(
            {
                "apply": True,
                "timestamp_utc": "2026-03-02T23:15:01+00:00",
                "rebalance_date": "2026-03-02",
                "broker": "ibkr",
                "n_orders": 20,
            }
        ),
        encoding="utf-8",
    )
    p3.write_text(
        json.dumps(
            {
                "apply": False,
                "timestamp_utc": "2026-03-01T12:00:00+00:00",
                "rebalance_date": "2026-03-01",
                "broker": "ibkr",
                "n_orders": 5,
            }
        ),
        encoding="utf-8",
    )

    latest = module._load_last_applied_rebalance(tmp_path)
    assert latest is not None
    assert latest["rebalance_date"] == "2026-03-02"
    assert latest["n_orders"] == 20
