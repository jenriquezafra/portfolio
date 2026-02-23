from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_rebalance_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "04_rebalance.py"
    spec = importlib.util.spec_from_file_location("rebalance_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_business_days_between() -> None:
    module = _load_rebalance_module()
    assert module._business_days_between(pd.Timestamp("2026-02-20"), pd.Timestamp("2026-02-20")) == 0
    assert module._business_days_between(pd.Timestamp("2026-02-20"), pd.Timestamp("2026-02-23")) == 1
    assert module._business_days_between(pd.Timestamp("2026-02-20"), pd.Timestamp("2026-02-24")) == 2


def test_load_target_weights_from_run_all_recommendation(tmp_path: Path) -> None:
    module = _load_rebalance_module()

    weights_csv = tmp_path / "weights_live.csv"
    pd.DataFrame(
        [
            {"ticker": "MSFT", "weight": 0.60},
            {"ticker": "AAPL", "weight": 0.40},
        ]
    ).to_csv(weights_csv, index=False)

    recommendation = {
        "recommended_strategy": "long_only",
        "rebalance_date": "2026-01-14",
        "live_signal_date": "2026-02-23",
        "artifacts": {
            "recommended_weights_csv": str(weights_csv),
        },
    }
    recommendation_path = tmp_path / "recommendation.json"
    recommendation_path.write_text(json.dumps(recommendation), encoding="utf-8")

    signal_date, target, meta = module._load_target_weights_from_run_all(
        recommendation_path=recommendation_path,
        weights_csv=None,
    )

    assert signal_date == pd.Timestamp("2026-02-23")
    assert list(target.index) == ["AAPL", "MSFT"]
    assert float(target.loc["AAPL"]) == 0.40
    assert float(target.loc["MSFT"]) == 0.60
    assert meta["weights_source"] == "run_all"
    assert meta["recommended_strategy"] == "long_only"
