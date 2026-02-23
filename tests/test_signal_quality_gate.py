from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest import run_backtest


def _make_clean_prices(n_days: int = 130, n_tickers: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(77)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    rows: list[dict[str, object]] = []
    for i, ticker in enumerate(tickers):
        drift = 0.00008 * (i - n_tickers / 2)
        noise = rng.normal(0.0, 0.01, size=n_days)
        returns = drift + noise
        prices = 100.0 * np.cumprod(1.0 + returns)
        for d, px in zip(dates, prices):
            rows.append({"date": d, "ticker": ticker, "adj_close": float(px)})
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def _make_predictions(clean_prices: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
    df = clean_prices.copy().sort_values(["ticker", "date"])
    g = df.groupby("ticker", group_keys=False)
    df["prediction"] = g["adj_close"].pct_change(periods=5).shift(1)
    df["fwd_return_5d_resid"] = g["adj_close"].shift(-horizon_days) / df["adj_close"] - 1.0
    return (
        df.dropna(subset=["prediction", "fwd_return_5d_resid"])[["date", "ticker", "prediction", "fwd_return_5d_resid"]]
        .copy()
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def test_signal_quality_gate_scales_target_exposure(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (project_root / "outputs" / "models").mkdir(parents=True, exist_ok=True)

    clean = _make_clean_prices()
    preds = _make_predictions(clean)
    clean.to_parquet(project_root / "data" / "processed" / "prices_clean.parquet", index=False)
    preds.to_parquet(project_root / "outputs" / "models" / "predictions_oos.parquet", index=False)

    # Force poor signal history so gate activates after warmup.
    rebalance_dates = pd.Series(pd.to_datetime(preds["date"]).sort_values().unique()).iloc[4::5]
    training_log = pd.DataFrame(
        {
            "rebalance_date": rebalance_dates,
            "oos_cs_ic_spearman": [-0.10 for _ in range(len(rebalance_dates))],
        }
    )
    training_log.to_parquet(project_root / "outputs" / "models" / "training_log.parquet", index=False)

    config_data = {
        "data": {"output_clean_path": "data/processed/prices_clean.parquet"},
        "labels": {"horizon_days": 5, "target_column": "fwd_return_5d_resid"},
    }
    config_backtest = {
        "backtest": {
            "rebalance_frequency": "every_n_days",
            "rebalance_every_n_days": 5,
            "risk_lookback_days": 20,
            "risk_shrinkage": 0.1,
            "signal_transform": {"cross_sectional_rank_zscore": True},
            "portfolio": {
                "mode": "long_only",
                "vol_lookback_days": 20,
                "beta_neutralization": {"enabled": False},
            },
            "costs": {"bps_per_side": 5.0, "slippage_bps": 2.0},
            "constraints": {"long_only": True, "fully_invested": True, "weight_max": 0.2},
            "objective": {"allocation_method": "score_over_vol"},
            "signal_quality_gate": {
                "enabled": True,
                "metric": "oos_cs_ic_spearman",
                "lookback_rebalances": 3,
                "min_history_rebalances": 2,
                "threshold": 0.0,
                "bad_state_multiplier": 0.4,
            },
        }
    }
    config_execution = {"risk_controls": {"max_turnover_per_rebalance": 0.35}}

    cfg_data = project_root / "configs" / "config_data.yaml"
    cfg_back = project_root / "configs" / "config_backtest.yaml"
    cfg_exec = project_root / "configs" / "config_execution.yaml"
    cfg_data.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    cfg_back.write_text(yaml.safe_dump(config_backtest, sort_keys=False), encoding="utf-8")
    cfg_exec.write_text(yaml.safe_dump(config_execution, sort_keys=False), encoding="utf-8")

    _, _, rebalance_log, summary, *_ = run_backtest(
        config_data_path=cfg_data,
        config_backtest_path=cfg_back,
        config_execution_path=cfg_exec,
    )

    assert summary["signal_quality_gate_enabled"] is True
    assert "signal_gate_multiplier" in rebalance_log.columns
    mult = pd.to_numeric(rebalance_log["signal_gate_multiplier"], errors="coerce").dropna()
    assert not mult.empty
    assert float(mult.min()) <= 0.4 + 1e-12
    assert "average_signal_gate_multiplier" in summary
    assert summary["average_signal_gate_multiplier"] < 1.0

    # File-level output compatibility.
    summary_path = project_root / "outputs" / "backtests" / "backtest_summary.json"
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["signal_quality_gate_enabled"] is True
