from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest import run_backtest


def _make_clean_prices(n_days: int = 130, n_tickers: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    rows: list[dict[str, object]] = []
    for i, ticker in enumerate(tickers):
        drift = 0.0001 * (i - n_tickers / 2)
        noise = rng.normal(0.0, 0.01, size=n_days)
        returns = drift + noise
        prices = 100.0 * np.cumprod(1.0 + returns)
        for d, px in zip(dates, prices):
            rows.append({"date": d, "ticker": ticker, "adj_close": float(px)})
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def _make_predictions(clean_prices: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
    df = clean_prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    g = df.groupby("ticker", group_keys=False)
    # Use lagged 5-day return as simple signal; target is true forward return.
    df["prediction"] = g["adj_close"].pct_change(periods=5).shift(1)
    df["fwd_return_5d_resid"] = g["adj_close"].shift(-horizon_days) / df["adj_close"] - 1.0
    out = df.dropna(subset=["prediction", "fwd_return_5d_resid"])[
        ["date", "ticker", "prediction", "fwd_return_5d_resid"]
    ].copy()
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def test_backtest_integration_and_trade_cost_application(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (project_root / "outputs" / "models").mkdir(parents=True, exist_ok=True)

    clean = _make_clean_prices(n_days=130, n_tickers=12)
    preds = _make_predictions(clean)
    clean_path = project_root / "data" / "processed" / "prices_clean.parquet"
    pred_path = project_root / "outputs" / "models" / "predictions_oos.parquet"
    clean.to_parquet(clean_path, index=False)
    preds.to_parquet(pred_path, index=False)

    config_data = {
        "data": {"output_clean_path": "data/processed/prices_clean.parquet"},
        "labels": {"horizon_days": 5, "target_column": "fwd_return_5d_resid"},
    }
    config_backtest = {
        "backtest": {
            "rebalance_frequency": "every_n_days",
            "rebalance_every_n_days": 5,
            "risk_lookback_days": 20,
            "risk_shrinkage": 0.10,
            "signal_transform": {"cross_sectional_rank_zscore": True},
            "portfolio": {
                "mode": "long_only",
                "long_quantile": 0.2,
                "short_quantile": 0.2,
                "gross_exposure_target": 1.0,
                "vol_lookback_days": 20,
                "beta_neutralization": {"enabled": False},
            },
            "costs": {"bps_per_side": 5.0, "slippage_bps": 2.0},
            "constraints": {"long_only": True, "fully_invested": True, "weight_max": 0.20},
            "objective": {"allocation_method": "score_over_vol"},
        }
    }
    config_execution = {"risk_controls": {"max_turnover_per_rebalance": 0.35}}

    cfg_data_path = project_root / "configs" / "config_data.yaml"
    cfg_back_path = project_root / "configs" / "config_backtest.yaml"
    cfg_exec_path = project_root / "configs" / "config_execution.yaml"
    cfg_data_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    cfg_back_path.write_text(yaml.safe_dump(config_backtest, sort_keys=False), encoding="utf-8")
    cfg_exec_path.write_text(yaml.safe_dump(config_execution, sort_keys=False), encoding="utf-8")

    daily_returns, weights_history, rebalance_log, summary, *_ = run_backtest(
        config_data_path=cfg_data_path,
        config_backtest_path=cfg_back_path,
        config_execution_path=cfg_exec_path,
    )

    assert not daily_returns.empty
    assert not rebalance_log.empty
    assert not weights_history.empty
    assert (pd.to_datetime(daily_returns["date"]) > pd.to_datetime(daily_returns["rebalance_date"])).all()
    assert summary["n_rebalances"] == int(rebalance_log["rebalance_date"].nunique())

    # Trade cost should only be applied on the first holding day after each rebalance.
    dr = daily_returns.copy()
    dr["date"] = pd.to_datetime(dr["date"])
    dr["rebalance_date"] = pd.to_datetime(dr["rebalance_date"])
    rl = rebalance_log.copy()
    rl["rebalance_date"] = pd.to_datetime(rl["rebalance_date"])

    for row in rl.itertuples(index=False):
        grp = dr[dr["rebalance_date"] == row.rebalance_date].sort_values("date")
        assert not grp.empty
        first = grp.iloc[0]
        first_day_cost = float(first["portfolio_return_gross"] - first["portfolio_return_net"])
        assert np.isclose(first_day_cost, float(row.trade_cost_return), atol=1e-10)
        if len(grp) > 1:
            rest = grp.iloc[1:]
            assert np.allclose(
                rest["portfolio_return_gross"].to_numpy(),
                rest["portfolio_return_net"].to_numpy(),
                atol=1e-12,
            )

    # New reporting artifact should always be produced.
    factor_report_path = project_root / "outputs" / "backtests" / "factor_exposure_report.json"
    assert factor_report_path.exists()
    factor_report = json.loads(factor_report_path.read_text(encoding="utf-8"))
    assert "ex_post" in factor_report
    diag_report_path = project_root / "outputs" / "backtests" / "factor_diagnostics_report.json"
    assert diag_report_path.exists()
    diag_report = json.loads(diag_report_path.read_text(encoding="utf-8"))
    assert "status" in diag_report
    assert "checks" in diag_report
