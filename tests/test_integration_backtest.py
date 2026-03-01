from __future__ import annotations

from copy import deepcopy
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


def test_backtest_risk_overlay_reduces_average_gross_exposure(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (project_root / "outputs" / "models").mkdir(parents=True, exist_ok=True)

    clean = _make_clean_prices(n_days=140, n_tickers=12)
    preds = _make_predictions(clean)
    clean_path = project_root / "data" / "processed" / "prices_clean.parquet"
    pred_path = project_root / "outputs" / "models" / "predictions_oos.parquet"
    clean.to_parquet(clean_path, index=False)
    preds.to_parquet(pred_path, index=False)

    config_data = {
        "data": {"output_clean_path": "data/processed/prices_clean.parquet"},
        "labels": {"horizon_days": 5, "target_column": "fwd_return_5d_resid"},
    }
    base_backtest = {
        "backtest": {
            "rebalance_frequency": "every_n_days",
            "rebalance_every_n_days": 5,
            "risk_lookback_days": 20,
            "risk_shrinkage": 0.10,
            "signal_transform": {"cross_sectional_rank_zscore": True},
            "portfolio": {
                "mode": "long_only",
                "vol_lookback_days": 20,
                "beta_neutralization": {"enabled": False},
            },
            "costs": {"bps_per_side": 5.0, "slippage_bps": 2.0},
            "constraints": {"long_only": True, "fully_invested": True, "weight_max": 0.20},
            "objective": {"allocation_method": "score_over_vol"},
        }
    }
    config_execution = {"risk_controls": {"max_turnover_per_rebalance": 0.35}}

    overlay_backtest = deepcopy(base_backtest)
    overlay_backtest["backtest"]["risk_overlay"] = {
        "enabled": True,
        "vol_target_annual": 0.03,
        "realized_vol_lookback_days": 20,
        "min_leverage": 0.20,
        "max_leverage": 1.00,
        "drawdown_de_risk": {
            "enabled": True,
            "drawdown_trigger": -0.05,
            "leverage_multiplier": 0.60,
        },
    }

    cfg_data_path = project_root / "configs" / "config_data.yaml"
    cfg_back_base_path = project_root / "configs" / "config_backtest.base.yaml"
    cfg_back_overlay_path = project_root / "configs" / "config_backtest.overlay.yaml"
    cfg_exec_path = project_root / "configs" / "config_execution.yaml"
    cfg_data_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    cfg_back_base_path.write_text(yaml.safe_dump(base_backtest, sort_keys=False), encoding="utf-8")
    cfg_back_overlay_path.write_text(yaml.safe_dump(overlay_backtest, sort_keys=False), encoding="utf-8")
    cfg_exec_path.write_text(yaml.safe_dump(config_execution, sort_keys=False), encoding="utf-8")

    _, _, _, summary_base, *_ = run_backtest(
        config_data_path=cfg_data_path,
        config_backtest_path=cfg_back_base_path,
        config_execution_path=cfg_exec_path,
    )
    _, _, rebalance_log_overlay, summary_overlay, *_ = run_backtest(
        config_data_path=cfg_data_path,
        config_backtest_path=cfg_back_overlay_path,
        config_execution_path=cfg_exec_path,
    )

    assert summary_overlay["average_gross_exposure"] < summary_base["average_gross_exposure"]
    assert "average_overlay_leverage" in summary_overlay
    assert summary_overlay["average_overlay_leverage"] <= 1.0
    assert "risk_overlay_leverage" in rebalance_log_overlay.columns
    lev = pd.to_numeric(rebalance_log_overlay["risk_overlay_leverage"], errors="coerce").dropna()
    assert not lev.empty
    assert float(lev.max()) <= 1.0 + 1e-12


def test_backtest_signal_stack_enabled_emits_component_columns(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (project_root / "outputs" / "models").mkdir(parents=True, exist_ok=True)

    clean = _make_clean_prices(n_days=160, n_tickers=12)
    preds = _make_predictions(clean)
    clean.to_parquet(project_root / "data" / "processed" / "prices_clean.parquet", index=False)
    preds.to_parquet(project_root / "outputs" / "models" / "predictions_oos.parquet", index=False)

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
                "vol_lookback_days": 20,
                "beta_neutralization": {"enabled": False},
            },
            "costs": {"bps_per_side": 5.0, "slippage_bps": 2.0},
            "constraints": {"long_only": True, "fully_invested": True, "weight_max": 0.20},
            "objective": {"allocation_method": "score_over_vol"},
            "signal_stack": {
                "enabled": True,
                "normalize_weights": True,
                "weights": {
                    "model_prediction": 1.0,
                    "momentum_residual": 0.25,
                    "reversal_regime": 0.15,
                    "vol_compression_breakout": 0.10,
                    "liquidity_impulse": 0.20,
                },
            },
        }
    }
    config_execution = {"risk_controls": {"max_turnover_per_rebalance": 0.35}}

    cfg_data_path = project_root / "configs" / "config_data.yaml"
    cfg_back_path = project_root / "configs" / "config_backtest.yaml"
    cfg_exec_path = project_root / "configs" / "config_execution.yaml"
    cfg_data_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    cfg_back_path.write_text(yaml.safe_dump(config_backtest, sort_keys=False), encoding="utf-8")
    cfg_exec_path.write_text(yaml.safe_dump(config_execution, sort_keys=False), encoding="utf-8")

    _, _, rebalance_log, summary, *_ = run_backtest(
        config_data_path=cfg_data_path,
        config_backtest_path=cfg_back_path,
        config_execution_path=cfg_exec_path,
    )

    assert summary["signal_stack_enabled"] is True
    assert "signal_stack_weights" in summary
    assert "signal_stack_contribution_stats" in summary
    for col in [
        "signal_model_component",
        "signal_momentum_component",
        "signal_reversal_component",
        "signal_vol_breakout_component",
        "signal_liquidity_component",
        "signal_composite",
    ]:
        assert col in rebalance_log.columns
        vals = pd.to_numeric(rebalance_log[col], errors="coerce").dropna()
        assert not vals.empty
        assert float(vals.min()) >= 0.0


def test_backtest_signal_stack_disabled_keeps_baseline_metrics(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (project_root / "outputs" / "models").mkdir(parents=True, exist_ok=True)

    clean = _make_clean_prices(n_days=150, n_tickers=10)
    preds = _make_predictions(clean)
    clean.to_parquet(project_root / "data" / "processed" / "prices_clean.parquet", index=False)
    preds.to_parquet(project_root / "outputs" / "models" / "predictions_oos.parquet", index=False)

    config_data = {
        "data": {"output_clean_path": "data/processed/prices_clean.parquet"},
        "labels": {"horizon_days": 5, "target_column": "fwd_return_5d_resid"},
    }
    base_backtest = {
        "backtest": {
            "rebalance_frequency": "every_n_days",
            "rebalance_every_n_days": 5,
            "risk_lookback_days": 20,
            "risk_shrinkage": 0.10,
            "signal_transform": {"cross_sectional_rank_zscore": True},
            "portfolio": {
                "mode": "long_only",
                "vol_lookback_days": 20,
                "beta_neutralization": {"enabled": False},
            },
            "costs": {"bps_per_side": 5.0, "slippage_bps": 2.0},
            "constraints": {"long_only": True, "fully_invested": True, "weight_max": 0.20},
            "objective": {"allocation_method": "score_over_vol"},
        }
    }
    disabled_stack_backtest = deepcopy(base_backtest)
    disabled_stack_backtest["backtest"]["signal_stack"] = {
        "enabled": False,
        "normalize_weights": True,
        "weights": {
            "model_prediction": 1.0,
            "momentum_residual": 0.0,
            "reversal_regime": 0.0,
            "vol_compression_breakout": 0.0,
            "liquidity_impulse": 0.0,
        },
    }
    config_execution = {"risk_controls": {"max_turnover_per_rebalance": 0.35}}

    cfg_data_path = project_root / "configs" / "config_data.yaml"
    cfg_back_base_path = project_root / "configs" / "config_backtest.base.yaml"
    cfg_back_disabled_path = project_root / "configs" / "config_backtest.disabled.yaml"
    cfg_exec_path = project_root / "configs" / "config_execution.yaml"
    cfg_data_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
    cfg_back_base_path.write_text(yaml.safe_dump(base_backtest, sort_keys=False), encoding="utf-8")
    cfg_back_disabled_path.write_text(yaml.safe_dump(disabled_stack_backtest, sort_keys=False), encoding="utf-8")
    cfg_exec_path.write_text(yaml.safe_dump(config_execution, sort_keys=False), encoding="utf-8")

    _, _, _, summary_base, *_ = run_backtest(
        config_data_path=cfg_data_path,
        config_backtest_path=cfg_back_base_path,
        config_execution_path=cfg_exec_path,
    )
    _, _, _, summary_disabled, *_ = run_backtest(
        config_data_path=cfg_data_path,
        config_backtest_path=cfg_back_disabled_path,
        config_execution_path=cfg_exec_path,
    )

    for key in ["annualized_return", "weekly_sharpe_ratio", "max_drawdown", "average_turnover"]:
        assert np.isclose(float(summary_disabled[key]), float(summary_base[key]), atol=1e-12)
