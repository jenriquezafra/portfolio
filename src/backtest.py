from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import load_yaml
from src.optimizer import apply_turnover_cap, optimize_mean_variance_long_only
from src.risk import build_daily_returns, estimate_covariance_matrix, pivot_returns


def _get_rebalance_dates(
    available_dates: pd.Series,
    frequency: str,
    every_n_days: int | None = None,
) -> list[pd.Timestamp]:
    dates = pd.Series(pd.to_datetime(available_dates).sort_values().unique())
    freq = frequency.lower()

    if freq == "daily":
        return dates.tolist()
    if freq == "weekly":
        return dates.groupby(dates.dt.to_period("W-FRI")).max().sort_values().tolist()
    if freq == "monthly":
        return dates.groupby(dates.dt.to_period("M")).max().sort_values().tolist()
    if freq == "every_n_days":
        if every_n_days is None or every_n_days <= 0:
            raise ValueError("`rebalance_every_n_days` must be positive for `every_n_days` frequency.")
        return dates.iloc[every_n_days - 1 :: every_n_days].tolist()

    raise ValueError(f"Unsupported rebalance frequency: {frequency}")


def _compute_summary(daily_returns: pd.DataFrame, rebalance_log: pd.DataFrame) -> dict[str, Any]:
    if daily_returns.empty:
        raise ValueError("Backtest produced no daily returns.")

    ret = daily_returns["portfolio_return_net"].astype(float)
    equity = (1.0 + ret).cumprod()
    n_days = len(ret)

    total_return = float(equity.iloc[-1] - 1.0)
    ann_return = float((equity.iloc[-1] ** (252.0 / n_days)) - 1.0) if n_days > 0 else 0.0
    ann_vol = float(ret.std(ddof=1) * np.sqrt(252.0)) if n_days > 1 else 0.0
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else None

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    max_drawdown = float(drawdown.min())

    return {
        "start_date": str(daily_returns["date"].min().date()),
        "end_date": str(daily_returns["date"].max().date()),
        "n_days": int(n_days),
        "n_rebalances": int(rebalance_log["rebalance_date"].nunique()),
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "average_turnover": float(rebalance_log["turnover"].mean()) if not rebalance_log.empty else 0.0,
        "total_cost_bps_paid": float(rebalance_log["trade_cost_bps_paid"].sum()) if not rebalance_log.empty else 0.0,
    }


def run_backtest(
    config_data_path: Path,
    config_backtest_path: Path,
    config_execution_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], Path, Path, Path, Path]:
    data_cfg = load_yaml(config_data_path)
    back_cfg = load_yaml(config_backtest_path)
    exec_cfg = load_yaml(config_execution_path)

    data_section = data_cfg.get("data")
    labels_section = data_cfg.get("labels", {})
    backtest_section = back_cfg.get("backtest")
    risk_controls = exec_cfg.get("risk_controls", {})

    if not isinstance(data_section, dict):
        raise ValueError("Missing `data` section in config_data.yaml")
    if not isinstance(backtest_section, dict):
        raise ValueError("Missing `backtest` section in config_backtest.yaml")
    if not isinstance(risk_controls, dict):
        raise ValueError("Missing `risk_controls` section in config_execution.yaml")

    clean_rel = data_section.get("output_clean_path")
    target_column = labels_section.get("target_column", "fwd_return_5d")

    rebalance_frequency = backtest_section.get("rebalance_frequency", "monthly")
    rebalance_every_n_days = backtest_section.get("rebalance_every_n_days")
    risk_lookback_days = backtest_section.get("risk_lookback_days", 60)
    risk_shrinkage = backtest_section.get("risk_shrinkage", 0.10)

    costs_cfg = backtest_section.get("costs", {})
    constraints_cfg = backtest_section.get("constraints", {})
    objective_cfg = backtest_section.get("objective", {})

    risk_aversion_lambda = objective_cfg.get("risk_aversion_lambda", 10.0)
    turnover_penalty_eta = objective_cfg.get("turnover_penalty_eta", 5.0)
    fully_invested = constraints_cfg.get("fully_invested", True)
    weight_max = constraints_cfg.get("weight_max", 0.20)

    bps_per_side = costs_cfg.get("bps_per_side", 5.0)
    slippage_bps = costs_cfg.get("slippage_bps", 2.0)
    total_cost_bps = float(bps_per_side) + float(slippage_bps)

    max_turnover = risk_controls.get("max_turnover_per_rebalance")
    if max_turnover is not None:
        max_turnover = float(max_turnover)

    if not isinstance(clean_rel, str):
        raise ValueError("`data.output_clean_path` must be a string path.")
    if not isinstance(target_column, str):
        raise ValueError("`labels.target_column` must be a string.")
    if not isinstance(rebalance_frequency, str):
        raise ValueError("`backtest.rebalance_frequency` must be a string.")
    if rebalance_every_n_days is not None and not isinstance(rebalance_every_n_days, int):
        raise ValueError("`backtest.rebalance_every_n_days` must be null or integer.")

    project_root = config_data_path.parents[1]
    clean_path = (project_root / clean_rel).resolve()
    predictions_path = (project_root / "outputs/models/predictions_oos.parquet").resolve()

    clean_prices = pd.read_parquet(clean_path)
    predictions = pd.read_parquet(predictions_path)
    predictions["date"] = pd.to_datetime(predictions["date"], utc=False).dt.tz_localize(None)
    predictions["ticker"] = predictions["ticker"].astype(str)

    returns_wide = pivot_returns(build_daily_returns(clean_prices))
    returns_dates = pd.Series(returns_wide.index.to_list())

    available_prediction_dates = pd.Series(predictions["date"].sort_values().unique())
    pred_date_list = available_prediction_dates.tolist()
    if rebalance_frequency.lower() == "every_n_days" and rebalance_every_n_days is not None:
        # If predictions are already sparse (e.g., model was trained on the same cadence), keep them as-is.
        # This avoids unintentionally downsampling 5-day predictions again to ~25-day spacing.
        gaps = available_prediction_dates.diff().dropna().dt.days
        median_gap = float(gaps.median()) if not gaps.empty else 0.0
        if median_gap >= max(1.0, float(rebalance_every_n_days) - 1.0):
            rebalance_dates = pred_date_list
        else:
            rebalance_dates = _get_rebalance_dates(
                available_dates=available_prediction_dates,
                frequency=rebalance_frequency,
                every_n_days=rebalance_every_n_days,
            )
    else:
        rebalance_dates = _get_rebalance_dates(
            available_dates=available_prediction_dates,
            frequency=rebalance_frequency,
            every_n_days=rebalance_every_n_days,
        )

    if not rebalance_dates:
        raise ValueError("No rebalance dates available after applying frequency filter.")

    all_tickers = sorted(clean_prices["ticker"].astype(str).unique().tolist())
    prev_weights = pd.Series(0.0, index=all_tickers, name="weight")
    weights_records: list[dict[str, Any]] = []
    rebalance_logs: list[dict[str, Any]] = []
    daily_records: list[dict[str, Any]] = []

    for idx, rebalance_date in enumerate(rebalance_dates):
        pred_slice = predictions[predictions["date"] == rebalance_date].copy()
        if pred_slice.empty:
            continue

        pred_slice = pred_slice.sort_values("ticker")
        pred_slice = pred_slice[pred_slice["ticker"].isin(returns_wide.columns)]
        if pred_slice.empty:
            continue

        tickers = pred_slice["ticker"].tolist()
        mu = pred_slice.set_index("ticker")["prediction"].astype(float)
        cov_daily = estimate_covariance_matrix(
            returns_wide=returns_wide,
            tickers=tickers,
            as_of_date=rebalance_date,
            lookback_days=int(risk_lookback_days),
            shrinkage=float(risk_shrinkage),
        )

        # Predictions are H-day forward returns; scale daily covariance to the same horizon.
        horizon_cov = cov_daily * float(max(1, int(data_cfg.get("labels", {}).get("horizon_days", 5))))

        prev_sub = prev_weights.reindex(tickers).fillna(0.0)
        target = optimize_mean_variance_long_only(
            expected_returns=mu,
            covariance=horizon_cov,
            prev_weights=prev_sub,
            risk_aversion_lambda=float(risk_aversion_lambda),
            turnover_penalty_eta=float(turnover_penalty_eta),
            weight_max=float(weight_max),
            fully_invested=bool(fully_invested),
        )
        final_w, turnover = apply_turnover_cap(
            target_weights=target,
            prev_weights=prev_sub,
            max_turnover_per_rebalance=max_turnover,
        )
        trade_cost = turnover * total_cost_bps / 10000.0

        next_rebalance = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else returns_dates.max()
        hold_dates = returns_dates[(returns_dates > rebalance_date) & (returns_dates <= next_rebalance)]
        if len(hold_dates) == 0:
            continue

        returns_slice = returns_wide.loc[hold_dates, tickers].fillna(0.0)
        gross_series = returns_slice.dot(final_w.reindex(tickers).fillna(0.0))

        net_series = gross_series.copy()
        first_day = hold_dates.iloc[0]
        net_series.loc[first_day] = net_series.loc[first_day] - trade_cost

        for d in hold_dates:
            daily_records.append(
                {
                    "date": pd.Timestamp(d),
                    "rebalance_date": pd.Timestamp(rebalance_date),
                    "portfolio_return_gross": float(gross_series.loc[d]),
                    "portfolio_return_net": float(net_series.loc[d]),
                }
            )

        for t in tickers:
            weights_records.append(
                {
                    "rebalance_date": pd.Timestamp(rebalance_date),
                    "ticker": t,
                    "weight": float(final_w.loc[t]),
                    "predicted_return": float(mu.loc[t]),
                }
            )

        rebalance_logs.append(
            {
                "rebalance_date": pd.Timestamp(rebalance_date),
                "n_assets": int(len(tickers)),
                "turnover": float(turnover),
                "trade_cost_bps_paid": float(turnover * total_cost_bps),
                "trade_cost_return": float(trade_cost),
            }
        )

        prev_weights = pd.Series(0.0, index=all_tickers, name="weight")
        prev_weights.loc[final_w.index] = final_w.values

    daily_returns = pd.DataFrame(daily_records).sort_values("date").reset_index(drop=True)
    weights_history = pd.DataFrame(weights_records).sort_values(["rebalance_date", "ticker"]).reset_index(drop=True)
    rebalance_log = pd.DataFrame(rebalance_logs).sort_values("rebalance_date").reset_index(drop=True)
    summary = _compute_summary(daily_returns=daily_returns, rebalance_log=rebalance_log)

    out_dir = (project_root / "outputs/backtests").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_path = out_dir / "daily_returns.parquet"
    weights_path = out_dir / "weights_history.parquet"
    rebalance_log_path = out_dir / "rebalance_log.parquet"
    summary_path = out_dir / "backtest_summary.json"

    daily_returns.to_parquet(daily_path, index=False)
    weights_history.to_parquet(weights_path, index=False)
    rebalance_log.to_parquet(rebalance_log_path, index=False)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    return daily_returns, weights_history, rebalance_log, summary, daily_path, weights_path, rebalance_log_path, summary_path
