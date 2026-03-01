from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_backtest
from src.data import load_yaml, run_fetch_data
from src.features import run_build_panel
from src.model_xgb import run_predict_live, run_train
from src.optimizer import (
    apply_turnover_cap,
    optimize_mean_variance_long_only,
    signal_to_long_only_weights,
    signal_to_market_neutral_weights,
)
from src.risk import build_daily_returns, estimate_covariance_matrix, pivot_returns
from src.signals import (
    ENGINEERED_SIGNAL_COLUMNS,
    build_composite_signal,
    build_price_volume_signal_panel,
    compute_signal_attribution_stats,
    parse_signal_stack_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full paper pipeline (fetch -> panel -> train -> backtest), compare long_only vs "
            "market_neutral, and export recommended strategy + latest target weights."
        )
    )
    parser.add_argument("--config-data", type=Path, default=Path("configs/config_data.yaml"))
    parser.add_argument("--config-model", type=Path, default=Path("configs/config_model.yaml"))
    parser.add_argument("--config-backtest", type=Path, default=Path("configs/config_backtest.yaml"))
    parser.add_argument("--config-execution", type=Path, default=Path("configs/config_execution.yaml"))
    parser.add_argument("--top-k", type=int, default=10, help="How many top weights to print/store.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/run_all"),
        help="Directory where run summary and recommendation files are saved.",
    )
    parser.add_argument(
        "--snapshot-baseline",
        action="store_true",
        help="Persist a dated baseline snapshot bundle for signal-stack promotion tracking.",
    )
    parser.add_argument(
        "--baseline-snapshot-dir",
        type=Path,
        default=Path("outputs/experiments/signal_stack_baseline"),
        help="Directory where dated baseline snapshot bundles are saved.",
    )
    return parser.parse_args()


def _prepare_mode_config(base_cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    back = cfg.setdefault("backtest", {})
    if not isinstance(back, dict):
        raise ValueError("`backtest` section must be a mapping.")

    portfolio = back.setdefault("portfolio", {})
    constraints = back.setdefault("constraints", {})
    objective = back.setdefault("objective", {})
    if not isinstance(portfolio, dict) or not isinstance(constraints, dict) or not isinstance(objective, dict):
        raise ValueError("`portfolio`, `constraints`, and `objective` must be mappings.")

    base_allocation = str(objective.get("allocation_method", "score_over_vol")).lower()
    if base_allocation not in {"score_over_vol", "mean_variance"}:
        base_allocation = "score_over_vol"
    beta_cfg = portfolio.setdefault("beta_neutralization", {})
    if not isinstance(beta_cfg, dict):
        raise ValueError("`portfolio.beta_neutralization` must be a mapping.")

    if mode == "long_only":
        objective["allocation_method"] = base_allocation
        portfolio["mode"] = "long_only"
        constraints["long_only"] = True
        constraints["fully_invested"] = True
        beta_cfg["enabled"] = False
        return cfg

    if mode == "market_neutral":
        # market_neutral currently supports score-based allocation only.
        objective["allocation_method"] = "score_over_vol"
        portfolio["mode"] = "market_neutral"
        constraints["long_only"] = False
        constraints["fully_invested"] = False
        portfolio.setdefault("gross_exposure_target", 1.0)
        portfolio.setdefault("long_quantile", 0.20)
        portfolio.setdefault("short_quantile", 0.20)
        # Keep beta neutralization setting from base config if already specified.
        beta_cfg.setdefault("enabled", False)
        return cfg

    raise ValueError("Unsupported mode. Expected `long_only` or `market_neutral`.")


def _score_summary(summary: dict[str, Any]) -> float:
    weekly_sharpe = summary.get("weekly_sharpe_ratio")
    if weekly_sharpe is None:
        return float("-inf")
    drawdown = abs(float(summary.get("max_drawdown", 0.0) or 0.0))
    total_cost_bps = float(summary.get("total_cost_bps_paid", 0.0) or 0.0)
    # Same scoring rule used in earlier selection: reward weekly Sharpe, penalize DD and cost.
    return float(weekly_sharpe) - (0.7 * drawdown) - (0.0001 * total_cost_bps)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _serialize_weights_row(df: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        out.append(
            {
                "ticker": str(row["ticker"]),
                "weight": float(row["weight"]),
            }
        )
    return out


def _build_promotion_comparison_schema(recommendation: dict[str, Any]) -> dict[str, Any]:
    strategy = str(recommendation.get("recommended_strategy", "long_only"))
    summary = recommendation.get("strategy_summaries", {}).get(strategy, {})
    annualized_return = float(summary.get("annualized_return", 0.0) or 0.0)
    weekly_sharpe = summary.get("weekly_sharpe_ratio")
    max_drawdown = float(summary.get("max_drawdown", 0.0) or 0.0)
    avg_turnover = float(summary.get("average_turnover", 0.0) or 0.0)
    return {
        "schema_version": "signal_stack_promotion_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline": {
            "recommended_strategy": strategy,
            "annualized_return": annualized_return,
            "weekly_sharpe_ratio": None if weekly_sharpe is None else float(weekly_sharpe),
            "max_drawdown": max_drawdown,
            "average_turnover": avg_turnover,
            "signal_gate_active_rate": summary.get("signal_gate_active_rate"),
        },
        "promotion_thresholds": {
            "annualized_return_min_relative_improvement": 0.05,
            "weekly_sharpe_max_drop": 0.05,
            "max_drawdown_max_worsening": 0.05,
            "average_turnover_max": 0.40,
        },
    }


def _write_baseline_snapshot_bundle(
    *,
    recommendation: dict[str, Any],
    config_data_path: Path,
    config_model_path: Path,
    config_backtest_path: Path,
    config_execution_path: Path,
    baseline_snapshot_dir: Path,
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = baseline_snapshot_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    comparison = _build_promotion_comparison_schema(recommendation)
    (run_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "recommendation.json").write_text(json.dumps(recommendation, indent=2, sort_keys=True), encoding="utf-8")

    config_map = {
        "config_data.yaml": config_data_path,
        "config_model.yaml": config_model_path,
        "config_backtest.yaml": config_backtest_path,
        "config_execution.yaml": config_execution_path,
    }
    for filename, src in config_map.items():
        shutil.copy2(src, run_dir / filename)

    latest_dir = baseline_snapshot_dir / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)
    return run_dir


def _rank_to_zscore(signal: pd.Series) -> pd.Series:
    ranked = signal.rank(method="average")
    std = float(ranked.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(0.0, index=signal.index, dtype=float)
    return ((ranked - float(ranked.mean())) / std).astype(float)


def _estimate_asset_volatility(
    returns_wide: pd.DataFrame,
    tickers: list[str],
    as_of_date: pd.Timestamp,
    lookback_days: int,
) -> pd.Series:
    history = returns_wide[returns_wide.index <= as_of_date]
    history = history.tail(max(2, int(lookback_days)))
    if history.empty:
        return pd.Series(1.0, index=tickers, dtype=float)

    vol = history[tickers].std(ddof=1).replace([np.inf, -np.inf], np.nan)
    median = float(vol[vol > 0].median()) if (vol > 0).any() else 1.0
    vol = vol.where(vol > 1e-12, median if median > 0 else 1.0).fillna(median if median > 0 else 1.0)
    return vol.astype(float)


def _build_signal_quality_lookup(
    training_log: pd.DataFrame,
    metric_col: str,
    lookback_rebalances: int,
    min_history_rebalances: int,
) -> dict[pd.Timestamp, float]:
    if training_log.empty or metric_col not in training_log.columns:
        return {}
    if lookback_rebalances <= 0 or min_history_rebalances <= 0:
        return {}

    log = training_log.copy()
    log["rebalance_date"] = pd.to_datetime(log["rebalance_date"], utc=False).dt.tz_localize(None)
    log = log.sort_values("rebalance_date").dropna(subset=["rebalance_date"]).reset_index(drop=True)
    metric_series = pd.to_numeric(log[metric_col], errors="coerce")

    out: dict[pd.Timestamp, float] = {}
    for i, row in log.iterrows():
        hist = metric_series.iloc[max(0, i - lookback_rebalances) : i].dropna()
        if len(hist) < min_history_rebalances:
            continue
        out[pd.Timestamp(row["rebalance_date"])] = float(hist.mean())
    return out


def _last_prices_at_or_before(
    clean_prices: pd.DataFrame,
    symbols: list[str],
    as_of_date: pd.Timestamp,
) -> dict[str, float]:
    if not symbols:
        return {}
    prices = clean_prices[["date", "ticker", "adj_close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], utc=False).dt.tz_localize(None)
    prices["ticker"] = prices["ticker"].astype(str)
    filtered = prices[(prices["ticker"].isin(symbols)) & (prices["date"] <= as_of_date)]
    if filtered.empty:
        return {}
    last = (
        filtered.sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .tail(1)[["ticker", "adj_close"]]
    )
    return {str(row["ticker"]): float(row["adj_close"]) for _, row in last.iterrows() if float(row["adj_close"]) > 0}


def _load_prev_weights_from_paper_state(
    execution_cfg: dict[str, Any],
    clean_prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> tuple[pd.Series, bool]:
    mode = str(execution_cfg.get("execution", {}).get("mode", "paper")).lower()
    paper_cfg = execution_cfg.get("paper", {})
    if mode != "paper" or not isinstance(paper_cfg, dict):
        return pd.Series(dtype=float), False

    state_rel = paper_cfg.get("state_path")
    if not isinstance(state_rel, str):
        return pd.Series(dtype=float), False

    state_path = Path(state_rel)
    if not state_path.is_absolute():
        state_path = (PROJECT_ROOT / state_path).resolve()
    if not state_path.exists():
        return pd.Series(dtype=float), False

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return pd.Series(dtype=float), False

    positions_raw = state.get("positions", {})
    if not isinstance(positions_raw, dict) or not positions_raw:
        return pd.Series(dtype=float), False

    symbols = [str(sym) for sym in positions_raw.keys()]
    prices = _last_prices_at_or_before(clean_prices=clean_prices, symbols=symbols, as_of_date=as_of_date)

    values: dict[str, float] = {}
    cash = float(state.get("cash", 0.0) or 0.0)
    total_positions = 0.0
    for sym, raw in positions_raw.items():
        if not isinstance(raw, dict):
            continue
        qty = float(raw.get("quantity", 0.0) or 0.0)
        if abs(qty) <= 1e-12:
            continue
        px = prices.get(str(sym))
        if px is None:
            avg_cost = raw.get("avg_cost")
            if avg_cost is None:
                continue
            px = float(avg_cost)
        values[str(sym)] = float(qty * float(px))
        total_positions += values[str(sym)]

    equity = float(cash + total_positions)
    if equity <= 1e-12 or not values:
        return pd.Series(dtype=float), False

    weights = pd.Series({sym: val / equity for sym, val in values.items()}, dtype=float)
    return weights, True


def _compute_live_weights(
    *,
    data_cfg: dict[str, Any],
    mode_backtest_cfg: dict[str, Any],
    execution_cfg: dict[str, Any],
    clean_prices: pd.DataFrame,
    training_log: pd.DataFrame,
    live_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    back = mode_backtest_cfg.get("backtest")
    if not isinstance(back, dict):
        raise ValueError("Missing `backtest` section for live weight computation.")

    signal_cfg = back.get("signal_transform", {})
    portfolio_cfg = back.get("portfolio", {})
    constraints_cfg = back.get("constraints", {})
    objective_cfg = back.get("objective", {})
    gate_cfg = back.get("signal_quality_gate", {})
    signal_stack_cfg = back.get("signal_stack", {})
    risk_controls = execution_cfg.get("risk_controls", {})
    labels_cfg = data_cfg.get("labels", {})
    if (
        not isinstance(signal_cfg, dict)
        or not isinstance(portfolio_cfg, dict)
        or not isinstance(constraints_cfg, dict)
        or not isinstance(objective_cfg, dict)
        or not isinstance(gate_cfg, dict)
        or not isinstance(signal_stack_cfg, dict)
        or not isinstance(risk_controls, dict)
        or not isinstance(labels_cfg, dict)
    ):
        raise ValueError("Invalid config sections for live weight computation.")

    use_rank_zscore = bool(signal_cfg.get("cross_sectional_rank_zscore", False))
    portfolio_mode = str(portfolio_cfg.get("mode", "long_only")).lower()
    allocation_method = str(objective_cfg.get("allocation_method", "score_over_vol")).lower()
    risk_lookback_days = int(back.get("risk_lookback_days", 60))
    risk_shrinkage = float(back.get("risk_shrinkage", 0.10))
    weight_max = float(constraints_cfg.get("weight_max", 0.20))
    fully_invested = bool(constraints_cfg.get("fully_invested", True))
    long_quantile = float(portfolio_cfg.get("long_quantile", 0.20))
    short_quantile = float(portfolio_cfg.get("short_quantile", 0.20))
    gross_exposure_target = float(portfolio_cfg.get("gross_exposure_target", 1.0))
    vol_lookback_days = int(portfolio_cfg.get("vol_lookback_days", risk_lookback_days))
    risk_aversion_lambda = float(objective_cfg.get("risk_aversion_lambda", 10.0))
    turnover_penalty_eta = float(objective_cfg.get("turnover_penalty_eta", 5.0))
    horizon_days = int(labels_cfg.get("horizon_days", 5))
    signal_stack_enabled = bool(signal_stack_cfg.get("enabled", False))
    signal_stack_weights, signal_stack_normalize_weights = parse_signal_stack_weights(signal_stack_cfg)

    live_df = live_predictions.copy()
    live_df["date"] = pd.to_datetime(live_df["date"], utc=False).dt.tz_localize(None)
    live_df["ticker"] = live_df["ticker"].astype(str)
    if live_df.empty:
        raise ValueError("Live predictions are empty.")

    live_date = pd.Timestamp(live_df["date"].max())
    live_slice = live_df[live_df["date"] == live_date].copy()
    live_slice = live_slice.sort_values("ticker")

    returns_wide = pivot_returns(build_daily_returns(clean_prices))
    live_slice = live_slice[live_slice["ticker"].isin(returns_wide.columns)]
    if live_slice.empty:
        raise ValueError("No live tickers overlap with available return history.")

    tickers = live_slice["ticker"].tolist()
    mu_raw = live_slice.set_index("ticker")["prediction"].astype(float)
    signal = _rank_to_zscore(mu_raw) if use_rank_zscore else mu_raw.copy()
    signal_attribution: dict[str, float] = {
        "signal_model_component": float(signal.abs().mean()),
        "signal_momentum_component": 0.0,
        "signal_reversal_component": 0.0,
        "signal_vol_breakout_component": 0.0,
        "signal_liquidity_component": 0.0,
        "signal_composite": float(signal.abs().mean()),
    }
    if signal_stack_enabled:
        engineered = build_price_volume_signal_panel(clean_prices=clean_prices)
        engineered["date"] = pd.to_datetime(engineered["date"], utc=False).dt.tz_localize(None)
        engineered["ticker"] = engineered["ticker"].astype(str)
        engineered_idx = engineered.set_index(["date", "ticker"]).sort_index()
        try:
            engineered_slice = engineered_idx.xs(live_date, level="date").reindex(tickers)
        except KeyError:
            engineered_slice = pd.DataFrame(index=tickers, columns=list(ENGINEERED_SIGNAL_COLUMNS), dtype=float)
        composite_signal, components, _ = build_composite_signal(
            model_signal=signal.reindex(tickers).fillna(0.0),
            engineered_signals=engineered_slice,
            weights=signal_stack_weights,
            normalize_weights=False,
        )
        signal = composite_signal
        signal_attribution = compute_signal_attribution_stats(
            components=components,
            weights=signal_stack_weights,
            composite=signal,
        )
    vol_est = _estimate_asset_volatility(
        returns_wide=returns_wide,
        tickers=tickers,
        as_of_date=live_date,
        lookback_days=vol_lookback_days,
    )

    prev_weights, has_prev_positions = _load_prev_weights_from_paper_state(
        execution_cfg=execution_cfg,
        clean_prices=clean_prices,
        as_of_date=live_date,
    )
    prev_sub = prev_weights.reindex(tickers).fillna(0.0)
    effective_turnover_penalty_eta = turnover_penalty_eta if has_prev_positions else 0.0

    if allocation_method == "mean_variance":
        cov_daily = estimate_covariance_matrix(
            returns_wide=returns_wide,
            tickers=tickers,
            as_of_date=live_date,
            lookback_days=risk_lookback_days,
            shrinkage=risk_shrinkage,
        )
        horizon_cov = cov_daily * float(max(1, horizon_days))
        target = optimize_mean_variance_long_only(
            expected_returns=signal,
            covariance=horizon_cov,
            prev_weights=prev_sub,
            risk_aversion_lambda=risk_aversion_lambda,
            turnover_penalty_eta=effective_turnover_penalty_eta,
            weight_max=weight_max,
            fully_invested=fully_invested,
        )
    else:
        if portfolio_mode == "long_only":
            target = signal_to_long_only_weights(
                signal=signal,
                volatility=vol_est,
                weight_max=weight_max,
                fully_invested=fully_invested,
            )
        else:
            target = signal_to_market_neutral_weights(
                signal=signal,
                volatility=vol_est,
                weight_max_abs=weight_max,
                gross_exposure_target=gross_exposure_target,
                long_quantile=long_quantile,
                short_quantile=short_quantile,
            )

    gate_enabled = bool(gate_cfg.get("enabled", False))
    gate_metric = str(gate_cfg.get("metric", "oos_cs_ic_spearman"))
    gate_lookback = int(gate_cfg.get("lookback_rebalances", 20))
    gate_min_history = int(gate_cfg.get("min_history_rebalances", 8))
    gate_threshold = float(gate_cfg.get("threshold", 0.0))
    gate_bad_state_multiplier = float(gate_cfg.get("bad_state_multiplier", 0.35))
    gate_metric_value: float | None = None
    gate_multiplier = 1.0

    if gate_enabled:
        gate_lookup = _build_signal_quality_lookup(
            training_log=training_log,
            metric_col=gate_metric,
            lookback_rebalances=gate_lookback,
            min_history_rebalances=gate_min_history,
        )
        gate_metric_value = gate_lookup.get(live_date)
        if gate_metric_value is None and gate_lookup:
            prior_dates = [d for d in gate_lookup.keys() if d <= live_date]
            if prior_dates:
                gate_metric_value = gate_lookup[max(prior_dates)]
        if gate_metric_value is not None and gate_metric_value < gate_threshold:
            gate_multiplier = gate_bad_state_multiplier
            target = target * float(gate_multiplier)

    max_turnover = risk_controls.get("max_turnover_per_rebalance")
    max_turnover = float(max_turnover) if max_turnover is not None else None
    # If there is no existing portfolio in paper state, publish full target weights.
    # Turnover cap is enforced only when transitioning from existing positions.
    if not has_prev_positions:
        max_turnover = None
    final_w, turnover = apply_turnover_cap(
        target_weights=target,
        prev_weights=prev_sub,
        max_turnover_per_rebalance=max_turnover,
    )

    out = final_w.sort_values(ascending=False).rename("weight").reset_index().rename(columns={"index": "ticker"})
    out["ticker"] = out["ticker"].astype(str)
    out["weight"] = out["weight"].astype(float)
    summary = {
        "live_date": str(live_date.date()),
        "portfolio_mode": portfolio_mode,
        "allocation_method": allocation_method,
        "used_existing_positions": bool(has_prev_positions),
        "turnover_cap_applied": bool(max_turnover is not None),
        "turnover": float(turnover),
        "gross_exposure": float(out["weight"].abs().sum()),
        "net_exposure": float(out["weight"].sum()),
        "n_positions": int(len(out)),
        "signal_quality_gate_enabled": bool(gate_enabled),
        "signal_gate_metric": gate_metric,
        "signal_gate_metric_value": None if gate_metric_value is None else float(gate_metric_value),
        "signal_gate_threshold": float(gate_threshold),
        "signal_gate_multiplier": float(gate_multiplier),
        "signal_stack_enabled": bool(signal_stack_enabled),
        "signal_stack_weights": {k: float(v) for k, v in signal_stack_weights.items()},
        "signal_stack_normalize_weights": bool(signal_stack_normalize_weights),
        "signal_model_component": float(signal_attribution["signal_model_component"]),
        "signal_momentum_component": float(signal_attribution["signal_momentum_component"]),
        "signal_reversal_component": float(signal_attribution["signal_reversal_component"]),
        "signal_vol_breakout_component": float(signal_attribution["signal_vol_breakout_component"]),
        "signal_liquidity_component": float(signal_attribution["signal_liquidity_component"]),
        "signal_composite": float(signal_attribution["signal_composite"]),
    }
    return out, summary


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_snapshot_dir = (PROJECT_ROOT / args.baseline_snapshot_dir).resolve()

    if args.top_k <= 0:
        raise ValueError("`--top-k` must be a positive integer.")

    config_data_path = args.config_data.resolve()
    config_model_path = args.config_model.resolve()
    config_backtest_path = args.config_backtest.resolve()
    config_execution_path = args.config_execution.resolve()
    data_cfg = load_yaml(config_data_path)
    execution_cfg = load_yaml(config_execution_path)

    print("[1/5] Fetching data (paper workflow; no IBKR actions)")
    prices, raw_path = run_fetch_data(config_path=config_data_path)

    print("[2/5] Building clean panel")
    clean_df, panel_df, clean_path, panel_path = run_build_panel(config_path=config_data_path)

    print("[3/5] Training model (single training pass used for both strategy modes)")
    (
        predictions,
        training_log,
        importances,
        train_summary,
        predictions_path,
        training_log_path,
        importance_path,
        train_summary_path,
    ) = run_train(
        config_data_path=config_data_path,
        config_model_path=config_model_path,
        config_backtest_path=config_backtest_path,
    )

    base_backtest_cfg = load_yaml(config_backtest_path)
    mode_cfgs = {
        "long_only": _prepare_mode_config(base_cfg=base_backtest_cfg, mode="long_only"),
        "market_neutral": _prepare_mode_config(base_cfg=base_backtest_cfg, mode="market_neutral"),
    }

    mode_summaries: dict[str, dict[str, Any]] = {}
    mode_scores: dict[str, float] = {}
    mode_tmp_paths: dict[str, Path] = {}
    modes = ["long_only", "market_neutral"]

    print("[4/5] Running backtests by mode")
    try:
        for mode in modes:
            tmp_path = (PROJECT_ROOT / "configs" / f"config_backtest.run_all.{mode}.yaml").resolve()
            mode_tmp_paths[mode] = tmp_path
            _write_yaml(path=tmp_path, payload=mode_cfgs[mode])

            _, _, _, summary, *_ = run_backtest(
                config_data_path=config_data_path,
                config_backtest_path=tmp_path,
                config_execution_path=config_execution_path,
            )
            mode_summaries[mode] = summary
            mode_scores[mode] = _score_summary(summary=summary)
            print(
                f"  - {mode}: weekly_sharpe={summary.get('weekly_sharpe_ratio')} "
                f"max_dd={summary.get('max_drawdown')} score={mode_scores[mode]:.6f}"
            )

        recommended_mode = max(mode_scores.items(), key=lambda kv: kv[1])[0]

        # Ensure outputs/backtests corresponds to recommended strategy for downstream paper-rebalance use.
        if recommended_mode != modes[-1]:
            run_backtest(
                config_data_path=config_data_path,
                config_backtest_path=mode_tmp_paths[recommended_mode],
                config_execution_path=config_execution_path,
            )
    finally:
        for tmp_path in mode_tmp_paths.values():
            tmp_path.unlink(missing_ok=True)

    weights_path = (PROJECT_ROOT / "outputs" / "backtests" / "weights_history.parquet").resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights history at {weights_path}")

    weights_history = pd.read_parquet(weights_path)
    if weights_history.empty:
        raise ValueError("weights_history.parquet is empty.")

    weights_history["rebalance_date"] = pd.to_datetime(weights_history["rebalance_date"], utc=False).dt.tz_localize(None)
    latest_rebalance_date = pd.Timestamp(weights_history["rebalance_date"].max())
    latest_backtest_weights = (
        weights_history[weights_history["rebalance_date"] == latest_rebalance_date][["ticker", "weight"]]
        .copy()
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    latest_backtest_weights["ticker"] = latest_backtest_weights["ticker"].astype(str)
    latest_backtest_weights["weight"] = latest_backtest_weights["weight"].astype(float)
    recommended_backtest_weights_path = (
        output_dir / f"recommended_weights_backtest_{recommended_mode}_{latest_rebalance_date.date()}.csv"
    )
    latest_backtest_weights.to_csv(recommended_backtest_weights_path, index=False)

    print("[5/5] Building live recommendation snapshot")
    (
        live_predictions,
        live_importances,
        live_predict_summary,
        live_predictions_path,
        live_importance_path,
        live_predict_summary_path,
    ) = run_predict_live(
        config_data_path=config_data_path,
        config_model_path=config_model_path,
    )
    live_weights, live_weights_summary = _compute_live_weights(
        data_cfg=data_cfg,
        mode_backtest_cfg=mode_cfgs[recommended_mode],
        execution_cfg=execution_cfg,
        clean_prices=clean_df,
        training_log=training_log,
        live_predictions=live_predictions,
    )
    live_signal_date = pd.Timestamp(live_predictions["date"].max())
    recommended_weights_path = output_dir / f"recommended_weights_{recommended_mode}_{live_signal_date.date()}.csv"
    live_weights.to_csv(recommended_weights_path, index=False)

    top_k = int(args.top_k)
    top_longs = live_weights[live_weights["weight"] > 0].sort_values("weight", ascending=False).head(top_k)
    top_shorts = live_weights[live_weights["weight"] < 0].sort_values("weight", ascending=True).head(top_k)
    backtest_top_longs = (
        latest_backtest_weights[latest_backtest_weights["weight"] > 0].sort_values("weight", ascending=False).head(top_k)
    )
    backtest_top_shorts = (
        latest_backtest_weights[latest_backtest_weights["weight"] < 0].sort_values("weight", ascending=True).head(top_k)
    )

    recommendation = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "paper_only": True,
        "ibkr_used": False,
        "recommended_strategy": recommended_mode,
        "rebalance_date": str(latest_rebalance_date.date()),
        "live_signal_date": str(live_signal_date.date()),
        "strategy_scores": mode_scores,
        "strategy_summaries": mode_summaries,
        "train_summary": train_summary,
        "predict_live_summary": live_predict_summary,
        "live_weights_summary": live_weights_summary,
        "artifacts": {
            "raw_prices_path": str(raw_path),
            "clean_prices_path": str(clean_path),
            "panel_path": str(panel_path),
            "predictions_path": str(predictions_path),
            "predictions_live_path": str(live_predictions_path),
            "training_log_path": str(training_log_path),
            "importance_path": str(importance_path),
            "importance_live_path": str(live_importance_path),
            "train_summary_path": str(train_summary_path),
            "predict_live_summary_path": str(live_predict_summary_path),
            "weights_history_path": str(weights_path),
            "recommended_weights_csv": str(recommended_weights_path),
            "recommended_weights_backtest_csv": str(recommended_backtest_weights_path),
        },
        "weights_snapshot": {
            "n_positions": int(len(live_weights)),
            "gross_exposure": float(live_weights["weight"].abs().sum()),
            "net_exposure": float(live_weights["weight"].sum()),
            "top_longs": _serialize_weights_row(top_longs),
            "top_shorts": _serialize_weights_row(top_shorts),
        },
        "backtest_weights_snapshot": {
            "n_positions": int(len(latest_backtest_weights)),
            "gross_exposure": float(latest_backtest_weights["weight"].abs().sum()),
            "net_exposure": float(latest_backtest_weights["weight"].sum()),
            "top_longs": _serialize_weights_row(backtest_top_longs),
            "top_shorts": _serialize_weights_row(backtest_top_shorts),
        },
    }

    recommendation_path = output_dir / "recommendation.json"
    recommendation_path.write_text(json.dumps(recommendation, indent=2, sort_keys=True), encoding="utf-8")
    snapshot_path: Path | None = None
    if args.snapshot_baseline:
        snapshot_path = _write_baseline_snapshot_bundle(
            recommendation=recommendation,
            config_data_path=config_data_path,
            config_model_path=config_model_path,
            config_backtest_path=config_backtest_path,
            config_execution_path=config_execution_path,
            baseline_snapshot_dir=baseline_snapshot_dir,
        )

    print("[done] Recommendation ready")
    print(f"Data rows fetched: {len(prices):,}")
    print(f"Panel rows: {len(panel_df):,}")
    print(f"Predictions rows: {len(predictions):,}")
    print(f"Training rebalances: {training_log['rebalance_date'].nunique()}")
    print(f"Recommended strategy: {recommended_mode}")
    print(f"Latest backtest rebalance date: {latest_rebalance_date.date()}")
    print(f"Live signal date: {live_signal_date.date()}")
    stale_days = live_predict_summary.get("market_context_stale_business_days")
    fallback_applied = live_predict_summary.get("market_context_fallback_applied")
    if stale_days is not None:
        print(
            "Live market-context freshness: "
            f"stale_business_days={stale_days} "
            f"fallback_applied={bool(fallback_applied)}"
        )
    print(f"Recommended live weights file: {recommended_weights_path}")
    print(f"Backtest snapshot weights file: {recommended_backtest_weights_path}")
    print(f"Recommendation report: {recommendation_path}")
    if snapshot_path is not None:
        print(f"Baseline snapshot bundle: {snapshot_path}")
    if not top_longs.empty:
        print("Top longs:")
        for _, row in top_longs.iterrows():
            print(f"  - {row['ticker']}: {row['weight']:.4f}")
    if not top_shorts.empty:
        print("Top shorts:")
        for _, row in top_shorts.iterrows():
            print(f"  - {row['ticker']}: {row['weight']:.4f}")
    if importances is not None and not importances.empty:
        top_feats = (
            importances.groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
            .head(5)
        )
        print("Top model features:")
        for _, row in top_feats.iterrows():
            print(f"  - {row['feature']}: {row['importance']:.6f}")


if __name__ == "__main__":
    main()
