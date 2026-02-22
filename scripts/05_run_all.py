from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_backtest
from src.data import load_yaml, run_fetch_data
from src.features import run_build_panel
from src.model_xgb import run_train


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

    objective["allocation_method"] = "score_over_vol"
    beta_cfg = portfolio.setdefault("beta_neutralization", {})
    if not isinstance(beta_cfg, dict):
        raise ValueError("`portfolio.beta_neutralization` must be a mapping.")

    if mode == "long_only":
        portfolio["mode"] = "long_only"
        constraints["long_only"] = True
        constraints["fully_invested"] = True
        beta_cfg["enabled"] = False
        return cfg

    if mode == "market_neutral":
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


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.top_k <= 0:
        raise ValueError("`--top-k` must be a positive integer.")

    config_data_path = args.config_data.resolve()
    config_model_path = args.config_model.resolve()
    config_backtest_path = args.config_backtest.resolve()
    config_execution_path = args.config_execution.resolve()

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
    latest_weights = (
        weights_history[weights_history["rebalance_date"] == latest_rebalance_date][["ticker", "weight"]]
        .copy()
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    latest_weights["ticker"] = latest_weights["ticker"].astype(str)
    latest_weights["weight"] = latest_weights["weight"].astype(float)

    recommended_weights_path = output_dir / f"recommended_weights_{recommended_mode}_{latest_rebalance_date.date()}.csv"
    latest_weights.to_csv(recommended_weights_path, index=False)

    top_k = int(args.top_k)
    top_longs = latest_weights[latest_weights["weight"] > 0].sort_values("weight", ascending=False).head(top_k)
    top_shorts = latest_weights[latest_weights["weight"] < 0].sort_values("weight", ascending=True).head(top_k)

    recommendation = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "paper_only": True,
        "ibkr_used": False,
        "recommended_strategy": recommended_mode,
        "rebalance_date": str(latest_rebalance_date.date()),
        "strategy_scores": mode_scores,
        "strategy_summaries": mode_summaries,
        "train_summary": train_summary,
        "artifacts": {
            "raw_prices_path": str(raw_path),
            "clean_prices_path": str(clean_path),
            "panel_path": str(panel_path),
            "predictions_path": str(predictions_path),
            "training_log_path": str(training_log_path),
            "importance_path": str(importance_path),
            "train_summary_path": str(train_summary_path),
            "weights_history_path": str(weights_path),
            "recommended_weights_csv": str(recommended_weights_path),
        },
        "weights_snapshot": {
            "n_positions": int(len(latest_weights)),
            "gross_exposure": float(latest_weights["weight"].abs().sum()),
            "net_exposure": float(latest_weights["weight"].sum()),
            "top_longs": _serialize_weights_row(top_longs),
            "top_shorts": _serialize_weights_row(top_shorts),
        },
    }

    recommendation_path = output_dir / "recommendation.json"
    recommendation_path.write_text(json.dumps(recommendation, indent=2, sort_keys=True), encoding="utf-8")

    print("[5/5] Recommendation ready")
    print(f"Data rows fetched: {len(prices):,}")
    print(f"Panel rows: {len(panel_df):,}")
    print(f"Predictions rows: {len(predictions):,}")
    print(f"Training rebalances: {training_log['rebalance_date'].nunique()}")
    print(f"Recommended strategy: {recommended_mode}")
    print(f"Latest rebalance date: {latest_rebalance_date.date()}")
    print(f"Recommended weights file: {recommended_weights_path}")
    print(f"Recommendation report: {recommendation_path}")
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
