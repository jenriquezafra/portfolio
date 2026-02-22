from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_backtest
from src.data import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare long-only vs market-neutral backtests on the same predictions.")
    parser.add_argument("--config-data", type=Path, default=Path("configs/config_data.yaml"))
    parser.add_argument("--config-backtest", type=Path, default=Path("configs/config_backtest.yaml"))
    parser.add_argument("--config-execution", type=Path, default=Path("configs/config_execution.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/portfolio_mode_compare"),
        help="Directory to save per-mode artifacts and comparison table.",
    )
    return parser.parse_args()


def _prepare_mode_config(base_cfg: dict[str, object], mode: str) -> dict[str, object]:
    cfg = deepcopy(base_cfg)
    back = cfg.setdefault("backtest", {})
    if not isinstance(back, dict):
        raise ValueError("`backtest` section must be a mapping.")

    portfolio = back.setdefault("portfolio", {})
    constraints = back.setdefault("constraints", {})
    objective = back.setdefault("objective", {})
    if not isinstance(portfolio, dict) or not isinstance(constraints, dict) or not isinstance(objective, dict):
        raise ValueError("Portfolio/constraints/objective sections must be mappings.")

    objective["allocation_method"] = "score_over_vol"
    portfolio["mode"] = mode
    portfolio.setdefault("gross_exposure_target", 1.0)
    portfolio.setdefault("long_quantile", 0.20)
    portfolio.setdefault("short_quantile", 0.20)
    portfolio.setdefault("vol_lookback_days", int(back.get("risk_lookback_days", 60)))

    beta_cfg = portfolio.setdefault("beta_neutralization", {})
    if not isinstance(beta_cfg, dict):
        raise ValueError("`beta_neutralization` must be a mapping.")

    if mode == "long_only":
        constraints["long_only"] = True
        constraints["fully_invested"] = True
        beta_cfg["enabled"] = False
    elif mode == "market_neutral":
        constraints["long_only"] = False
        constraints["fully_invested"] = False
        beta_cfg["enabled"] = True
        beta_cfg.setdefault("frequency", "weekly")
        beta_cfg.setdefault("lookback_days", 126)
        beta_cfg.setdefault("factor_source", "universe_equal_weight")
        beta_cfg.setdefault("target_beta", 0.0)
    else:
        raise ValueError("Unsupported mode.")

    return cfg


def _copy_backtest_outputs(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    src_dir = (PROJECT_ROOT / "outputs/backtests").resolve()
    for name in [
        "daily_returns.parquet",
        "weights_history.parquet",
        "rebalance_log.parquet",
        "backtest_summary.json",
        "subperiod_report.json",
        "factor_exposure_report.json",
        "factor_diagnostics_report.json",
    ]:
        shutil.copy2(src_dir / name, destination / name)


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_backtest_cfg = load_yaml(args.config_backtest.resolve())
    modes = ["long_only", "market_neutral"]
    rows: list[dict[str, object]] = []

    for mode in modes:
        mode_cfg = _prepare_mode_config(base_backtest_cfg, mode=mode)
        tmp_cfg = (PROJECT_ROOT / "configs" / f"config_backtest.mode_{mode}.yaml").resolve()
        tmp_cfg.write_text(yaml.safe_dump(mode_cfg, sort_keys=False), encoding="utf-8")
        try:
            _, _, _, summary, _, _, _, _ = run_backtest(
                config_data_path=args.config_data.resolve(),
                config_backtest_path=tmp_cfg,
                config_execution_path=args.config_execution.resolve(),
            )
            mode_out = output_dir / mode
            _copy_backtest_outputs(destination=mode_out)

            row = {
                "mode": mode,
                "annualized_return": summary.get("annualized_return"),
                "annualized_volatility": summary.get("annualized_volatility"),
                "sharpe_ratio": summary.get("sharpe_ratio"),
                "weekly_sharpe_ratio": summary.get("weekly_sharpe_ratio"),
                "max_drawdown": summary.get("max_drawdown"),
                "average_turnover": summary.get("average_turnover"),
                "average_gross_exposure": summary.get("average_gross_exposure"),
                "average_net_exposure": summary.get("average_net_exposure"),
                "average_ex_ante_beta_to_factor": summary.get("average_ex_ante_beta_to_factor"),
                "average_ex_ante_factor_exposure": summary.get("average_ex_ante_factor_exposure"),
                "ex_post_factor_betas": summary.get("ex_post_factor_betas"),
                "ex_post_factor_r2": summary.get("ex_post_factor_r2"),
            }
            rows.append(row)
            print(f"[{mode}] Sharpe={row['sharpe_ratio']} WeeklySharpe={row['weekly_sharpe_ratio']}")
        finally:
            tmp_cfg.unlink(missing_ok=True)

    comparison_json = output_dir / "comparison.json"
    comparison_csv = output_dir / "comparison.csv"
    comparison_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with comparison_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved comparison JSON: {comparison_json}")
    print(f"Saved comparison CSV: {comparison_csv}")


if __name__ == "__main__":
    main()
