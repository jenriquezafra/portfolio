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
from src.features import run_build_panel
from src.model_xgb import run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep target horizon and holding period with fair methodology.")
    parser.add_argument("--config-data", type=Path, default=Path("configs/config_data.yaml"))
    parser.add_argument("--config-model", type=Path, default=Path("configs/config_model.yaml"))
    parser.add_argument("--config-backtest", type=Path, default=Path("configs/config_backtest.yaml"))
    parser.add_argument("--config-execution", type=Path, default=Path("configs/config_execution.yaml"))
    parser.add_argument(
        "--horizons",
        type=str,
        default="5,10,15",
        help="Comma-separated target horizons in trading days.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/target_holding_sweep"),
        help="Directory where comparison table and per-horizon artifacts are saved.",
    )
    return parser.parse_args()


def _parse_horizons(raw: str) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("All horizons must be positive integers.")
        out.append(value)
    if not out:
        raise ValueError("At least one horizon is required.")
    return out


def _copy_outputs(dst: Path) -> None:
    (dst / "models").mkdir(parents=True, exist_ok=True)
    (dst / "backtests").mkdir(parents=True, exist_ok=True)
    for name in [
        "predictions_oos.parquet",
        "training_log.parquet",
        "feature_importance.parquet",
        "train_summary.json",
    ]:
        shutil.copy2(PROJECT_ROOT / "outputs" / "models" / name, dst / "models" / name)
    for name in [
        "daily_returns.parquet",
        "weights_history.parquet",
        "rebalance_log.parquet",
        "backtest_summary.json",
        "subperiod_report.json",
        "factor_exposure_report.json",
        "factor_diagnostics_report.json",
    ]:
        shutil.copy2(PROJECT_ROOT / "outputs" / "backtests" / name, dst / "backtests" / name)


def main() -> None:
    args = parse_args()
    horizons = _parse_horizons(args.horizons)
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_data = load_yaml(args.config_data.resolve())
    base_back = load_yaml(args.config_backtest.resolve())

    rows: list[dict[str, object]] = []
    for horizon in horizons:
        data_cfg = deepcopy(base_data)
        back_cfg = deepcopy(base_back)

        labels = data_cfg.setdefault("labels", {})
        if not isinstance(labels, dict):
            raise ValueError("`labels` section must be a mapping.")
        labels["horizon_days"] = int(horizon)
        labels["target_column"] = f"fwd_return_{horizon}d_resid"
        labels.setdefault("target_mode", "cross_sectional_demeaned")

        backtest = back_cfg.setdefault("backtest", {})
        if not isinstance(backtest, dict):
            raise ValueError("`backtest` section must be a mapping.")
        if str(backtest.get("rebalance_frequency", "monthly")).lower() == "every_n_days":
            backtest["rebalance_every_n_days"] = int(horizon)

        tmp_data = (PROJECT_ROOT / "configs" / f"config_data.sweep_h{horizon}.yaml").resolve()
        tmp_back = (PROJECT_ROOT / "configs" / f"config_backtest.sweep_h{horizon}.yaml").resolve()
        tmp_data.write_text(yaml.safe_dump(data_cfg, sort_keys=False), encoding="utf-8")
        tmp_back.write_text(yaml.safe_dump(back_cfg, sort_keys=False), encoding="utf-8")

        try:
            run_build_panel(config_path=tmp_data)
            _, _, _, train_summary, *_ = run_train(
                config_data_path=tmp_data,
                config_model_path=args.config_model.resolve(),
                config_backtest_path=tmp_back,
            )
            _, _, _, backtest_summary, *_ = run_backtest(
                config_data_path=tmp_data,
                config_backtest_path=tmp_back,
                config_execution_path=args.config_execution.resolve(),
            )

            horizon_out = out_dir / f"h{horizon}"
            _copy_outputs(horizon_out)

            row = {
                "horizon_days": horizon,
                "holding_days": backtest.get("rebalance_every_n_days"),
                "n_rebalances": train_summary.get("n_rebalances"),
                "target_column": labels["target_column"],
                "oos_cs_ic_mean": train_summary.get("oos_cs_ic_mean"),
                "oos_cs_ic_ir": train_summary.get("oos_cs_ic_ir"),
                "oos_top_bottom_mean": train_summary.get("oos_top_bottom_mean"),
                "annualized_return": backtest_summary.get("annualized_return"),
                "annualized_volatility": backtest_summary.get("annualized_volatility"),
                "sharpe_ratio": backtest_summary.get("sharpe_ratio"),
                "weekly_sharpe_ratio": backtest_summary.get("weekly_sharpe_ratio"),
                "max_drawdown": backtest_summary.get("max_drawdown"),
                "average_turnover": backtest_summary.get("average_turnover"),
                "total_cost_bps_paid": backtest_summary.get("total_cost_bps_paid"),
            }
            rows.append(row)
            print(
                f"[h={horizon}] IC={row['oos_cs_ic_mean']} ICIR={row['oos_cs_ic_ir']} "
                f"Sharpe={row['sharpe_ratio']} WeeklySharpe={row['weekly_sharpe_ratio']}"
            )
        finally:
            tmp_data.unlink(missing_ok=True)
            tmp_back.unlink(missing_ok=True)

    comparison_json = out_dir / "comparison.json"
    comparison_csv = out_dir / "comparison.csv"
    comparison_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with comparison_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved comparison JSON: {comparison_json}")
    print(f"Saved comparison CSV: {comparison_csv}")


if __name__ == "__main__":
    main()
