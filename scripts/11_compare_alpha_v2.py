from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_backtest
from src.data import run_fetch_data
from src.features import run_build_panel
from src.model_xgb import run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline model vs Alpha v2 package.")
    parser.add_argument("--config-data", type=Path, default=Path("configs/config_data.yaml"))
    parser.add_argument("--config-execution", type=Path, default=Path("configs/config_execution.yaml"))
    parser.add_argument("--baseline-model", type=Path, default=Path("configs/config_model.baseline_v1.yaml"))
    parser.add_argument("--baseline-backtest", type=Path, default=Path("configs/config_backtest.baseline_v1.yaml"))
    parser.add_argument("--alpha-model", type=Path, default=Path("configs/config_model.alpha_v2.yaml"))
    parser.add_argument("--alpha-backtest", type=Path, default=Path("configs/config_backtest.alpha_v2.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/alpha_v2_compare"),
        help="Directory where comparison artifacts will be saved.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Fetch data before running comparison.",
    )
    return parser.parse_args()


def _copy_artifacts(dst: Path) -> None:
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


def _run_case(
    name: str,
    config_data: Path,
    config_model: Path,
    config_backtest: Path,
    config_execution: Path,
    out_dir: Path,
    refresh_data: bool,
) -> dict[str, Any]:
    if refresh_data:
        run_fetch_data(config_path=config_data)

    clean_df, panel_df, _, _ = run_build_panel(config_path=config_data)
    _, _, _, train_summary, *_ = run_train(
        config_data_path=config_data,
        config_model_path=config_model,
        config_backtest_path=config_backtest,
    )
    _, _, _, backtest_summary, *_ = run_backtest(
        config_data_path=config_data,
        config_backtest_path=config_backtest,
        config_execution_path=config_execution,
    )

    scenario_dir = out_dir / name
    _copy_artifacts(scenario_dir)

    result = {
        "scenario": name,
        "requested_universe_size": int(clean_df["ticker"].nunique()),
        "panel_rows": int(len(panel_df)),
        "n_rebalances_train": train_summary.get("n_rebalances"),
        "training_target_transform": train_summary.get("training_target_transform"),
        "oos_cs_ic_mean": train_summary.get("oos_cs_ic_mean"),
        "oos_cs_ic_ir": train_summary.get("oos_cs_ic_ir"),
        "oos_top_bottom_mean": train_summary.get("oos_top_bottom_mean"),
        "oos_top_bottom_tstat": train_summary.get("oos_top_bottom_tstat"),
        "annualized_return": backtest_summary.get("annualized_return"),
        "annualized_volatility": backtest_summary.get("annualized_volatility"),
        "sharpe_ratio": backtest_summary.get("sharpe_ratio"),
        "weekly_sharpe_ratio": backtest_summary.get("weekly_sharpe_ratio"),
        "max_drawdown": backtest_summary.get("max_drawdown"),
        "average_turnover": backtest_summary.get("average_turnover"),
        "total_cost_bps_paid": backtest_summary.get("total_cost_bps_paid"),
        "signal_quality_gate_enabled": backtest_summary.get("signal_quality_gate_enabled"),
        "average_signal_gate_multiplier": backtest_summary.get("average_signal_gate_multiplier"),
        "signal_gate_active_rate": backtest_summary.get("signal_gate_active_rate"),
    }
    (scenario_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _delta(alpha: dict[str, Any], baseline: dict[str, Any], key: str) -> float | None:
    a = alpha.get(key)
    b = baseline.get(key)
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    baseline = report["baseline"]
    alpha = report["alpha_v2"]
    delta = report["delta"]
    lines = [
        "# Alpha V2 Comparison",
        "",
        "## Baseline",
        f"- annualized_return: {baseline.get('annualized_return')}",
        f"- sharpe_ratio: {baseline.get('sharpe_ratio')}",
        f"- max_drawdown: {baseline.get('max_drawdown')}",
        f"- oos_cs_ic_mean: {baseline.get('oos_cs_ic_mean')}",
        "",
        "## Alpha V2",
        f"- annualized_return: {alpha.get('annualized_return')}",
        f"- sharpe_ratio: {alpha.get('sharpe_ratio')}",
        f"- max_drawdown: {alpha.get('max_drawdown')}",
        f"- oos_cs_ic_mean: {alpha.get('oos_cs_ic_mean')}",
        "",
        "## Delta (Alpha V2 - Baseline)",
        f"- annualized_return: {delta.get('annualized_return')}",
        f"- sharpe_ratio: {delta.get('sharpe_ratio')}",
        f"- max_drawdown: {delta.get('max_drawdown')}",
        f"- oos_cs_ic_mean: {delta.get('oos_cs_ic_mean')}",
        f"- oos_top_bottom_mean: {delta.get('oos_top_bottom_mean')}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config_data = args.config_data.resolve()
    config_execution = args.config_execution.resolve()
    baseline_model = args.baseline_model.resolve()
    baseline_backtest = args.baseline_backtest.resolve()
    alpha_model = args.alpha_model.resolve()
    alpha_backtest = args.alpha_backtest.resolve()

    baseline = _run_case(
        name="baseline_v1",
        config_data=config_data,
        config_model=baseline_model,
        config_backtest=baseline_backtest,
        config_execution=config_execution,
        out_dir=out_dir,
        refresh_data=args.refresh_data,
    )
    alpha = _run_case(
        name="alpha_v2",
        config_data=config_data,
        config_model=alpha_model,
        config_backtest=alpha_backtest,
        config_execution=config_execution,
        out_dir=out_dir,
        refresh_data=False,
    )

    delta = {
        "annualized_return": _delta(alpha, baseline, "annualized_return"),
        "sharpe_ratio": _delta(alpha, baseline, "sharpe_ratio"),
        "max_drawdown": _delta(alpha, baseline, "max_drawdown"),
        "oos_cs_ic_mean": _delta(alpha, baseline, "oos_cs_ic_mean"),
        "oos_top_bottom_mean": _delta(alpha, baseline, "oos_top_bottom_mean"),
    }

    report = {"baseline": baseline, "alpha_v2": alpha, "delta": delta}
    report_json = out_dir / "comparison_report.json"
    report_md = out_dir / "comparison_report.md"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, report_md)

    print(f"Saved comparison report JSON: {report_json}")
    print(f"Saved comparison report Markdown: {report_md}")
    print(
        "Delta summary: "
        f"ann_return={delta['annualized_return']} "
        f"sharpe={delta['sharpe_ratio']} "
        f"max_dd={delta['max_drawdown']} "
        f"ic_mean={delta['oos_cs_ic_mean']}"
    )


if __name__ == "__main__":
    main()
