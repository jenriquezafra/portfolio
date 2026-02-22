from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_backtest
from src.data import load_yaml
from src.features import run_build_panel
from src.model_xgb import run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune market-neutral portfolio parameters with a fixed model.")
    parser.add_argument("--config-data", type=Path, default=Path("configs/config_data.yaml"))
    parser.add_argument("--config-model", type=Path, default=Path("configs/config_model.yaml"))
    parser.add_argument("--config-backtest", type=Path, default=Path("configs/config_backtest.yaml"))
    parser.add_argument("--config-execution", type=Path, default=Path("configs/config_execution.yaml"))
    parser.add_argument("--horizon", type=int, default=5, help="Target horizon used to build panel/train before tuning.")
    parser.add_argument(
        "--quantiles",
        type=str,
        default="0.15,0.20,0.30",
        help="Comma-separated quantiles for both long/short buckets.",
    )
    parser.add_argument(
        "--weight-max-values",
        type=str,
        default="0.10,0.15,0.20",
        help="Comma-separated weight cap values (`constraints.weight_max`).",
    )
    parser.add_argument(
        "--gross-target-values",
        type=str,
        default="1.0,1.5",
        help="Comma-separated gross exposure targets.",
    )
    parser.add_argument(
        "--turnover-cap-values",
        type=str,
        default="0.20,0.35,none",
        help="Comma-separated turnover caps. Use `none` for no cap.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/market_neutral_tuning"),
        help="Directory to save tuning table and per-run artifacts.",
    )
    return parser.parse_args()


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def _parse_optional_float_list(raw: str) -> list[float | None]:
    values: list[float | None] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in {"none", "null", "na"}:
            values.append(None)
        else:
            values.append(float(token))
    if not values:
        raise ValueError("Expected at least one turnover cap value.")
    return values


def _safe_name(value: float | None) -> str:
    if value is None:
        return "none"
    return str(value).replace(".", "p")


def _copy_run_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    src_backtests = (PROJECT_ROOT / "outputs" / "backtests").resolve()
    for filename in [
        "backtest_summary.json",
        "subperiod_report.json",
        "factor_exposure_report.json",
        "factor_diagnostics_report.json",
    ]:
        shutil.copy2(src_backtests / filename, run_dir / filename)


def _score_row(row: dict[str, Any]) -> tuple[float, float, float]:
    # Higher weekly_sharpe is better.
    weekly_sharpe = row.get("weekly_sharpe_ratio")
    score_1 = float(weekly_sharpe) if weekly_sharpe is not None else float("-inf")
    # Lower absolute ex-post factor beta sum is better.
    ex_post_l1 = row.get("ex_post_factor_beta_l1")
    score_2 = -float(ex_post_l1) if ex_post_l1 is not None else float("-inf")
    # Lower drawdown magnitude (less negative) is better.
    max_dd = row.get("max_drawdown")
    score_3 = float(max_dd) if max_dd is not None else float("-inf")
    return score_1, score_2, score_3


def main() -> None:
    args = parse_args()
    quantiles = _parse_float_list(args.quantiles)
    weight_max_values = _parse_float_list(args.weight_max_values)
    gross_target_values = _parse_float_list(args.gross_target_values)
    turnover_caps = _parse_optional_float_list(args.turnover_cap_values)

    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_data = load_yaml(args.config_data.resolve())
    base_back = load_yaml(args.config_backtest.resolve())
    base_exec = load_yaml(args.config_execution.resolve())

    # 1) Train once with selected horizon (fair for all portfolio parameter combinations).
    data_cfg = deepcopy(base_data)
    labels = data_cfg.setdefault("labels", {})
    if not isinstance(labels, dict):
        raise ValueError("`labels` section in config_data must be a mapping.")
    labels["horizon_days"] = int(args.horizon)
    labels["target_column"] = f"fwd_return_{args.horizon}d_resid"
    labels.setdefault("target_mode", "cross_sectional_demeaned")

    train_back_cfg = deepcopy(base_back)
    train_back = train_back_cfg.setdefault("backtest", {})
    if not isinstance(train_back, dict):
        raise ValueError("`backtest` section must be a mapping.")
    if str(train_back.get("rebalance_frequency", "monthly")).lower() == "every_n_days":
        train_back["rebalance_every_n_days"] = int(args.horizon)

    tmp_data = (PROJECT_ROOT / "configs" / f"config_data.mn_tune_h{args.horizon}.yaml").resolve()
    tmp_train_back = (PROJECT_ROOT / "configs" / f"config_backtest.mn_tune_train_h{args.horizon}.yaml").resolve()
    tmp_data.write_text(yaml.safe_dump(data_cfg, sort_keys=False), encoding="utf-8")
    tmp_train_back.write_text(yaml.safe_dump(train_back_cfg, sort_keys=False), encoding="utf-8")

    try:
        run_build_panel(config_path=tmp_data)
        _, _, _, train_summary, *_ = run_train(
            config_data_path=tmp_data,
            config_model_path=args.config_model.resolve(),
            config_backtest_path=tmp_train_back,
        )

        print(
            f"[train] horizon={args.horizon} "
            f"IC={train_summary.get('oos_cs_ic_mean')} ICIR={train_summary.get('oos_cs_ic_ir')}"
        )

        # 2) Grid-search market-neutral portfolio hyperparameters using fixed predictions.
        rows: list[dict[str, Any]] = []
        run_idx = 0
        for q in quantiles:
            for wmax in weight_max_values:
                for gross in gross_target_values:
                    for cap in turnover_caps:
                        run_idx += 1
                        back_cfg = deepcopy(train_back_cfg)
                        exec_cfg = deepcopy(base_exec)

                        backtest = back_cfg.setdefault("backtest", {})
                        if not isinstance(backtest, dict):
                            raise ValueError("`backtest` section must be a mapping.")
                        portfolio = backtest.setdefault("portfolio", {})
                        constraints = backtest.setdefault("constraints", {})
                        objective = backtest.setdefault("objective", {})
                        if (
                            not isinstance(portfolio, dict)
                            or not isinstance(constraints, dict)
                            or not isinstance(objective, dict)
                        ):
                            raise ValueError("Portfolio/constraints/objective sections must be mappings.")

                        portfolio["mode"] = "market_neutral"
                        portfolio["long_quantile"] = float(q)
                        portfolio["short_quantile"] = float(q)
                        portfolio["gross_exposure_target"] = float(gross)
                        portfolio.setdefault("vol_lookback_days", int(backtest.get("risk_lookback_days", 60)))

                        beta_cfg = portfolio.setdefault("beta_neutralization", {})
                        if not isinstance(beta_cfg, dict):
                            raise ValueError("`beta_neutralization` must be a mapping.")
                        beta_cfg["enabled"] = True

                        constraints["long_only"] = False
                        constraints["fully_invested"] = False
                        constraints["weight_max"] = float(wmax)
                        objective["allocation_method"] = "score_over_vol"

                        risk_controls = exec_cfg.setdefault("risk_controls", {})
                        if not isinstance(risk_controls, dict):
                            raise ValueError("`risk_controls` must be a mapping.")
                        risk_controls["max_turnover_per_rebalance"] = None if cap is None else float(cap)

                        tmp_back = (PROJECT_ROOT / "configs" / "config_backtest.mn_tune.tmp.yaml").resolve()
                        tmp_exec = (PROJECT_ROOT / "configs" / "config_execution.mn_tune.tmp.yaml").resolve()
                        tmp_back.write_text(yaml.safe_dump(back_cfg, sort_keys=False), encoding="utf-8")
                        tmp_exec.write_text(yaml.safe_dump(exec_cfg, sort_keys=False), encoding="utf-8")
                        try:
                            status = "ok"
                            error_message = None
                            try:
                                _, _, _, summary, *_ = run_backtest(
                                    config_data_path=tmp_data,
                                    config_backtest_path=tmp_back,
                                    config_execution_path=tmp_exec,
                                )
                                factor_report_path = (
                                    PROJECT_ROOT / "outputs" / "backtests" / "factor_exposure_report.json"
                                ).resolve()
                                factor_report = json.loads(factor_report_path.read_text(encoding="utf-8"))
                            except ValueError as exc:
                                status = "infeasible"
                                error_message = str(exc)
                                summary = {}
                                factor_report = {}
                        finally:
                            tmp_back.unlink(missing_ok=True)
                            tmp_exec.unlink(missing_ok=True)

                        ex_ante_map = summary.get("average_ex_ante_factor_exposure") or {}
                        ex_post_map = summary.get("ex_post_factor_betas") or {}
                        ex_ante_l1 = (
                            float(sum(abs(float(v)) for v in ex_ante_map.values()))
                            if isinstance(ex_ante_map, dict)
                            else None
                        )
                        ex_post_l1 = (
                            float(sum(abs(float(v)) for v in ex_post_map.values()))
                            if isinstance(ex_post_map, dict)
                            else None
                        )

                        row = {
                            "run_id": run_idx,
                            "horizon_days": int(args.horizon),
                            "long_short_quantile": float(q),
                            "weight_max": float(wmax),
                            "gross_exposure_target": float(gross),
                            "turnover_cap": cap,
                            "status": status,
                            "error": error_message,
                            "annualized_return": summary.get("annualized_return"),
                            "annualized_volatility": summary.get("annualized_volatility"),
                            "sharpe_ratio": summary.get("sharpe_ratio"),
                            "weekly_sharpe_ratio": summary.get("weekly_sharpe_ratio"),
                            "max_drawdown": summary.get("max_drawdown"),
                            "average_turnover": summary.get("average_turnover"),
                            "total_cost_bps_paid": summary.get("total_cost_bps_paid"),
                            "average_gross_exposure": summary.get("average_gross_exposure"),
                            "average_net_exposure": summary.get("average_net_exposure"),
                            "ex_ante_factor_exposure_l1": ex_ante_l1,
                            "ex_post_factor_beta_l1": ex_post_l1,
                            "ex_post_factor_r2": summary.get("ex_post_factor_r2"),
                        }
                        rows.append(row)

                        run_name = (
                            f"q{_safe_name(q)}_w{_safe_name(wmax)}_g{_safe_name(gross)}_tc{_safe_name(cap)}"
                        )
                        run_dir = runs_dir / run_name
                        if status == "ok":
                            _copy_run_artifacts(run_dir=run_dir)
                            (run_dir / "factor_exposure_report.json").write_text(
                                json.dumps(factor_report, indent=2),
                                encoding="utf-8",
                            )
                        else:
                            run_dir.mkdir(parents=True, exist_ok=True)

                        (run_dir / "config_backtest.yaml").write_text(
                            yaml.safe_dump(back_cfg, sort_keys=False),
                            encoding="utf-8",
                        )
                        (run_dir / "config_execution.yaml").write_text(
                            yaml.safe_dump(exec_cfg, sort_keys=False),
                            encoding="utf-8",
                        )
                        (run_dir / "result.json").write_text(
                            json.dumps(row, indent=2),
                            encoding="utf-8",
                        )

                        if status == "ok":
                            print(
                                f"[{run_idx:02d}] q={q:.2f} wmax={wmax:.2f} gross={gross:.2f} cap={cap} "
                                f"weekly_sharpe={row['weekly_sharpe_ratio']}"
                            )
                        else:
                            print(
                                f"[{run_idx:02d}] q={q:.2f} wmax={wmax:.2f} gross={gross:.2f} cap={cap} "
                                f"status={status} error={error_message}"
                            )

        ok_rows = [row for row in rows if row.get("status") == "ok"]
        bad_rows = [row for row in rows if row.get("status") != "ok"]
        rows_sorted_ok = sorted(ok_rows, key=_score_row, reverse=True)
        rows_sorted = rows_sorted_ok + bad_rows
        comparison_json = out_dir / "comparison.json"
        comparison_csv = out_dir / "comparison.csv"
        top10_json = out_dir / "top10.json"
        comparison_json.write_text(json.dumps(rows_sorted, indent=2), encoding="utf-8")
        if rows_sorted:
            with comparison_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows_sorted[0].keys()))
                writer.writeheader()
                writer.writerows(rows_sorted)
        top10_json.write_text(json.dumps(rows_sorted_ok[:10], indent=2), encoding="utf-8")

        print(f"Saved comparison JSON: {comparison_json}")
        print(f"Saved comparison CSV: {comparison_csv}")
        print(f"Saved top-10 JSON: {top10_json}")
    finally:
        tmp_data.unlink(missing_ok=True)
        tmp_train_back.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
