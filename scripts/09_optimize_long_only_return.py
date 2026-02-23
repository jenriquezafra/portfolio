from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any
import uuid

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
            "Optimize long-only return by sweeping rebalance cadence, allocation method, "
            "weight cap, and turnover cap; then generate before/after report + recommended configs."
        )
    )
    parser.add_argument("--config-model", type=Path, default=Path("configs/config_model.yaml"))
    parser.add_argument("--config-backtest", type=Path, default=Path("configs/config_backtest.yaml"))
    parser.add_argument("--config-execution", type=Path, default=Path("configs/config_execution.yaml"))
    parser.add_argument(
        "--baseline-config-backtest",
        type=Path,
        default=None,
        help="Optional backtest config used only for baseline. Defaults to --config-backtest.",
    )
    parser.add_argument(
        "--baseline-config-execution",
        type=Path,
        default=None,
        help="Optional execution config used only for baseline. Defaults to --config-execution.",
    )
    parser.add_argument(
        "--baseline-config-data",
        type=Path,
        default=Path("configs/config_data.core25.yaml"),
        help="Data config used for the baseline (before).",
    )
    parser.add_argument(
        "--candidate-config-data",
        type=str,
        default="configs/config_data.yaml,configs/config_data.universe_expanded.yaml",
        help="Comma-separated data config paths to evaluate as candidate universes.",
    )
    parser.add_argument(
        "--rebalance-days",
        type=str,
        default="10,15,20",
        help="Comma-separated rebalance cadence values (every_n_days).",
    )
    parser.add_argument(
        "--allocation-methods",
        type=str,
        default="score_over_vol,mean_variance",
        help="Comma-separated allocation methods for long-only evaluation.",
    )
    parser.add_argument(
        "--weight-max-values",
        type=str,
        default="0.10,0.15,0.20",
        help="Comma-separated long-only weight caps (`constraints.weight_max`).",
    )
    parser.add_argument(
        "--turnover-cap-values",
        type=str,
        default="0.25,0.35,0.45",
        help="Comma-separated turnover caps. Supports `none`.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="If set, fetches market data for each evaluated data config before build/train.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/return_boost"),
        help="Directory where optimization artifacts will be stored.",
    )
    return parser.parse_args()


def _parse_path_list(raw: str) -> list[Path]:
    paths: list[Path] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        paths.append(Path(token))
    if not paths:
        raise ValueError("At least one candidate data config path is required.")
    return paths


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("Rebalance days must be positive integers.")
        values.append(value)
    if not values:
        raise ValueError("At least one rebalance cadence is required.")
    return values


def _parse_str_list(raw: str) -> list[str]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one value.")
    return values


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


def _safe_name(value: str | float | int | None) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        cleaned = value.strip().replace("/", "_")
        return cleaned.replace(".", "p")
    return str(value).replace(".", "p")


def _requested_universe_size(data_cfg: dict[str, Any]) -> int | None:
    data_section = data_cfg.get("data", {})
    if not isinstance(data_section, dict):
        return None
    universe = data_section.get("universe")
    if isinstance(universe, list):
        return int(len(universe))
    return None


def _prepare_long_only_backtest_cfg(
    base_back_cfg: dict[str, Any],
    rebalance_every_n_days: int,
    allocation_method: str,
    weight_max: float,
) -> dict[str, Any]:
    cfg = deepcopy(base_back_cfg)
    back = cfg.setdefault("backtest", {})
    if not isinstance(back, dict):
        raise ValueError("`backtest` section must be a mapping.")
    portfolio = back.setdefault("portfolio", {})
    constraints = back.setdefault("constraints", {})
    objective = back.setdefault("objective", {})
    if not isinstance(portfolio, dict) or not isinstance(constraints, dict) or not isinstance(objective, dict):
        raise ValueError("`portfolio`, `constraints`, and `objective` must be mappings.")

    back["rebalance_frequency"] = "every_n_days"
    back["rebalance_every_n_days"] = int(rebalance_every_n_days)

    portfolio["mode"] = "long_only"
    beta_cfg = portfolio.setdefault("beta_neutralization", {})
    if not isinstance(beta_cfg, dict):
        raise ValueError("`portfolio.beta_neutralization` must be a mapping.")
    beta_cfg["enabled"] = False

    constraints["long_only"] = True
    constraints["fully_invested"] = True
    constraints["weight_max"] = float(weight_max)

    objective["allocation_method"] = str(allocation_method).lower()
    return cfg


def _prepare_execution_cfg(base_exec_cfg: dict[str, Any], turnover_cap: float | None) -> dict[str, Any]:
    cfg = deepcopy(base_exec_cfg)
    risk = cfg.setdefault("risk_controls", {})
    if not isinstance(risk, dict):
        raise ValueError("`risk_controls` section must be a mapping.")
    risk["max_turnover_per_rebalance"] = turnover_cap
    return cfg


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _score_candidate(row: dict[str, Any]) -> tuple[float, float, float]:
    ann_ret = float(row.get("annualized_return") or float("-inf"))
    sharpe = float(row.get("sharpe_ratio") or float("-inf"))
    max_dd = float(row.get("max_drawdown") or float("-inf"))
    return ann_ret, sharpe, max_dd


def _safe_delta(best: Any, baseline: Any) -> float | None:
    if best is None or baseline is None:
        return None
    return float(best) - float(baseline)


def _save_row_artifacts(
    run_dir: Path,
    data_cfg: dict[str, Any],
    back_cfg: dict[str, Any],
    exec_cfg: dict[str, Any],
    row: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(run_dir / "config_data.yaml", data_cfg)
    _write_yaml(run_dir / "config_backtest.yaml", back_cfg)
    _write_yaml(run_dir / "config_execution.yaml", exec_cfg)
    (run_dir / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")


def _extract_training_metrics(train_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "oos_cs_ic_mean": train_summary.get("oos_cs_ic_mean"),
        "oos_cs_ic_ir": train_summary.get("oos_cs_ic_ir"),
        "oos_top_bottom_mean": train_summary.get("oos_top_bottom_mean"),
        "n_rebalances_train": train_summary.get("n_rebalances"),
    }


def _build_markdown_report(report: dict[str, Any]) -> str:
    baseline = report.get("baseline")
    if not isinstance(baseline, dict):
        baseline = {}
    best = report.get("best_candidate")
    if not isinstance(best, dict):
        best = {}
    deltas = report.get("delta_vs_baseline")
    if not isinstance(deltas, dict):
        deltas = {}
    lines = [
        "# Return Boost Report",
        "",
        "## Baseline",
        f"- run_id: {baseline.get('run_id')}",
        f"- annualized_return: {baseline.get('annualized_return')}",
        f"- sharpe_ratio: {baseline.get('sharpe_ratio')}",
        f"- max_drawdown: {baseline.get('max_drawdown')}",
        "",
        "## Best Candidate",
        f"- run_id: {best.get('run_id')}",
        f"- data_config_source: {best.get('data_config_source')}",
        f"- rebalance_every_n_days: {best.get('rebalance_every_n_days')}",
        f"- allocation_method: {best.get('allocation_method')}",
        f"- weight_max: {best.get('weight_max')}",
        f"- turnover_cap: {best.get('turnover_cap')}",
        f"- annualized_return: {best.get('annualized_return')}",
        f"- sharpe_ratio: {best.get('sharpe_ratio')}",
        f"- max_drawdown: {best.get('max_drawdown')}",
        "",
        "## Delta vs Baseline",
        f"- annualized_return: {deltas.get('annualized_return')}",
        f"- sharpe_ratio: {deltas.get('sharpe_ratio')}",
        f"- max_drawdown: {deltas.get('max_drawdown')}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    candidate_data_paths = _parse_path_list(args.candidate_config_data)
    rebalance_days_values = _parse_int_list(args.rebalance_days)
    allocation_methods = [m.lower() for m in _parse_str_list(args.allocation_methods)]
    weight_max_values = _parse_float_list(args.weight_max_values)
    turnover_caps = _parse_optional_float_list(args.turnover_cap_values)

    unsupported_alloc = [m for m in allocation_methods if m not in {"score_over_vol", "mean_variance"}]
    if unsupported_alloc:
        raise ValueError(f"Unsupported allocation methods: {unsupported_alloc}")

    config_model_path = args.config_model.resolve()
    baseline_data_path = args.baseline_config_data.resolve()
    base_back_cfg_path = args.config_backtest.resolve()
    base_exec_cfg_path = args.config_execution.resolve()
    baseline_back_cfg_path = (
        args.baseline_config_backtest.resolve() if args.baseline_config_backtest is not None else base_back_cfg_path
    )
    baseline_exec_cfg_path = (
        args.baseline_config_execution.resolve() if args.baseline_config_execution is not None else base_exec_cfg_path
    )
    base_back_cfg = load_yaml(base_back_cfg_path)
    base_exec_cfg = load_yaml(base_exec_cfg_path)

    rows: list[dict[str, Any]] = []
    run_id = 0

    tmp_paths: list[Path] = []
    tmp_token = uuid.uuid4().hex[:8]

    def make_tmp_yaml(stem: str) -> Path:
        path = (PROJECT_ROOT / "configs" / f"tmp_return_boost_{tmp_token}_{stem}.yaml").resolve()
        tmp_paths.append(path)
        return path

    try:

        # Baseline run ("before"): use current baseline data config + backtest/execution config as-is.
        try:
            baseline_data_cfg = load_yaml(baseline_data_path)
            baseline_back_cfg = load_yaml(baseline_back_cfg_path)
            baseline_exec_cfg = load_yaml(baseline_exec_cfg_path)

            tmp_data = make_tmp_yaml("baseline_config_data")
            tmp_back = make_tmp_yaml("baseline_config_backtest")
            tmp_exec = make_tmp_yaml("baseline_config_execution")
            _write_yaml(tmp_data, baseline_data_cfg)
            _write_yaml(tmp_back, baseline_back_cfg)
            _write_yaml(tmp_exec, baseline_exec_cfg)

            if args.refresh_data:
                run_fetch_data(config_path=tmp_data)
            clean_df, panel_df, _, _ = run_build_panel(config_path=tmp_data)
            _, _, _, train_summary, *_ = run_train(
                config_data_path=tmp_data,
                config_model_path=config_model_path,
                config_backtest_path=tmp_back,
            )
            _, _, _, bt_summary, *_ = run_backtest(
                config_data_path=tmp_data,
                config_backtest_path=tmp_back,
                config_execution_path=tmp_exec,
            )

            run_id += 1
            back_section = baseline_back_cfg.get("backtest", {})
            portfolio = back_section.get("portfolio", {}) if isinstance(back_section, dict) else {}
            constraints = back_section.get("constraints", {}) if isinstance(back_section, dict) else {}
            objective = back_section.get("objective", {}) if isinstance(back_section, dict) else {}
            risk_controls = baseline_exec_cfg.get("risk_controls", {})
            if not isinstance(portfolio, dict):
                portfolio = {}
            if not isinstance(constraints, dict):
                constraints = {}
            if not isinstance(objective, dict):
                objective = {}
            if not isinstance(risk_controls, dict):
                risk_controls = {}

            row = {
                "run_id": run_id,
                "phase": "baseline",
                "status": "ok",
                "error": None,
                "data_config_source": str(baseline_data_path),
                "requested_universe_size": _requested_universe_size(baseline_data_cfg),
                "realized_clean_tickers": int(clean_df["ticker"].nunique()),
                "realized_panel_tickers": int(panel_df["ticker"].nunique()),
                "horizon_days": baseline_data_cfg.get("labels", {}).get("horizon_days"),
                "rebalance_every_n_days": back_section.get("rebalance_every_n_days")
                if isinstance(back_section, dict)
                else None,
                "allocation_method": objective.get("allocation_method"),
                "weight_max": constraints.get("weight_max"),
                "turnover_cap": risk_controls.get("max_turnover_per_rebalance"),
                **_extract_training_metrics(train_summary),
                "annualized_return": bt_summary.get("annualized_return"),
                "annualized_volatility": bt_summary.get("annualized_volatility"),
                "sharpe_ratio": bt_summary.get("sharpe_ratio"),
                "weekly_sharpe_ratio": bt_summary.get("weekly_sharpe_ratio"),
                "max_drawdown": bt_summary.get("max_drawdown"),
                "average_turnover": bt_summary.get("average_turnover"),
                "total_cost_bps_paid": bt_summary.get("total_cost_bps_paid"),
            }
            rows.append(row)
            _save_row_artifacts(
                run_dir=runs_dir / f"run_{run_id:03d}_baseline",
                data_cfg=baseline_data_cfg,
                back_cfg=baseline_back_cfg,
                exec_cfg=baseline_exec_cfg,
                row=row,
            )
            print(
                f"[baseline] ann_return={row['annualized_return']} "
                f"sharpe={row['sharpe_ratio']} max_dd={row['max_drawdown']}"
            )
        except Exception as exc:  # noqa: BLE001
            run_id += 1
            row = {
                "run_id": run_id,
                "phase": "baseline",
                "status": "error",
                "error": str(exc),
                "data_config_source": str(baseline_data_path),
                "requested_universe_size": None,
                "realized_clean_tickers": None,
                "realized_panel_tickers": None,
                "horizon_days": None,
                "rebalance_every_n_days": None,
                "allocation_method": None,
                "weight_max": None,
                "turnover_cap": None,
                "oos_cs_ic_mean": None,
                "oos_cs_ic_ir": None,
                "oos_top_bottom_mean": None,
                "n_rebalances_train": None,
                "annualized_return": None,
                "annualized_volatility": None,
                "sharpe_ratio": None,
                "weekly_sharpe_ratio": None,
                "max_drawdown": None,
                "average_turnover": None,
                "total_cost_bps_paid": None,
            }
            rows.append(row)
            (runs_dir / f"run_{run_id:03d}_baseline_error.json").write_text(
                json.dumps(row, indent=2),
                encoding="utf-8",
            )
            print(f"[baseline] error={exc}")

        # Candidate runs ("after"): long-only grid search.
        for data_cfg_path in candidate_data_paths:
            data_path = data_cfg_path.resolve()
            data_cfg = load_yaml(data_path)
            requested_universe_size = _requested_universe_size(data_cfg)

            for rebalance_days in rebalance_days_values:
                # Train once for (data_config, rebalance_days), then sweep backtest-only knobs.
                train_back_cfg = _prepare_long_only_backtest_cfg(
                    base_back_cfg=base_back_cfg,
                    rebalance_every_n_days=rebalance_days,
                    allocation_method="score_over_vol",
                    weight_max=max(weight_max_values),
                )

                group_key = f"{data_path.stem}_rb{rebalance_days}"
                tmp_data = make_tmp_yaml(f"{group_key}_data")
                tmp_train_back = make_tmp_yaml(f"{group_key}_train_back")
                _write_yaml(tmp_data, data_cfg)
                _write_yaml(tmp_train_back, train_back_cfg)

                try:
                    if args.refresh_data:
                        run_fetch_data(config_path=tmp_data)
                    clean_df, panel_df, _, _ = run_build_panel(config_path=tmp_data)
                    _, _, _, train_summary, *_ = run_train(
                        config_data_path=tmp_data,
                        config_model_path=config_model_path,
                        config_backtest_path=tmp_train_back,
                    )
                except Exception as exc:  # noqa: BLE001
                    run_id += 1
                    row = {
                        "run_id": run_id,
                        "phase": "candidate",
                        "status": "error",
                        "error": f"train_group_error: {exc}",
                        "data_config_source": str(data_path),
                        "requested_universe_size": requested_universe_size,
                        "realized_clean_tickers": None,
                        "realized_panel_tickers": None,
                        "horizon_days": data_cfg.get("labels", {}).get("horizon_days"),
                        "rebalance_every_n_days": rebalance_days,
                        "allocation_method": None,
                        "weight_max": None,
                        "turnover_cap": None,
                        "oos_cs_ic_mean": None,
                        "oos_cs_ic_ir": None,
                        "oos_top_bottom_mean": None,
                        "n_rebalances_train": None,
                        "annualized_return": None,
                        "annualized_volatility": None,
                        "sharpe_ratio": None,
                        "weekly_sharpe_ratio": None,
                        "max_drawdown": None,
                        "average_turnover": None,
                        "total_cost_bps_paid": None,
                    }
                    rows.append(row)
                    (runs_dir / f"run_{run_id:03d}_train_error.json").write_text(
                        json.dumps(row, indent=2),
                        encoding="utf-8",
                    )
                    print(f"[train-group] data={data_path.name} rb={rebalance_days} error={exc}")
                    continue

                realized_clean_tickers = int(clean_df["ticker"].nunique())
                realized_panel_tickers = int(panel_df["ticker"].nunique())

                for allocation_method in allocation_methods:
                    for weight_max in weight_max_values:
                        for turnover_cap in turnover_caps:
                            back_cfg = _prepare_long_only_backtest_cfg(
                                base_back_cfg=base_back_cfg,
                                rebalance_every_n_days=rebalance_days,
                                allocation_method=allocation_method,
                                weight_max=weight_max,
                            )
                            exec_cfg = _prepare_execution_cfg(
                                base_exec_cfg=base_exec_cfg,
                                turnover_cap=turnover_cap,
                            )

                            tmp_back = make_tmp_yaml("combo_back")
                            tmp_exec = make_tmp_yaml("combo_exec")
                            _write_yaml(tmp_back, back_cfg)
                            _write_yaml(tmp_exec, exec_cfg)

                            run_id += 1
                            row: dict[str, Any] = {
                                "run_id": run_id,
                                "phase": "candidate",
                                "status": "ok",
                                "error": None,
                                "data_config_source": str(data_path),
                                "requested_universe_size": requested_universe_size,
                                "realized_clean_tickers": realized_clean_tickers,
                                "realized_panel_tickers": realized_panel_tickers,
                                "horizon_days": data_cfg.get("labels", {}).get("horizon_days"),
                                "rebalance_every_n_days": rebalance_days,
                                "allocation_method": allocation_method,
                                "weight_max": weight_max,
                                "turnover_cap": turnover_cap,
                                **_extract_training_metrics(train_summary),
                            }

                            try:
                                _, _, _, bt_summary, *_ = run_backtest(
                                    config_data_path=tmp_data,
                                    config_backtest_path=tmp_back,
                                    config_execution_path=tmp_exec,
                                )
                                row.update(
                                    {
                                        "annualized_return": bt_summary.get("annualized_return"),
                                        "annualized_volatility": bt_summary.get("annualized_volatility"),
                                        "sharpe_ratio": bt_summary.get("sharpe_ratio"),
                                        "weekly_sharpe_ratio": bt_summary.get("weekly_sharpe_ratio"),
                                        "max_drawdown": bt_summary.get("max_drawdown"),
                                        "average_turnover": bt_summary.get("average_turnover"),
                                        "total_cost_bps_paid": bt_summary.get("total_cost_bps_paid"),
                                    }
                                )
                            except Exception as exc:  # noqa: BLE001
                                row.update(
                                    {
                                        "status": "error",
                                        "error": str(exc),
                                        "annualized_return": None,
                                        "annualized_volatility": None,
                                        "sharpe_ratio": None,
                                        "weekly_sharpe_ratio": None,
                                        "max_drawdown": None,
                                        "average_turnover": None,
                                        "total_cost_bps_paid": None,
                                    }
                                )

                            rows.append(row)
                            run_name = (
                                f"run_{run_id:03d}_"
                                f"{data_path.stem}_rb{rebalance_days}_"
                                f"{_safe_name(allocation_method)}_"
                                f"w{_safe_name(weight_max)}_tc{_safe_name(turnover_cap)}"
                            )
                            _save_row_artifacts(
                                run_dir=runs_dir / run_name,
                                data_cfg=data_cfg,
                                back_cfg=back_cfg,
                                exec_cfg=exec_cfg,
                                row=row,
                            )
                            print(
                                f"[{run_id:03d}] data={data_path.name} rb={rebalance_days} "
                                f"alloc={allocation_method} wmax={weight_max} tc={turnover_cap} "
                                f"status={row['status']} ann_return={row['annualized_return']}"
                            )
    finally:
        for path in tmp_paths:
            path.unlink(missing_ok=True)

    if not rows:
        raise RuntimeError("No optimization rows were generated.")

    comparison_json = output_dir / "comparison.json"
    comparison_csv = output_dir / "comparison.csv"
    comparison_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with comparison_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    baseline_ok = [row for row in rows if row.get("phase") == "baseline" and row.get("status") == "ok"]
    candidate_ok = [row for row in rows if row.get("phase") == "candidate" and row.get("status") == "ok"]

    report: dict[str, Any] = {
        "n_runs_total": len(rows),
        "n_runs_ok": int(sum(1 for row in rows if row.get("status") == "ok")),
        "n_runs_error": int(sum(1 for row in rows if row.get("status") != "ok")),
        "baseline": baseline_ok[0] if baseline_ok else None,
        "best_candidate": None,
        "delta_vs_baseline": None,
        "recommended_configs": None,
    }

    if candidate_ok:
        best = sorted(candidate_ok, key=_score_candidate, reverse=True)[0]
        report["best_candidate"] = best

        if baseline_ok:
            base = baseline_ok[0]
            report["delta_vs_baseline"] = {
                "annualized_return": _safe_delta(best.get("annualized_return"), base.get("annualized_return")),
                "sharpe_ratio": _safe_delta(best.get("sharpe_ratio"), base.get("sharpe_ratio")),
                "max_drawdown": _safe_delta(best.get("max_drawdown"), base.get("max_drawdown")),
                "average_turnover": _safe_delta(best.get("average_turnover"), base.get("average_turnover")),
            }

        run_name_prefix = f"run_{int(best['run_id']):03d}_"
        run_dir_matches = list(runs_dir.glob(f"{run_name_prefix}*"))
        if run_dir_matches:
            best_run_dir = run_dir_matches[0]
            rec_data = output_dir / "config_data.recommended.yaml"
            rec_back = output_dir / "config_backtest.recommended.yaml"
            rec_exec = output_dir / "config_execution.recommended.yaml"
            rec_data.write_text((best_run_dir / "config_data.yaml").read_text(encoding="utf-8"), encoding="utf-8")
            rec_back.write_text((best_run_dir / "config_backtest.yaml").read_text(encoding="utf-8"), encoding="utf-8")
            rec_exec.write_text((best_run_dir / "config_execution.yaml").read_text(encoding="utf-8"), encoding="utf-8")
            report["recommended_configs"] = {
                "config_data": str(rec_data),
                "config_backtest": str(rec_back),
                "config_execution": str(rec_exec),
            }

    report_path = output_dir / "report.json"
    report_md_path = output_dir / "report.md"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md_path.write_text(_build_markdown_report(report), encoding="utf-8")

    print(f"Saved comparison JSON: {comparison_json}")
    print(f"Saved comparison CSV: {comparison_csv}")
    print(f"Saved report JSON: {report_path}")
    print(f"Saved report Markdown: {report_md_path}")
    if report.get("best_candidate"):
        best = report["best_candidate"]
        print(
            "Best candidate: "
            f"run_id={best.get('run_id')} "
            f"ann_return={best.get('annualized_return')} "
            f"sharpe={best.get('sharpe_ratio')} "
            f"max_dd={best.get('max_drawdown')}"
        )


if __name__ == "__main__":
    main()
