from __future__ import annotations

import argparse
import json
import sys
import uuid
from copy import deepcopy
from pathlib import Path
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
        description="Tune Alpha v2 for balanced metrics (return, Sharpe, drawdown, signal quality)."
    )
    parser.add_argument("--config-data", type=Path, default=Path("configs/config_data.yaml"))
    parser.add_argument("--baseline-model", type=Path, default=Path("configs/config_model.alpha_v2.yaml"))
    parser.add_argument("--baseline-backtest", type=Path, default=Path("configs/config_backtest.alpha_v2.yaml"))
    parser.add_argument("--baseline-execution", type=Path, default=Path("configs/config_execution.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/alpha_v2_tuning"),
        help="Directory to store sweep tables and recommended configs.",
    )
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument(
        "--apply-defaults",
        action="store_true",
        help="If set, overwrite configs/config_model.yaml, config_backtest.yaml and config_execution.yaml with best.",
    )
    return parser.parse_args()


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _safe_metric(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _composite_score(train_summary: dict[str, Any], back_summary: dict[str, Any]) -> float:
    ann_return = _safe_metric(back_summary.get("annualized_return"), default=-1.0)
    sharpe = _safe_metric(back_summary.get("sharpe_ratio"), default=-1.0)
    weekly_sharpe = _safe_metric(back_summary.get("weekly_sharpe_ratio"), default=-1.0)
    ann_vol = _safe_metric(back_summary.get("annualized_volatility"), default=1.0)
    drawdown = abs(_safe_metric(back_summary.get("max_drawdown"), default=-1.0))
    turnover = _safe_metric(back_summary.get("average_turnover"), default=1.0)
    ic_mean = _safe_metric(train_summary.get("oos_cs_ic_mean"), default=0.0)
    top_bottom = _safe_metric(train_summary.get("oos_top_bottom_mean"), default=0.0)

    score = (
        1.5 * sharpe
        + 0.7 * weekly_sharpe
        + 0.8 * ann_return
        - 0.9 * drawdown
        - 0.15 * ann_vol
        - 0.15 * turnover
        + 8.0 * ic_mean
        + 12.0 * top_bottom
    )
    if drawdown > 0.50:
        score -= 3.0 * (drawdown - 0.50)
    if ic_mean < 0:
        score -= 0.5
    return float(score)


def _copy_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(cfg)


def _tmp_path(token: str, stem: str) -> Path:
    return (PROJECT_ROOT / "configs" / f"tmp_alpha_tune_{token}_{stem}.yaml").resolve()


def main() -> None:
    args = parse_args()
    config_data_path = args.config_data.resolve()
    baseline_model_path = args.baseline_model.resolve()
    baseline_backtest_path = args.baseline_backtest.resolve()
    baseline_execution_path = args.baseline_execution.resolve()

    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    token = uuid.uuid4().hex[:10]
    created_tmp: list[Path] = []

    def make_tmp(stem: str, payload: dict[str, Any]) -> Path:
        path = _tmp_path(token, stem)
        _write_yaml(path, payload)
        created_tmp.append(path)
        return path

    try:
        if args.refresh_data:
            run_fetch_data(config_path=config_data_path)

        run_build_panel(config_path=config_data_path)

        base_model_cfg = load_yaml(baseline_model_path)
        base_back_cfg = load_yaml(baseline_backtest_path)
        base_exec_cfg = load_yaml(baseline_execution_path)

        model_presets: list[dict[str, Any]] = [
            {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.75, "colsample_bytree": 0.75},
            {"n_estimators": 700, "max_depth": 4, "learning_rate": 0.02, "subsample": 0.75, "colsample_bytree": 0.75},
            {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.80, "colsample_bytree": 0.80},
            {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.02, "subsample": 0.85, "colsample_bytree": 0.85},
            {"n_estimators": 600, "max_depth": 4, "learning_rate": 0.025, "subsample": 0.70, "colsample_bytree": 0.70},
            {"n_estimators": 350, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.75, "colsample_bytree": 0.75},
        ]

        model_rows: list[dict[str, Any]] = []
        for i, preset in enumerate(model_presets, start=1):
            model_cfg = _copy_cfg(base_model_cfg)
            model_section = model_cfg.setdefault("model", {})
            if not isinstance(model_section, dict):
                raise ValueError("`model` section must be a mapping.")
            params = model_section.setdefault("params", {})
            if not isinstance(params, dict):
                raise ValueError("`model.params` section must be a mapping.")
            model_section["training_target_transform"] = "cross_sectional_rank"
            params.update(preset)

            tmp_model = make_tmp(f"model_{i}", model_cfg)
            _, _, _, train_summary, *_ = run_train(
                config_data_path=config_data_path,
                config_model_path=tmp_model,
                config_backtest_path=baseline_backtest_path,
            )
            _, _, _, back_summary, *_ = run_backtest(
                config_data_path=config_data_path,
                config_backtest_path=baseline_backtest_path,
                config_execution_path=baseline_execution_path,
            )
            row = {
                "stage": "model",
                "run_id": i,
                "preset": preset,
                "train_summary": train_summary,
                "backtest_summary": back_summary,
                "score": _composite_score(train_summary=train_summary, back_summary=back_summary),
            }
            model_rows.append(row)
            print(
                f"[model {i:02d}] score={row['score']:.6f} "
                f"ann={back_summary.get('annualized_return')} sharpe={back_summary.get('sharpe_ratio')} "
                f"ic={train_summary.get('oos_cs_ic_mean')}"
            )

        model_rows_sorted = sorted(model_rows, key=lambda r: float(r["score"]), reverse=True)
        best_model_row = model_rows_sorted[0]
        best_model_cfg = _copy_cfg(base_model_cfg)
        best_model_params = best_model_row["preset"]
        best_model_section = best_model_cfg.setdefault("model", {})
        best_model_section["training_target_transform"] = "cross_sectional_rank"
        if not isinstance(best_model_section, dict):
            raise ValueError("`model` section must be a mapping.")
        best_params = best_model_section.setdefault("params", {})
        if not isinstance(best_params, dict):
            raise ValueError("`model.params` section must be a mapping.")
        best_params.update(best_model_params)
        best_model_path = make_tmp("best_model", best_model_cfg)

        structural_rows: list[dict[str, Any]] = []
        run_idx = 0
        for rebalance_days in [10, 15, 20]:
            # Re-train for each rebalance cadence so predictions are aligned with that cadence.
            train_back_cfg = _copy_cfg(base_back_cfg)
            train_back = train_back_cfg.setdefault("backtest", {})
            if not isinstance(train_back, dict):
                raise ValueError("`backtest` section must be mapping.")
            train_back["rebalance_frequency"] = "every_n_days"
            train_back["rebalance_every_n_days"] = int(rebalance_days)
            tmp_train_back = make_tmp(f"struct_train_back_{rebalance_days}", train_back_cfg)
            _, _, _, train_summary_for_day, *_ = run_train(
                config_data_path=config_data_path,
                config_model_path=best_model_path,
                config_backtest_path=tmp_train_back,
            )

            for weight_max in [0.12, 0.15, 0.20]:
                for turnover_cap in [0.20, 0.25, 0.30, 0.35]:
                    for risk_lambda in [5.0, 10.0, 15.0]:
                        for overlay_enabled in [False, True]:
                            run_idx += 1
                            back_cfg = _copy_cfg(base_back_cfg)
                            exec_cfg = _copy_cfg(base_exec_cfg)

                            back = back_cfg.setdefault("backtest", {})
                            if not isinstance(back, dict):
                                raise ValueError("`backtest` section must be mapping.")
                            back["rebalance_frequency"] = "every_n_days"
                            back["rebalance_every_n_days"] = int(rebalance_days)

                            constraints = back.setdefault("constraints", {})
                            objective = back.setdefault("objective", {})
                            overlay_cfg = back.setdefault("risk_overlay", {})
                            if not isinstance(constraints, dict) or not isinstance(objective, dict) or not isinstance(
                                overlay_cfg, dict
                            ):
                                raise ValueError("`constraints`, `objective`, `risk_overlay` must be mappings.")
                            constraints["weight_max"] = float(weight_max)
                            objective["risk_aversion_lambda"] = float(risk_lambda)
                            overlay_cfg["enabled"] = bool(overlay_enabled)
                            overlay_cfg["vol_target_annual"] = 0.18
                            overlay_cfg["min_leverage"] = 0.35
                            overlay_cfg["max_leverage"] = 1.0

                            risk_controls = exec_cfg.setdefault("risk_controls", {})
                            if not isinstance(risk_controls, dict):
                                raise ValueError("`risk_controls` must be mapping.")
                            risk_controls["max_turnover_per_rebalance"] = float(turnover_cap)

                            tmp_back = make_tmp(f"struct_back_{run_idx}", back_cfg)
                            tmp_exec = make_tmp(f"struct_exec_{run_idx}", exec_cfg)
                            _, _, _, back_summary, *_ = run_backtest(
                                config_data_path=config_data_path,
                                config_backtest_path=tmp_back,
                                config_execution_path=tmp_exec,
                            )

                            row = {
                                "stage": "structural",
                                "run_id": run_idx,
                                "rebalance_days": rebalance_days,
                                "weight_max": weight_max,
                                "turnover_cap": turnover_cap,
                                "risk_aversion_lambda": risk_lambda,
                                "overlay_enabled": overlay_enabled,
                                "train_summary": train_summary_for_day,
                                "backtest_summary": back_summary,
                                "score": _composite_score(
                                    train_summary=train_summary_for_day,
                                    back_summary=back_summary,
                                ),
                            }
                            structural_rows.append(row)
                            if run_idx % 20 == 0:
                                print(
                                    f"[struct {run_idx:03d}] score={row['score']:.6f} "
                                    f"ann={back_summary.get('annualized_return')} "
                                    f"sharpe={back_summary.get('sharpe_ratio')} "
                                    f"dd={back_summary.get('max_drawdown')}"
                                )

        structural_rows_sorted = sorted(structural_rows, key=lambda r: float(r["score"]), reverse=True)
        best_struct = structural_rows_sorted[0]

        # Re-train at the selected structural cadence before gate sweep.
        gate_train_back_cfg = _copy_cfg(base_back_cfg)
        gate_train_back = gate_train_back_cfg.setdefault("backtest", {})
        if not isinstance(gate_train_back, dict):
            raise ValueError("`backtest` section must be mapping.")
        gate_train_back["rebalance_frequency"] = "every_n_days"
        gate_train_back["rebalance_every_n_days"] = int(best_struct["rebalance_days"])
        tmp_gate_train_back = make_tmp("gate_train_back", gate_train_back_cfg)
        _, _, _, gate_train_summary, *_ = run_train(
            config_data_path=config_data_path,
            config_model_path=best_model_path,
            config_backtest_path=tmp_gate_train_back,
        )

        gate_rows: list[dict[str, Any]] = []
        gate_idx = 0
        for threshold in [0.0, 0.005, 0.01, 0.015, 0.02]:
            for bad_mult in [0.20, 0.30, 0.35, 0.45, 0.60]:
                for lookback in [12, 20]:
                    for min_history in [6, 8]:
                        gate_idx += 1
                        back_cfg = _copy_cfg(base_back_cfg)
                        exec_cfg = _copy_cfg(base_exec_cfg)

                        back = back_cfg.setdefault("backtest", {})
                        constraints = back.setdefault("constraints", {})
                        objective = back.setdefault("objective", {})
                        overlay_cfg = back.setdefault("risk_overlay", {})
                        gate_cfg = back.setdefault("signal_quality_gate", {})
                        if (
                            not isinstance(back, dict)
                            or not isinstance(constraints, dict)
                            or not isinstance(objective, dict)
                            or not isinstance(overlay_cfg, dict)
                            or not isinstance(gate_cfg, dict)
                        ):
                            raise ValueError("Backtest blocks must be mappings.")

                        back["rebalance_frequency"] = "every_n_days"
                        back["rebalance_every_n_days"] = int(best_struct["rebalance_days"])
                        constraints["weight_max"] = float(best_struct["weight_max"])
                        objective["risk_aversion_lambda"] = float(best_struct["risk_aversion_lambda"])
                        overlay_cfg["enabled"] = bool(best_struct["overlay_enabled"])
                        overlay_cfg["vol_target_annual"] = 0.18
                        overlay_cfg["min_leverage"] = 0.35
                        overlay_cfg["max_leverage"] = 1.0

                        gate_cfg["enabled"] = True
                        gate_cfg["metric"] = "oos_cs_ic_spearman"
                        gate_cfg["lookback_rebalances"] = int(lookback)
                        gate_cfg["min_history_rebalances"] = int(min_history)
                        gate_cfg["threshold"] = float(threshold)
                        gate_cfg["bad_state_multiplier"] = float(bad_mult)

                        risk_controls = exec_cfg.setdefault("risk_controls", {})
                        if not isinstance(risk_controls, dict):
                            raise ValueError("`risk_controls` must be mapping.")
                        risk_controls["max_turnover_per_rebalance"] = float(best_struct["turnover_cap"])

                        tmp_back = make_tmp(f"gate_back_{gate_idx}", back_cfg)
                        tmp_exec = make_tmp(f"gate_exec_{gate_idx}", exec_cfg)
                        _, _, _, back_summary, *_ = run_backtest(
                            config_data_path=config_data_path,
                            config_backtest_path=tmp_back,
                            config_execution_path=tmp_exec,
                        )
                        row = {
                            "stage": "gate",
                            "run_id": gate_idx,
                            "threshold": threshold,
                            "bad_state_multiplier": bad_mult,
                            "lookback_rebalances": lookback,
                            "min_history_rebalances": min_history,
                            "train_summary": gate_train_summary,
                            "backtest_summary": back_summary,
                            "score": _composite_score(train_summary=gate_train_summary, back_summary=back_summary),
                        }
                        gate_rows.append(row)
                        if gate_idx % 20 == 0:
                            print(
                                f"[gate {gate_idx:03d}] score={row['score']:.6f} "
                                f"ann={back_summary.get('annualized_return')} "
                                f"sharpe={back_summary.get('sharpe_ratio')} "
                                f"dd={back_summary.get('max_drawdown')}"
                            )

        gate_rows_sorted = sorted(gate_rows, key=lambda r: float(r["score"]), reverse=True)
        best_gate = gate_rows_sorted[0]

        # Compose final recommended configs.
        recommended_model_cfg = _copy_cfg(best_model_cfg)
        recommended_back_cfg = _copy_cfg(base_back_cfg)
        recommended_exec_cfg = _copy_cfg(base_exec_cfg)

        back = recommended_back_cfg.setdefault("backtest", {})
        constraints = back.setdefault("constraints", {})
        objective = back.setdefault("objective", {})
        overlay_cfg = back.setdefault("risk_overlay", {})
        gate_cfg = back.setdefault("signal_quality_gate", {})
        if (
            not isinstance(back, dict)
            or not isinstance(constraints, dict)
            or not isinstance(objective, dict)
            or not isinstance(overlay_cfg, dict)
            or not isinstance(gate_cfg, dict)
        ):
            raise ValueError("Backtest config sections must be mappings.")

        back["rebalance_frequency"] = "every_n_days"
        back["rebalance_every_n_days"] = int(best_struct["rebalance_days"])
        constraints["weight_max"] = float(best_struct["weight_max"])
        objective["risk_aversion_lambda"] = float(best_struct["risk_aversion_lambda"])
        overlay_cfg["enabled"] = bool(best_struct["overlay_enabled"])
        overlay_cfg["vol_target_annual"] = 0.18
        overlay_cfg["min_leverage"] = 0.35
        overlay_cfg["max_leverage"] = 1.0

        gate_cfg["enabled"] = True
        gate_cfg["metric"] = "oos_cs_ic_spearman"
        gate_cfg["lookback_rebalances"] = int(best_gate["lookback_rebalances"])
        gate_cfg["min_history_rebalances"] = int(best_gate["min_history_rebalances"])
        gate_cfg["threshold"] = float(best_gate["threshold"])
        gate_cfg["bad_state_multiplier"] = float(best_gate["bad_state_multiplier"])

        risk_controls = recommended_exec_cfg.setdefault("risk_controls", {})
        if not isinstance(risk_controls, dict):
            raise ValueError("`risk_controls` must be mapping.")
        risk_controls["max_turnover_per_rebalance"] = float(best_struct["turnover_cap"])

        rec_model_path = out_dir / "config_model.alpha_v2.tuned.yaml"
        rec_back_path = out_dir / "config_backtest.alpha_v2.tuned.yaml"
        rec_exec_path = out_dir / "config_execution.alpha_v2.tuned.yaml"
        _write_yaml(rec_model_path, recommended_model_cfg)
        _write_yaml(rec_back_path, recommended_back_cfg)
        _write_yaml(rec_exec_path, recommended_exec_cfg)

        # Evaluate baseline and final recommendation for report.
        _, _, _, baseline_train, *_ = run_train(
            config_data_path=config_data_path,
            config_model_path=baseline_model_path,
            config_backtest_path=baseline_backtest_path,
        )
        _, _, _, baseline_back, *_ = run_backtest(
            config_data_path=config_data_path,
            config_backtest_path=baseline_backtest_path,
            config_execution_path=baseline_execution_path,
        )
        _, _, _, final_train, *_ = run_train(
            config_data_path=config_data_path,
            config_model_path=rec_model_path,
            config_backtest_path=rec_back_path,
        )
        _, _, _, final_back, *_ = run_backtest(
            config_data_path=config_data_path,
            config_backtest_path=rec_back_path,
            config_execution_path=rec_exec_path,
        )

        report = {
            "baseline": {
                "train_summary": baseline_train,
                "backtest_summary": baseline_back,
                "score": _composite_score(train_summary=baseline_train, back_summary=baseline_back),
            },
            "recommended": {
                "train_summary": final_train,
                "backtest_summary": final_back,
                "score": _composite_score(train_summary=final_train, back_summary=final_back),
            },
            "delta": {
                "annualized_return": _safe_metric(final_back.get("annualized_return"))
                - _safe_metric(baseline_back.get("annualized_return")),
                "sharpe_ratio": _safe_metric(final_back.get("sharpe_ratio"))
                - _safe_metric(baseline_back.get("sharpe_ratio")),
                "max_drawdown": _safe_metric(final_back.get("max_drawdown"))
                - _safe_metric(baseline_back.get("max_drawdown")),
                "oos_cs_ic_mean": _safe_metric(final_train.get("oos_cs_ic_mean"))
                - _safe_metric(baseline_train.get("oos_cs_ic_mean")),
                "oos_top_bottom_mean": _safe_metric(final_train.get("oos_top_bottom_mean"))
                - _safe_metric(baseline_train.get("oos_top_bottom_mean")),
                "score": _composite_score(train_summary=final_train, back_summary=final_back)
                - _composite_score(train_summary=baseline_train, back_summary=baseline_back),
            },
            "best_model_preset": best_model_params,
            "best_structural": {
                "rebalance_days": best_struct["rebalance_days"],
                "weight_max": best_struct["weight_max"],
                "turnover_cap": best_struct["turnover_cap"],
                "risk_aversion_lambda": best_struct["risk_aversion_lambda"],
                "overlay_enabled": best_struct["overlay_enabled"],
            },
            "best_gate": {
                "threshold": best_gate["threshold"],
                "bad_state_multiplier": best_gate["bad_state_multiplier"],
                "lookback_rebalances": best_gate["lookback_rebalances"],
                "min_history_rebalances": best_gate["min_history_rebalances"],
            },
            "artifacts": {
                "recommended_model": str(rec_model_path),
                "recommended_backtest": str(rec_back_path),
                "recommended_execution": str(rec_exec_path),
            },
        }

        model_sweep_path = out_dir / "model_sweep.json"
        structural_sweep_path = out_dir / "structural_sweep.json"
        gate_sweep_path = out_dir / "gate_sweep.json"
        report_path = out_dir / "report.json"
        model_sweep_path.write_text(json.dumps(model_rows_sorted, indent=2), encoding="utf-8")
        structural_sweep_path.write_text(json.dumps(structural_rows_sorted, indent=2), encoding="utf-8")
        gate_sweep_path.write_text(json.dumps(gate_rows_sorted, indent=2), encoding="utf-8")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        # Also export compact CSV views.
        pd.DataFrame(
            [
                {
                    "run_id": row["run_id"],
                    "score": row["score"],
                    "n_estimators": row["preset"]["n_estimators"],
                    "max_depth": row["preset"]["max_depth"],
                    "learning_rate": row["preset"]["learning_rate"],
                    "subsample": row["preset"]["subsample"],
                    "colsample_bytree": row["preset"]["colsample_bytree"],
                    "annualized_return": row["backtest_summary"].get("annualized_return"),
                    "sharpe_ratio": row["backtest_summary"].get("sharpe_ratio"),
                    "max_drawdown": row["backtest_summary"].get("max_drawdown"),
                    "oos_cs_ic_mean": row["train_summary"].get("oos_cs_ic_mean"),
                    "oos_top_bottom_mean": row["train_summary"].get("oos_top_bottom_mean"),
                }
                for row in model_rows_sorted
            ]
        ).to_csv(out_dir / "model_sweep.csv", index=False)
        pd.DataFrame(
            [
                {
                    "run_id": row["run_id"],
                    "score": row["score"],
                    "rebalance_days": row["rebalance_days"],
                    "weight_max": row["weight_max"],
                    "turnover_cap": row["turnover_cap"],
                    "risk_aversion_lambda": row["risk_aversion_lambda"],
                    "overlay_enabled": row["overlay_enabled"],
                    "annualized_return": row["backtest_summary"].get("annualized_return"),
                    "sharpe_ratio": row["backtest_summary"].get("sharpe_ratio"),
                    "max_drawdown": row["backtest_summary"].get("max_drawdown"),
                }
                for row in structural_rows_sorted
            ]
        ).to_csv(out_dir / "structural_sweep.csv", index=False)
        pd.DataFrame(
            [
                {
                    "run_id": row["run_id"],
                    "score": row["score"],
                    "threshold": row["threshold"],
                    "bad_state_multiplier": row["bad_state_multiplier"],
                    "lookback_rebalances": row["lookback_rebalances"],
                    "min_history_rebalances": row["min_history_rebalances"],
                    "annualized_return": row["backtest_summary"].get("annualized_return"),
                    "sharpe_ratio": row["backtest_summary"].get("sharpe_ratio"),
                    "max_drawdown": row["backtest_summary"].get("max_drawdown"),
                    "signal_gate_active_rate": row["backtest_summary"].get("signal_gate_active_rate"),
                }
                for row in gate_rows_sorted
            ]
        ).to_csv(out_dir / "gate_sweep.csv", index=False)

        if args.apply_defaults:
            _write_yaml(PROJECT_ROOT / "configs" / "config_model.yaml", recommended_model_cfg)
            _write_yaml(PROJECT_ROOT / "configs" / "config_backtest.yaml", recommended_back_cfg)
            _write_yaml(PROJECT_ROOT / "configs" / "config_execution.yaml", recommended_exec_cfg)

        print(f"Saved report: {report_path}")
        print(f"Recommended model: {rec_model_path}")
        print(f"Recommended backtest: {rec_back_path}")
        print(f"Recommended execution: {rec_exec_path}")
        print(
            "Delta summary: "
            f"ann_return={report['delta']['annualized_return']} "
            f"sharpe={report['delta']['sharpe_ratio']} "
            f"max_dd={report['delta']['max_drawdown']} "
            f"ic_mean={report['delta']['oos_cs_ic_mean']} "
            f"score={report['delta']['score']}"
        )
    finally:
        for tmp in created_tmp:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
