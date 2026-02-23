from __future__ import annotations

import argparse
import json
import sys
import uuid
from copy import deepcopy
from pathlib import Path
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
from src.model_xgb import run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize for lower drawdown while preserving Sharpe on a strict holdout period. "
            "Search is staged: structural -> risk overlay -> signal gate."
        )
    )
    parser.add_argument("--config-data", type=Path, default=Path("configs/config_data.yaml"))
    parser.add_argument("--config-model", type=Path, default=Path("configs/config_model.yaml"))
    parser.add_argument("--config-backtest", type=Path, default=Path("configs/config_backtest.yaml"))
    parser.add_argument("--config-execution", type=Path, default=Path("configs/config_execution.yaml"))
    parser.add_argument(
        "--holdout-start",
        type=str,
        default="2024-01-01",
        help="Inclusive holdout start date in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--max-weekly-sharpe-drop",
        type=float,
        default=0.00,
        help="Maximum allowed drop in holdout weekly_sharpe_ratio vs baseline.",
    )
    parser.add_argument(
        "--max-sharpe-drop",
        type=float,
        default=0.00,
        help="Maximum allowed drop in holdout sharpe_ratio vs baseline.",
    )
    parser.add_argument("--rebalance-days", type=str, default="15,20")
    parser.add_argument("--weight-max-values", type=str, default="0.18,0.20")
    parser.add_argument("--turnover-cap-values", type=str, default="0.30,0.35")
    parser.add_argument("--risk-lambda-values", type=str, default="12,15,20")
    parser.add_argument("--overlay-vol-target-values", type=str, default="0.16,0.18")
    parser.add_argument("--overlay-min-leverage-values", type=str, default="0.35,0.50")
    parser.add_argument("--overlay-lookback-values", type=str, default="42,63")
    parser.add_argument("--overlay-dd-trigger-values", type=str, default="-0.08,-0.10")
    parser.add_argument("--overlay-dd-mult-values", type=str, default="0.40,0.60")
    parser.add_argument("--gate-threshold-values", type=str, default="0.015,0.020,0.025")
    parser.add_argument("--gate-bad-mult-values", type=str, default="0.50,0.60,0.70")
    parser.add_argument("--gate-lookback-values", type=str, default="20")
    parser.add_argument("--gate-min-history-values", type=str, default="8,10")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/drawdown_holdout_tuning"),
    )
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument(
        "--apply-defaults",
        action="store_true",
        help="If feasible best candidate exists, overwrite configs/config_backtest.yaml and config_execution.yaml.",
    )
    return parser.parse_args()


def _parse_ints(raw: str) -> list[int]:
    out = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one integer value.")
    return out


def _parse_floats(raw: str) -> list[float]:
    out = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one float value.")
    return out


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _annualized_stats(returns: pd.Series, periods_per_year: float) -> dict[str, float | None]:
    r = returns.astype(float).dropna()
    if r.empty:
        return {
            "n_obs": 0,
            "total_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe_ratio": None,
        }

    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    ann_return = float((equity.iloc[-1] ** (periods_per_year / len(r))) - 1.0) if len(r) > 0 else None
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else None
    sharpe = float(ann_return / ann_vol) if ann_return is not None and ann_vol and ann_vol > 0 else None
    return {
        "n_obs": int(len(r)),
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
    }


def _max_drawdown(returns: pd.Series) -> float | None:
    r = returns.astype(float).dropna()
    if r.empty:
        return None
    equity = (1.0 + r).cumprod()
    peaks = equity.cummax()
    drawdown = equity / peaks - 1.0
    return float(drawdown.min())


def _compute_metrics(daily_returns: pd.DataFrame, start_date: pd.Timestamp | None = None) -> dict[str, Any]:
    frame = daily_returns.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=False).dt.tz_localize(None)
    frame = frame.sort_values("date")
    if start_date is not None:
        frame = frame[frame["date"] >= pd.Timestamp(start_date)]
    if frame.empty:
        return {
            "start_date": None,
            "end_date": None,
            "n_days": 0,
            "total_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe_ratio": None,
            "weekly_sharpe_ratio": None,
            "max_drawdown": None,
        }

    series = frame["portfolio_return_net"].astype(float)
    daily = _annualized_stats(series, periods_per_year=252.0)
    weekly_series = (
        frame.set_index("date")["portfolio_return_net"]
        .resample("W-FRI")
        .apply(lambda x: float((1.0 + x).prod() - 1.0))
    )
    weekly = _annualized_stats(weekly_series, periods_per_year=52.0)
    return {
        "start_date": str(frame["date"].min().date()),
        "end_date": str(frame["date"].max().date()),
        "n_days": int(len(frame)),
        "total_return": daily["total_return"],
        "annualized_return": daily["annualized_return"],
        "annualized_volatility": daily["annualized_volatility"],
        "sharpe_ratio": daily["sharpe_ratio"],
        "weekly_sharpe_ratio": weekly["sharpe_ratio"],
        "max_drawdown": _max_drawdown(series),
    }


def _with_rebalance_days(cfg: dict[str, Any], rebalance_days: int) -> dict[str, Any]:
    out = deepcopy(cfg)
    back = out.setdefault("backtest", {})
    if not isinstance(back, dict):
        raise ValueError("`backtest` section must be a mapping.")
    back["rebalance_frequency"] = "every_n_days"
    back["rebalance_every_n_days"] = int(rebalance_days)
    return out


def _prepare_candidate_cfgs(
    base_back_cfg: dict[str, Any],
    base_exec_cfg: dict[str, Any],
    *,
    rebalance_days: int,
    weight_max: float,
    turnover_cap: float,
    risk_lambda: float,
    overlay_enabled: bool,
    overlay_vol_target: float,
    overlay_min_leverage: float,
    overlay_lookback_days: int,
    overlay_dd_trigger: float,
    overlay_dd_multiplier: float,
    gate_enabled: bool,
    gate_threshold: float,
    gate_bad_mult: float,
    gate_lookback: int,
    gate_min_history: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    back_cfg = deepcopy(base_back_cfg)
    exec_cfg = deepcopy(base_exec_cfg)

    back = back_cfg.setdefault("backtest", {})
    constraints = back.setdefault("constraints", {})
    objective = back.setdefault("objective", {})
    overlay_cfg = back.setdefault("risk_overlay", {})
    overlay_dd_cfg = overlay_cfg.setdefault("drawdown_de_risk", {})
    gate_cfg = back.setdefault("signal_quality_gate", {})
    if (
        not isinstance(back, dict)
        or not isinstance(constraints, dict)
        or not isinstance(objective, dict)
        or not isinstance(overlay_cfg, dict)
        or not isinstance(overlay_dd_cfg, dict)
        or not isinstance(gate_cfg, dict)
    ):
        raise ValueError("Backtest config sections must be mappings.")

    back["rebalance_frequency"] = "every_n_days"
    back["rebalance_every_n_days"] = int(rebalance_days)
    constraints["weight_max"] = float(weight_max)
    objective["risk_aversion_lambda"] = float(risk_lambda)

    overlay_cfg["enabled"] = bool(overlay_enabled)
    overlay_cfg["vol_target_annual"] = float(overlay_vol_target)
    overlay_cfg["realized_vol_lookback_days"] = int(overlay_lookback_days)
    overlay_cfg["min_leverage"] = float(overlay_min_leverage)
    overlay_cfg["max_leverage"] = 1.0
    overlay_dd_cfg["enabled"] = True
    overlay_dd_cfg["drawdown_trigger"] = float(overlay_dd_trigger)
    overlay_dd_cfg["leverage_multiplier"] = float(overlay_dd_multiplier)

    gate_cfg["enabled"] = bool(gate_enabled)
    gate_cfg["metric"] = "oos_cs_ic_spearman"
    gate_cfg["lookback_rebalances"] = int(gate_lookback)
    gate_cfg["min_history_rebalances"] = int(gate_min_history)
    gate_cfg["threshold"] = float(gate_threshold)
    gate_cfg["bad_state_multiplier"] = float(gate_bad_mult)

    risk_controls = exec_cfg.setdefault("risk_controls", {})
    if not isinstance(risk_controls, dict):
        raise ValueError("`risk_controls` section must be a mapping.")
    risk_controls["max_turnover_per_rebalance"] = float(turnover_cap)
    return back_cfg, exec_cfg


def _row_score(
    row: dict[str, Any],
    baseline_holdout: dict[str, Any],
    baseline_full: dict[str, Any],
    max_weekly_sharpe_drop: float,
    max_sharpe_drop: float,
) -> tuple[bool, float]:
    holdout_weekly = _safe_float(row.get("holdout_weekly_sharpe_ratio"))
    holdout_sharpe = _safe_float(row.get("holdout_sharpe_ratio"))
    holdout_dd = _safe_float(row.get("holdout_max_drawdown"))
    holdout_ann = _safe_float(row.get("holdout_annualized_return"))
    full_weekly = _safe_float(row.get("full_weekly_sharpe_ratio"))
    full_dd = _safe_float(row.get("full_max_drawdown"))

    base_holdout_weekly = _safe_float(baseline_holdout.get("weekly_sharpe_ratio"))
    base_holdout_sharpe = _safe_float(baseline_holdout.get("sharpe_ratio"))
    base_holdout_dd = _safe_float(baseline_holdout.get("max_drawdown"))
    base_holdout_ann = _safe_float(baseline_holdout.get("annualized_return"))
    base_full_weekly = _safe_float(baseline_full.get("weekly_sharpe_ratio"))
    base_full_dd = _safe_float(baseline_full.get("max_drawdown"))

    weekly_delta = (holdout_weekly - base_holdout_weekly) if holdout_weekly is not None and base_holdout_weekly is not None else 0.0
    sharpe_delta = (holdout_sharpe - base_holdout_sharpe) if holdout_sharpe is not None and base_holdout_sharpe is not None else 0.0
    holdout_dd_impr = (holdout_dd - base_holdout_dd) if holdout_dd is not None and base_holdout_dd is not None else -1.0
    holdout_ann_delta = (holdout_ann - base_holdout_ann) if holdout_ann is not None and base_holdout_ann is not None else 0.0
    full_weekly_delta = (full_weekly - base_full_weekly) if full_weekly is not None and base_full_weekly is not None else 0.0
    full_dd_impr = (full_dd - base_full_dd) if full_dd is not None and base_full_dd is not None else 0.0

    feasible = True
    if base_holdout_weekly is not None and holdout_weekly is not None:
        feasible = feasible and (weekly_delta >= -float(max_weekly_sharpe_drop))
    if base_holdout_sharpe is not None and holdout_sharpe is not None:
        feasible = feasible and (sharpe_delta >= -float(max_sharpe_drop))

    score = (
        12.0 * holdout_dd_impr
        + 3.0 * weekly_delta
        + 1.5 * sharpe_delta
        + 0.8 * holdout_ann_delta
        + 2.0 * full_dd_impr
        + 0.4 * full_weekly_delta
    )
    if not feasible:
        score -= 1_000.0
    return feasible, float(score)


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    return out


def _select_best(
    rows: list[dict[str, Any]],
    baseline_holdout: dict[str, Any],
    baseline_full: dict[str, Any],
    max_weekly_sharpe_drop: float,
    max_sharpe_drop: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No candidate rows were generated.")
    best: dict[str, Any] | None = None
    for row in rows:
        feasible, score = _row_score(
            row=row,
            baseline_holdout=baseline_holdout,
            baseline_full=baseline_full,
            max_weekly_sharpe_drop=max_weekly_sharpe_drop,
            max_sharpe_drop=max_sharpe_drop,
        )
        row["feasible"] = bool(feasible)
        row["selection_score"] = float(score)
        if best is None:
            best = row
            continue
        if float(row["selection_score"]) > float(best["selection_score"]):
            best = row
    assert best is not None
    return best


def main() -> None:
    args = parse_args()
    config_data_path = args.config_data.resolve()
    config_model_path = args.config_model.resolve()
    config_backtest_path = args.config_backtest.resolve()
    config_execution_path = args.config_execution.resolve()
    holdout_start = pd.Timestamp(args.holdout_start)

    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rebalance_days_values = _parse_ints(args.rebalance_days)
    weight_max_values = _parse_floats(args.weight_max_values)
    turnover_cap_values = _parse_floats(args.turnover_cap_values)
    risk_lambda_values = _parse_floats(args.risk_lambda_values)
    overlay_vol_target_values = _parse_floats(args.overlay_vol_target_values)
    overlay_min_leverage_values = _parse_floats(args.overlay_min_leverage_values)
    overlay_lookback_values = _parse_ints(args.overlay_lookback_values)
    overlay_dd_trigger_values = _parse_floats(args.overlay_dd_trigger_values)
    overlay_dd_mult_values = _parse_floats(args.overlay_dd_mult_values)
    gate_threshold_values = _parse_floats(args.gate_threshold_values)
    gate_bad_mult_values = _parse_floats(args.gate_bad_mult_values)
    gate_lookback_values = _parse_ints(args.gate_lookback_values)
    gate_min_history_values = _parse_ints(args.gate_min_history_values)

    base_back_cfg = load_yaml(config_backtest_path)
    base_exec_cfg = load_yaml(config_execution_path)

    token = uuid.uuid4().hex[:10]
    created_tmp: list[Path] = []

    def make_tmp(stem: str, payload: dict[str, Any]) -> Path:
        path = (PROJECT_ROOT / "configs" / f"tmp_drawdown_tune_{token}_{stem}.yaml").resolve()
        _write_yaml(path, payload)
        created_tmp.append(path)
        return path

    try:
        if args.refresh_data:
            run_fetch_data(config_path=config_data_path)
        run_build_panel(config_path=config_data_path)

        # Baseline run.
        _, _, _, baseline_train, *_ = run_train(
            config_data_path=config_data_path,
            config_model_path=config_model_path,
            config_backtest_path=config_backtest_path,
        )
        baseline_daily, _, _, baseline_summary, *_ = run_backtest(
            config_data_path=config_data_path,
            config_backtest_path=config_backtest_path,
            config_execution_path=config_execution_path,
        )
        baseline_holdout = _compute_metrics(baseline_daily, start_date=holdout_start)
        baseline_full = {
            "annualized_return": baseline_summary.get("annualized_return"),
            "annualized_volatility": baseline_summary.get("annualized_volatility"),
            "sharpe_ratio": baseline_summary.get("sharpe_ratio"),
            "weekly_sharpe_ratio": baseline_summary.get("weekly_sharpe_ratio"),
            "max_drawdown": baseline_summary.get("max_drawdown"),
        }

        print(
            "[baseline] "
            f"full_weekly_sharpe={baseline_full.get('weekly_sharpe_ratio')} "
            f"full_max_dd={baseline_full.get('max_drawdown')} "
            f"holdout_weekly_sharpe={baseline_holdout.get('weekly_sharpe_ratio')} "
            f"holdout_max_dd={baseline_holdout.get('max_drawdown')}"
        )

        # Stage 1: structural sweep.
        structural_rows: list[dict[str, Any]] = []
        train_cache: dict[int, dict[str, Any]] = {}
        run_id = 0
        for rebalance_days in rebalance_days_values:
            train_back_cfg = _with_rebalance_days(base_back_cfg, rebalance_days=rebalance_days)
            tmp_train_back = make_tmp(f"struct_train_back_{rebalance_days}", train_back_cfg)
            _, _, _, train_summary, *_ = run_train(
                config_data_path=config_data_path,
                config_model_path=config_model_path,
                config_backtest_path=tmp_train_back,
            )
            train_cache[rebalance_days] = train_summary

            for weight_max in weight_max_values:
                for turnover_cap in turnover_cap_values:
                    for risk_lambda in risk_lambda_values:
                        run_id += 1
                        back_cfg, exec_cfg = _prepare_candidate_cfgs(
                            base_back_cfg=base_back_cfg,
                            base_exec_cfg=base_exec_cfg,
                            rebalance_days=rebalance_days,
                            weight_max=weight_max,
                            turnover_cap=turnover_cap,
                            risk_lambda=risk_lambda,
                            overlay_enabled=bool(base_back_cfg.get("backtest", {}).get("risk_overlay", {}).get("enabled", False)),
                            overlay_vol_target=float(base_back_cfg.get("backtest", {}).get("risk_overlay", {}).get("vol_target_annual", 0.18)),
                            overlay_min_leverage=float(base_back_cfg.get("backtest", {}).get("risk_overlay", {}).get("min_leverage", 0.35)),
                            overlay_lookback_days=int(base_back_cfg.get("backtest", {}).get("risk_overlay", {}).get("realized_vol_lookback_days", 63)),
                            overlay_dd_trigger=float(
                                base_back_cfg.get("backtest", {})
                                .get("risk_overlay", {})
                                .get("drawdown_de_risk", {})
                                .get("drawdown_trigger", -0.10)
                            ),
                            overlay_dd_multiplier=float(
                                base_back_cfg.get("backtest", {})
                                .get("risk_overlay", {})
                                .get("drawdown_de_risk", {})
                                .get("leverage_multiplier", 0.5)
                            ),
                            gate_enabled=bool(base_back_cfg.get("backtest", {}).get("signal_quality_gate", {}).get("enabled", True)),
                            gate_threshold=float(base_back_cfg.get("backtest", {}).get("signal_quality_gate", {}).get("threshold", 0.02)),
                            gate_bad_mult=float(
                                base_back_cfg.get("backtest", {}).get("signal_quality_gate", {}).get("bad_state_multiplier", 0.6)
                            ),
                            gate_lookback=int(
                                base_back_cfg.get("backtest", {}).get("signal_quality_gate", {}).get("lookback_rebalances", 20)
                            ),
                            gate_min_history=int(
                                base_back_cfg.get("backtest", {}).get("signal_quality_gate", {}).get("min_history_rebalances", 8)
                            ),
                        )
                        tmp_back = make_tmp(f"struct_back_{run_id}", back_cfg)
                        tmp_exec = make_tmp(f"struct_exec_{run_id}", exec_cfg)

                        daily, _, _, summary, *_ = run_backtest(
                            config_data_path=config_data_path,
                            config_backtest_path=tmp_back,
                            config_execution_path=tmp_exec,
                        )
                        holdout = _compute_metrics(daily, start_date=holdout_start)
                        row = {
                            "stage": "structural",
                            "run_id": run_id,
                            "rebalance_days": rebalance_days,
                            "weight_max": weight_max,
                            "turnover_cap": turnover_cap,
                            "risk_aversion_lambda": risk_lambda,
                            "overlay_enabled": bool(back_cfg["backtest"]["risk_overlay"]["enabled"]),
                            "gate_enabled": bool(back_cfg["backtest"]["signal_quality_gate"]["enabled"]),
                            "full_annualized_return": summary.get("annualized_return"),
                            "full_sharpe_ratio": summary.get("sharpe_ratio"),
                            "full_weekly_sharpe_ratio": summary.get("weekly_sharpe_ratio"),
                            "full_max_drawdown": summary.get("max_drawdown"),
                            "holdout_annualized_return": holdout.get("annualized_return"),
                            "holdout_sharpe_ratio": holdout.get("sharpe_ratio"),
                            "holdout_weekly_sharpe_ratio": holdout.get("weekly_sharpe_ratio"),
                            "holdout_max_drawdown": holdout.get("max_drawdown"),
                            "train_oos_cs_ic_mean": train_summary.get("oos_cs_ic_mean"),
                            "train_oos_top_bottom_mean": train_summary.get("oos_top_bottom_mean"),
                        }
                        structural_rows.append(row)
                        print(
                            f"[struct {run_id:03d}] holdout_weekly_sharpe={row['holdout_weekly_sharpe_ratio']} "
                            f"holdout_max_dd={row['holdout_max_drawdown']}"
                        )

        best_struct = _select_best(
            rows=structural_rows,
            baseline_holdout=baseline_holdout,
            baseline_full=baseline_full,
            max_weekly_sharpe_drop=float(args.max_weekly_sharpe_drop),
            max_sharpe_drop=float(args.max_sharpe_drop),
        )
        print(
            "[best structural] "
            f"run={best_struct['run_id']} "
            f"feasible={best_struct['feasible']} "
            f"score={best_struct['selection_score']}"
        )

        best_rebalance_days = int(best_struct["rebalance_days"])
        # Ensure predictions are aligned with the selected cadence before stage 2/3.
        stage_back_cfg = _with_rebalance_days(base_back_cfg, rebalance_days=best_rebalance_days)
        tmp_stage_back = make_tmp("best_rebalance_for_stage2", stage_back_cfg)
        _, _, _, stage_train_summary, *_ = run_train(
            config_data_path=config_data_path,
            config_model_path=config_model_path,
            config_backtest_path=tmp_stage_back,
        )

        # Stage 2: risk overlay sweep.
        overlay_rows: list[dict[str, Any]] = []
        overlay_run_id = 0

        # Include disabled overlay candidate.
        overlay_candidates: list[dict[str, Any]] = [
            {
                "overlay_enabled": False,
                "overlay_vol_target": float(base_back_cfg.get("backtest", {}).get("risk_overlay", {}).get("vol_target_annual", 0.18)),
                "overlay_min_leverage": float(base_back_cfg.get("backtest", {}).get("risk_overlay", {}).get("min_leverage", 0.35)),
                "overlay_lookback_days": int(
                    base_back_cfg.get("backtest", {}).get("risk_overlay", {}).get("realized_vol_lookback_days", 63)
                ),
                "overlay_dd_trigger": float(
                    base_back_cfg.get("backtest", {})
                    .get("risk_overlay", {})
                    .get("drawdown_de_risk", {})
                    .get("drawdown_trigger", -0.10)
                ),
                "overlay_dd_multiplier": float(
                    base_back_cfg.get("backtest", {})
                    .get("risk_overlay", {})
                    .get("drawdown_de_risk", {})
                    .get("leverage_multiplier", 0.5)
                ),
            }
        ]
        for vol_target in overlay_vol_target_values:
            for min_lev in overlay_min_leverage_values:
                for lookback_days in overlay_lookback_values:
                    for dd_trigger in overlay_dd_trigger_values:
                        for dd_mult in overlay_dd_mult_values:
                            overlay_candidates.append(
                                {
                                    "overlay_enabled": True,
                                    "overlay_vol_target": float(vol_target),
                                    "overlay_min_leverage": float(min_lev),
                                    "overlay_lookback_days": int(lookback_days),
                                    "overlay_dd_trigger": float(dd_trigger),
                                    "overlay_dd_multiplier": float(dd_mult),
                                }
                            )

        for candidate in overlay_candidates:
            overlay_run_id += 1
            back_cfg, exec_cfg = _prepare_candidate_cfgs(
                base_back_cfg=base_back_cfg,
                base_exec_cfg=base_exec_cfg,
                rebalance_days=best_rebalance_days,
                weight_max=float(best_struct["weight_max"]),
                turnover_cap=float(best_struct["turnover_cap"]),
                risk_lambda=float(best_struct["risk_aversion_lambda"]),
                overlay_enabled=bool(candidate["overlay_enabled"]),
                overlay_vol_target=float(candidate["overlay_vol_target"]),
                overlay_min_leverage=float(candidate["overlay_min_leverage"]),
                overlay_lookback_days=int(candidate["overlay_lookback_days"]),
                overlay_dd_trigger=float(candidate["overlay_dd_trigger"]),
                overlay_dd_multiplier=float(candidate["overlay_dd_multiplier"]),
                gate_enabled=bool(base_back_cfg.get("backtest", {}).get("signal_quality_gate", {}).get("enabled", True)),
                gate_threshold=float(base_back_cfg.get("backtest", {}).get("signal_quality_gate", {}).get("threshold", 0.02)),
                gate_bad_mult=float(base_back_cfg.get("backtest", {}).get("signal_quality_gate", {}).get("bad_state_multiplier", 0.6)),
                gate_lookback=int(base_back_cfg.get("backtest", {}).get("signal_quality_gate", {}).get("lookback_rebalances", 20)),
                gate_min_history=int(
                    base_back_cfg.get("backtest", {}).get("signal_quality_gate", {}).get("min_history_rebalances", 8)
                ),
            )
            tmp_back = make_tmp(f"overlay_back_{overlay_run_id}", back_cfg)
            tmp_exec = make_tmp(f"overlay_exec_{overlay_run_id}", exec_cfg)
            daily, _, _, summary, *_ = run_backtest(
                config_data_path=config_data_path,
                config_backtest_path=tmp_back,
                config_execution_path=tmp_exec,
            )
            holdout = _compute_metrics(daily, start_date=holdout_start)
            row = {
                "stage": "overlay",
                "run_id": overlay_run_id,
                "rebalance_days": best_rebalance_days,
                "weight_max": float(best_struct["weight_max"]),
                "turnover_cap": float(best_struct["turnover_cap"]),
                "risk_aversion_lambda": float(best_struct["risk_aversion_lambda"]),
                "overlay_enabled": bool(candidate["overlay_enabled"]),
                "overlay_vol_target": float(candidate["overlay_vol_target"]),
                "overlay_min_leverage": float(candidate["overlay_min_leverage"]),
                "overlay_lookback_days": int(candidate["overlay_lookback_days"]),
                "overlay_dd_trigger": float(candidate["overlay_dd_trigger"]),
                "overlay_dd_multiplier": float(candidate["overlay_dd_multiplier"]),
                "gate_enabled": bool(back_cfg["backtest"]["signal_quality_gate"]["enabled"]),
                "gate_threshold": float(back_cfg["backtest"]["signal_quality_gate"]["threshold"]),
                "gate_bad_mult": float(back_cfg["backtest"]["signal_quality_gate"]["bad_state_multiplier"]),
                "gate_lookback": int(back_cfg["backtest"]["signal_quality_gate"]["lookback_rebalances"]),
                "gate_min_history": int(back_cfg["backtest"]["signal_quality_gate"]["min_history_rebalances"]),
                "full_annualized_return": summary.get("annualized_return"),
                "full_sharpe_ratio": summary.get("sharpe_ratio"),
                "full_weekly_sharpe_ratio": summary.get("weekly_sharpe_ratio"),
                "full_max_drawdown": summary.get("max_drawdown"),
                "holdout_annualized_return": holdout.get("annualized_return"),
                "holdout_sharpe_ratio": holdout.get("sharpe_ratio"),
                "holdout_weekly_sharpe_ratio": holdout.get("weekly_sharpe_ratio"),
                "holdout_max_drawdown": holdout.get("max_drawdown"),
                "train_oos_cs_ic_mean": stage_train_summary.get("oos_cs_ic_mean"),
                "train_oos_top_bottom_mean": stage_train_summary.get("oos_top_bottom_mean"),
            }
            overlay_rows.append(row)
            if overlay_run_id % 10 == 0:
                print(
                    f"[overlay {overlay_run_id:03d}] holdout_weekly_sharpe={row['holdout_weekly_sharpe_ratio']} "
                    f"holdout_max_dd={row['holdout_max_drawdown']}"
                )

        best_overlay = _select_best(
            rows=overlay_rows,
            baseline_holdout=baseline_holdout,
            baseline_full=baseline_full,
            max_weekly_sharpe_drop=float(args.max_weekly_sharpe_drop),
            max_sharpe_drop=float(args.max_sharpe_drop),
        )
        print(
            "[best overlay] "
            f"run={best_overlay['run_id']} "
            f"feasible={best_overlay['feasible']} "
            f"score={best_overlay['selection_score']}"
        )

        # Stage 3: gate sweep.
        gate_rows: list[dict[str, Any]] = []
        gate_run_id = 0
        gate_candidates: list[dict[str, Any]] = [{"gate_enabled": False, "gate_threshold": 0.0, "gate_bad_mult": 1.0, "gate_lookback": 20, "gate_min_history": 8}]
        for threshold in gate_threshold_values:
            for bad_mult in gate_bad_mult_values:
                for lookback in gate_lookback_values:
                    for min_hist in gate_min_history_values:
                        gate_candidates.append(
                            {
                                "gate_enabled": True,
                                "gate_threshold": float(threshold),
                                "gate_bad_mult": float(bad_mult),
                                "gate_lookback": int(lookback),
                                "gate_min_history": int(min_hist),
                            }
                        )

        for candidate in gate_candidates:
            gate_run_id += 1
            back_cfg, exec_cfg = _prepare_candidate_cfgs(
                base_back_cfg=base_back_cfg,
                base_exec_cfg=base_exec_cfg,
                rebalance_days=best_rebalance_days,
                weight_max=float(best_struct["weight_max"]),
                turnover_cap=float(best_struct["turnover_cap"]),
                risk_lambda=float(best_struct["risk_aversion_lambda"]),
                overlay_enabled=bool(best_overlay["overlay_enabled"]),
                overlay_vol_target=float(best_overlay["overlay_vol_target"]),
                overlay_min_leverage=float(best_overlay["overlay_min_leverage"]),
                overlay_lookback_days=int(best_overlay["overlay_lookback_days"]),
                overlay_dd_trigger=float(best_overlay["overlay_dd_trigger"]),
                overlay_dd_multiplier=float(best_overlay["overlay_dd_multiplier"]),
                gate_enabled=bool(candidate["gate_enabled"]),
                gate_threshold=float(candidate["gate_threshold"]),
                gate_bad_mult=float(candidate["gate_bad_mult"]),
                gate_lookback=int(candidate["gate_lookback"]),
                gate_min_history=int(candidate["gate_min_history"]),
            )
            tmp_back = make_tmp(f"gate_back_{gate_run_id}", back_cfg)
            tmp_exec = make_tmp(f"gate_exec_{gate_run_id}", exec_cfg)
            daily, _, _, summary, *_ = run_backtest(
                config_data_path=config_data_path,
                config_backtest_path=tmp_back,
                config_execution_path=tmp_exec,
            )
            holdout = _compute_metrics(daily, start_date=holdout_start)
            row = {
                "stage": "gate",
                "run_id": gate_run_id,
                "rebalance_days": best_rebalance_days,
                "weight_max": float(best_struct["weight_max"]),
                "turnover_cap": float(best_struct["turnover_cap"]),
                "risk_aversion_lambda": float(best_struct["risk_aversion_lambda"]),
                "overlay_enabled": bool(best_overlay["overlay_enabled"]),
                "overlay_vol_target": float(best_overlay["overlay_vol_target"]),
                "overlay_min_leverage": float(best_overlay["overlay_min_leverage"]),
                "overlay_lookback_days": int(best_overlay["overlay_lookback_days"]),
                "overlay_dd_trigger": float(best_overlay["overlay_dd_trigger"]),
                "overlay_dd_multiplier": float(best_overlay["overlay_dd_multiplier"]),
                "gate_enabled": bool(candidate["gate_enabled"]),
                "gate_threshold": float(candidate["gate_threshold"]),
                "gate_bad_mult": float(candidate["gate_bad_mult"]),
                "gate_lookback": int(candidate["gate_lookback"]),
                "gate_min_history": int(candidate["gate_min_history"]),
                "full_annualized_return": summary.get("annualized_return"),
                "full_sharpe_ratio": summary.get("sharpe_ratio"),
                "full_weekly_sharpe_ratio": summary.get("weekly_sharpe_ratio"),
                "full_max_drawdown": summary.get("max_drawdown"),
                "holdout_annualized_return": holdout.get("annualized_return"),
                "holdout_sharpe_ratio": holdout.get("sharpe_ratio"),
                "holdout_weekly_sharpe_ratio": holdout.get("weekly_sharpe_ratio"),
                "holdout_max_drawdown": holdout.get("max_drawdown"),
                "train_oos_cs_ic_mean": stage_train_summary.get("oos_cs_ic_mean"),
                "train_oos_top_bottom_mean": stage_train_summary.get("oos_top_bottom_mean"),
            }
            gate_rows.append(row)

        best_gate = _select_best(
            rows=gate_rows,
            baseline_holdout=baseline_holdout,
            baseline_full=baseline_full,
            max_weekly_sharpe_drop=float(args.max_weekly_sharpe_drop),
            max_sharpe_drop=float(args.max_sharpe_drop),
        )
        print(
            "[best gate] "
            f"run={best_gate['run_id']} "
            f"feasible={best_gate['feasible']} "
            f"score={best_gate['selection_score']}"
        )

        # Build final recommended configs from stage winners and evaluate one final time.
        recommended_back_cfg, recommended_exec_cfg = _prepare_candidate_cfgs(
            base_back_cfg=base_back_cfg,
            base_exec_cfg=base_exec_cfg,
            rebalance_days=best_rebalance_days,
            weight_max=float(best_struct["weight_max"]),
            turnover_cap=float(best_struct["turnover_cap"]),
            risk_lambda=float(best_struct["risk_aversion_lambda"]),
            overlay_enabled=bool(best_overlay["overlay_enabled"]),
            overlay_vol_target=float(best_overlay["overlay_vol_target"]),
            overlay_min_leverage=float(best_overlay["overlay_min_leverage"]),
            overlay_lookback_days=int(best_overlay["overlay_lookback_days"]),
            overlay_dd_trigger=float(best_overlay["overlay_dd_trigger"]),
            overlay_dd_multiplier=float(best_overlay["overlay_dd_multiplier"]),
            gate_enabled=bool(best_gate["gate_enabled"]),
            gate_threshold=float(best_gate["gate_threshold"]),
            gate_bad_mult=float(best_gate["gate_bad_mult"]),
            gate_lookback=int(best_gate["gate_lookback"]),
            gate_min_history=int(best_gate["gate_min_history"]),
        )

        rec_back_path = out_dir / "config_backtest.drawdown_holdout.tuned.yaml"
        rec_exec_path = out_dir / "config_execution.drawdown_holdout.tuned.yaml"
        _write_yaml(rec_back_path, recommended_back_cfg)
        _write_yaml(rec_exec_path, recommended_exec_cfg)

        tmp_rec_back = make_tmp("rec_back_eval", recommended_back_cfg)
        tmp_rec_exec = make_tmp("rec_exec_eval", recommended_exec_cfg)
        _, _, _, recommended_train, *_ = run_train(
            config_data_path=config_data_path,
            config_model_path=config_model_path,
            config_backtest_path=tmp_rec_back,
        )
        rec_daily, _, _, rec_summary, *_ = run_backtest(
            config_data_path=config_data_path,
            config_backtest_path=tmp_rec_back,
            config_execution_path=tmp_rec_exec,
        )
        rec_holdout = _compute_metrics(rec_daily, start_date=holdout_start)
        rec_full = {
            "annualized_return": rec_summary.get("annualized_return"),
            "annualized_volatility": rec_summary.get("annualized_volatility"),
            "sharpe_ratio": rec_summary.get("sharpe_ratio"),
            "weekly_sharpe_ratio": rec_summary.get("weekly_sharpe_ratio"),
            "max_drawdown": rec_summary.get("max_drawdown"),
        }

        final_feasible, final_score = _row_score(
            row={
                "holdout_weekly_sharpe_ratio": rec_holdout.get("weekly_sharpe_ratio"),
                "holdout_sharpe_ratio": rec_holdout.get("sharpe_ratio"),
                "holdout_max_drawdown": rec_holdout.get("max_drawdown"),
                "holdout_annualized_return": rec_holdout.get("annualized_return"),
                "full_weekly_sharpe_ratio": rec_full.get("weekly_sharpe_ratio"),
                "full_max_drawdown": rec_full.get("max_drawdown"),
            },
            baseline_holdout=baseline_holdout,
            baseline_full=baseline_full,
            max_weekly_sharpe_drop=float(args.max_weekly_sharpe_drop),
            max_sharpe_drop=float(args.max_sharpe_drop),
        )

        report = {
            "holdout_start": str(holdout_start.date()),
            "constraints": {
                "max_weekly_sharpe_drop": float(args.max_weekly_sharpe_drop),
                "max_sharpe_drop": float(args.max_sharpe_drop),
            },
            "baseline": {
                "train_summary": baseline_train,
                "full_metrics": baseline_full,
                "holdout_metrics": baseline_holdout,
            },
            "recommended": {
                "train_summary": recommended_train,
                "full_metrics": rec_full,
                "holdout_metrics": rec_holdout,
                "feasible": bool(final_feasible),
                "selection_score": float(final_score),
            },
            "delta": {
                "full_weekly_sharpe_ratio": _safe_float(rec_full.get("weekly_sharpe_ratio"))
                - _safe_float(baseline_full.get("weekly_sharpe_ratio"))
                if _safe_float(rec_full.get("weekly_sharpe_ratio")) is not None
                and _safe_float(baseline_full.get("weekly_sharpe_ratio")) is not None
                else None,
                "full_max_drawdown": _safe_float(rec_full.get("max_drawdown"))
                - _safe_float(baseline_full.get("max_drawdown"))
                if _safe_float(rec_full.get("max_drawdown")) is not None
                and _safe_float(baseline_full.get("max_drawdown")) is not None
                else None,
                "holdout_weekly_sharpe_ratio": _safe_float(rec_holdout.get("weekly_sharpe_ratio"))
                - _safe_float(baseline_holdout.get("weekly_sharpe_ratio"))
                if _safe_float(rec_holdout.get("weekly_sharpe_ratio")) is not None
                and _safe_float(baseline_holdout.get("weekly_sharpe_ratio")) is not None
                else None,
                "holdout_max_drawdown": _safe_float(rec_holdout.get("max_drawdown"))
                - _safe_float(baseline_holdout.get("max_drawdown"))
                if _safe_float(rec_holdout.get("max_drawdown")) is not None
                and _safe_float(baseline_holdout.get("max_drawdown")) is not None
                else None,
            },
            "best_structural": best_struct,
            "best_overlay": best_overlay,
            "best_gate": best_gate,
            "artifacts": {
                "recommended_backtest_config": str(rec_back_path),
                "recommended_execution_config": str(rec_exec_path),
            },
        }

        (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        pd.DataFrame([_flatten_row(r) for r in structural_rows]).sort_values("selection_score", ascending=False).to_csv(
            out_dir / "structural_sweep.csv", index=False
        )
        pd.DataFrame([_flatten_row(r) for r in overlay_rows]).sort_values("selection_score", ascending=False).to_csv(
            out_dir / "overlay_sweep.csv", index=False
        )
        pd.DataFrame([_flatten_row(r) for r in gate_rows]).sort_values("selection_score", ascending=False).to_csv(
            out_dir / "gate_sweep.csv", index=False
        )

        if args.apply_defaults and bool(final_feasible):
            _write_yaml(PROJECT_ROOT / "configs" / "config_backtest.yaml", recommended_back_cfg)
            _write_yaml(PROJECT_ROOT / "configs" / "config_execution.yaml", recommended_exec_cfg)
            print("Applied recommended backtest/execution configs to defaults.")
        elif args.apply_defaults:
            print("Best candidate is not feasible under Sharpe constraints; defaults not changed.")

        print(
            f"Saved report: {out_dir / 'report.json'}\n"
            f"Delta holdout: weekly_sharpe={report['delta']['holdout_weekly_sharpe_ratio']} "
            f"max_dd={report['delta']['holdout_max_drawdown']}\n"
            f"Delta full: weekly_sharpe={report['delta']['full_weekly_sharpe_ratio']} "
            f"max_dd={report['delta']['full_max_drawdown']}"
        )
    finally:
        for tmp in created_tmp:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
