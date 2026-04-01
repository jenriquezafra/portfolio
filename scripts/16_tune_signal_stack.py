from __future__ import annotations

import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_backtest

SIGNAL_KEYS = [
    "model_prediction",
    "momentum_residual",
    "reversal_regime",
    "vol_compression_breakout",
    "liquidity_impulse",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune backtest signal-stack weights (coarse grid + local refinement), score candidates, "
            "and export recommendation artifacts."
        )
    )
    parser.add_argument("--config-data", type=Path, default=Path("configs/config_data.yaml"))
    parser.add_argument("--config-backtest", type=Path, default=Path("configs/config_backtest.yaml"))
    parser.add_argument("--config-execution", type=Path, default=Path("configs/config_execution.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/signal_stack_tuning"),
        help="Directory for tuning comparison artifacts.",
    )
    parser.add_argument(
        "--model-grid",
        type=str,
        default="0.8,1.0,1.2",
        help="Comma-separated model_prediction weights for coarse search.",
    )
    parser.add_argument(
        "--engineered-grid",
        type=str,
        default="0.0,0.15,0.30",
        help="Comma-separated engineered-signal weights for coarse search.",
    )
    parser.add_argument("--top-k-refine", type=int, default=6, help="How many coarse winners to refine locally.")
    parser.add_argument("--refine-step", type=float, default=0.075, help="Weight step for local refinement.")
    parser.add_argument("--drawdown-guardrail", type=float, default=0.05, help="Allowed DD worsening before penalty.")
    parser.add_argument("--turnover-cap", type=float, default=0.40, help="Max allowed average turnover.")
    return parser.parse_args()


def _parse_float_grid(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one grid value.")
    return values


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _prepare_signal_stack_cfg(
    base_cfg: dict[str, Any],
    *,
    enabled: bool,
    weights: dict[str, float],
    normalize_weights: bool = True,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    back = cfg.setdefault("backtest", {})
    if not isinstance(back, dict):
        raise ValueError("`backtest` section must be a mapping.")
    stack = back.setdefault("signal_stack", {})
    if not isinstance(stack, dict):
        raise ValueError("`backtest.signal_stack` must be a mapping.")

    stack["enabled"] = bool(enabled)
    stack["normalize_weights"] = bool(normalize_weights)
    stack["weights"] = {k: float(weights[k]) for k in SIGNAL_KEYS}
    return cfg


def _score_candidate(
    *,
    summary: dict[str, Any],
    baseline: dict[str, float],
    drawdown_guardrail: float,
    turnover_cap: float,
) -> dict[str, float | bool]:
    ann_return = float(summary.get("annualized_return", 0.0) or 0.0)
    weekly_sharpe = float(summary.get("weekly_sharpe_ratio", 0.0) or 0.0)
    max_drawdown = float(summary.get("max_drawdown", 0.0) or 0.0)
    avg_turnover = float(summary.get("average_turnover", 0.0) or 0.0)

    baseline_ann = baseline["annualized_return"]
    baseline_weekly = baseline["weekly_sharpe_ratio"]
    baseline_dd_abs = abs(baseline["max_drawdown"])

    rel_improvement = 0.0
    if abs(baseline_ann) > 1e-12:
        rel_improvement = (ann_return / baseline_ann) - 1.0
    weekly_sharpe_drop = baseline_weekly - weekly_sharpe
    drawdown_worsening = abs(max_drawdown) - baseline_dd_abs
    turnover_excess = avg_turnover - turnover_cap

    penalty_drawdown = max(0.0, drawdown_worsening - drawdown_guardrail) * 5.0
    penalty_turnover = max(0.0, turnover_excess) * 1.5
    score = ann_return + (0.10 * weekly_sharpe) - penalty_drawdown - penalty_turnover

    accepted = (
        rel_improvement >= 0.05
        and weekly_sharpe_drop <= 0.05
        and drawdown_worsening <= 0.05
        and avg_turnover <= turnover_cap
    )
    return {
        "score": float(score),
        "annualized_return": ann_return,
        "weekly_sharpe_ratio": weekly_sharpe,
        "max_drawdown": max_drawdown,
        "average_turnover": avg_turnover,
        "annualized_return_rel_improvement": float(rel_improvement),
        "weekly_sharpe_drop": float(weekly_sharpe_drop),
        "drawdown_worsening": float(drawdown_worsening),
        "turnover_excess": float(turnover_excess),
        "accepted": bool(accepted),
    }


def _coarse_candidates(model_grid: list[float], engineered_grid: list[float]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for m, mom, rev, vol, liq in itertools.product(
        model_grid,
        engineered_grid,
        engineered_grid,
        engineered_grid,
        engineered_grid,
    ):
        out.append(
            {
                "model_prediction": float(m),
                "momentum_residual": float(mom),
                "reversal_regime": float(rev),
                "vol_compression_breakout": float(vol),
                "liquidity_impulse": float(liq),
            }
        )
    return out


def _refine_candidates(
    top_rows: pd.DataFrame,
    *,
    step: float,
) -> list[dict[str, float]]:
    if top_rows.empty:
        return []
    out: list[dict[str, float]] = []
    for _, row in top_rows.iterrows():
        base = {k: float(row[k]) for k in SIGNAL_KEYS}
        out.append(base)
        for key in SIGNAL_KEYS:
            for delta in (-step, step):
                cand = dict(base)
                cand[key] = float(cand[key] + delta)
                if key == "model_prediction":
                    cand[key] = float(min(max(cand[key], 0.40), 1.80))
                else:
                    cand[key] = float(min(max(cand[key], 0.00), 0.80))
                out.append(cand)
    return out


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_data_path = args.config_data.resolve()
    config_backtest_path = args.config_backtest.resolve()
    config_execution_path = args.config_execution.resolve()
    base_backtest = yaml.safe_load(config_backtest_path.read_text(encoding="utf-8"))
    if not isinstance(base_backtest, dict):
        raise ValueError("Invalid backtest config.")

    model_grid = _parse_float_grid(args.model_grid)
    engineered_grid = _parse_float_grid(args.engineered_grid)

    tmp_cfg_path = output_dir / "tmp_backtest_signal_stack.yaml"
    records: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float, float, float]] = set()

    try:
        baseline_weights = {
            "model_prediction": 1.0,
            "momentum_residual": 0.0,
            "reversal_regime": 0.0,
            "vol_compression_breakout": 0.0,
            "liquidity_impulse": 0.0,
        }
        baseline_cfg = _prepare_signal_stack_cfg(
            base_cfg=base_backtest,
            enabled=False,
            weights=baseline_weights,
            normalize_weights=True,
        )
        _write_yaml(tmp_cfg_path, baseline_cfg)
        _, _, _, baseline_summary, *_ = run_backtest(
            config_data_path=config_data_path,
            config_backtest_path=tmp_cfg_path,
            config_execution_path=config_execution_path,
        )
        baseline_metrics = {
            "annualized_return": float(baseline_summary.get("annualized_return", 0.0) or 0.0),
            "weekly_sharpe_ratio": float(baseline_summary.get("weekly_sharpe_ratio", 0.0) or 0.0),
            "max_drawdown": float(baseline_summary.get("max_drawdown", 0.0) or 0.0),
            "average_turnover": float(baseline_summary.get("average_turnover", 0.0) or 0.0),
        }
        print("Baseline")
        print(
            "  "
            f"ann_return={baseline_metrics['annualized_return']:.6f} "
            f"weekly_sharpe={baseline_metrics['weekly_sharpe_ratio']:.6f} "
            f"max_dd={baseline_metrics['max_drawdown']:.6f} "
            f"avg_turnover={baseline_metrics['average_turnover']:.6f}"
        )

        coarse = _coarse_candidates(model_grid=model_grid, engineered_grid=engineered_grid)
        print(f"Running coarse search: {len(coarse)} candidates")
        for idx, weights in enumerate(coarse, start=1):
            key = tuple(weights[k] for k in SIGNAL_KEYS)
            if key in seen:
                continue
            seen.add(key)
            cfg = _prepare_signal_stack_cfg(
                base_cfg=base_backtest,
                enabled=True,
                weights=weights,
                normalize_weights=True,
            )
            _write_yaml(tmp_cfg_path, cfg)
            _, _, _, summary, *_ = run_backtest(
                config_data_path=config_data_path,
                config_backtest_path=tmp_cfg_path,
                config_execution_path=config_execution_path,
            )
            scored = _score_candidate(
                summary=summary,
                baseline=baseline_metrics,
                drawdown_guardrail=float(args.drawdown_guardrail),
                turnover_cap=float(args.turnover_cap),
            )
            rec: dict[str, Any] = {
                "stage": "coarse",
                "candidate_id": f"coarse_{idx:04d}",
                **weights,
                **scored,
            }
            records.append(rec)

        coarse_df = pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)
        top_for_refine = coarse_df.head(max(1, int(args.top_k_refine)))
        refine_candidates = _refine_candidates(top_for_refine, step=float(args.refine_step))
        print(f"Running refined search: {len(refine_candidates)} candidates")
        for idx, weights in enumerate(refine_candidates, start=1):
            key = tuple(weights[k] for k in SIGNAL_KEYS)
            if key in seen:
                continue
            seen.add(key)
            cfg = _prepare_signal_stack_cfg(
                base_cfg=base_backtest,
                enabled=True,
                weights=weights,
                normalize_weights=True,
            )
            _write_yaml(tmp_cfg_path, cfg)
            _, _, _, summary, *_ = run_backtest(
                config_data_path=config_data_path,
                config_backtest_path=tmp_cfg_path,
                config_execution_path=config_execution_path,
            )
            scored = _score_candidate(
                summary=summary,
                baseline=baseline_metrics,
                drawdown_guardrail=float(args.drawdown_guardrail),
                turnover_cap=float(args.turnover_cap),
            )
            rec = {
                "stage": "refined",
                "candidate_id": f"refined_{idx:04d}",
                **weights,
                **scored,
            }
            records.append(rec)
    finally:
        tmp_cfg_path.unlink(missing_ok=True)

    if not records:
        raise RuntimeError("No signal-stack candidates were evaluated.")

    comparison_df = pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)
    best_overall = comparison_df.iloc[0].to_dict()
    accepted_df = comparison_df[comparison_df["accepted"] == True]  # noqa: E712
    best_accepted = accepted_df.iloc[0].to_dict() if not accepted_df.empty else None

    recommended_row = best_accepted if best_accepted is not None else best_overall
    recommended_weights = {k: float(recommended_row[k]) for k in SIGNAL_KEYS}
    recommended_cfg = _prepare_signal_stack_cfg(
        base_cfg=base_backtest,
        enabled=bool(best_accepted is not None),
        weights=recommended_weights if best_accepted is not None else {
            "model_prediction": 1.0,
            "momentum_residual": 0.0,
            "reversal_regime": 0.0,
            "vol_compression_breakout": 0.0,
            "liquidity_impulse": 0.0,
        },
        normalize_weights=True,
    )

    comparison_path = output_dir / "comparison.csv"
    comparison_json_path = output_dir / "comparison.json"
    top_path = output_dir / "top_candidates.json"
    patch_path = output_dir / "recommended_signal_stack_patch.yaml"
    full_cfg_path = output_dir / "config_backtest.signal_stack.recommended.yaml"

    comparison_df.to_csv(comparison_path, index=False)
    comparison_payload = {
        "schema_version": "signal_stack_tuning_v1",
        "baseline_metrics": baseline_metrics,
        "search": {
            "n_candidates": int(len(comparison_df)),
            "coarse_grid_model": model_grid,
            "coarse_grid_engineered": engineered_grid,
            "top_k_refine": int(args.top_k_refine),
            "refine_step": float(args.refine_step),
            "drawdown_guardrail": float(args.drawdown_guardrail),
            "turnover_cap": float(args.turnover_cap),
        },
        "best_overall": best_overall,
        "best_accepted": best_accepted,
        "promotion_allowed": bool(best_accepted is not None),
    }
    comparison_json_path.write_text(json.dumps(comparison_payload, indent=2, sort_keys=True), encoding="utf-8")
    top_path.write_text(
        json.dumps(comparison_df.head(10).to_dict(orient="records"), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    patch_payload = {
        "backtest": {
            "signal_stack": recommended_cfg["backtest"]["signal_stack"],
        }
    }
    patch_path.write_text(yaml.safe_dump(patch_payload, sort_keys=False), encoding="utf-8")
    full_cfg_path.write_text(yaml.safe_dump(recommended_cfg, sort_keys=False), encoding="utf-8")

    print("Signal stack tuning complete")
    print(f"Candidates evaluated: {len(comparison_df)}")
    print(f"Best overall score: {float(best_overall['score']):.6f}")
    if best_accepted is None:
        print("No accepted candidate met promotion criteria; patch defaults to disabled stack.")
    else:
        print(f"Best accepted score: {float(best_accepted['score']):.6f}")
    print(f"Comparison CSV: {comparison_path}")
    print(f"Comparison JSON: {comparison_json_path}")
    print(f"Top candidates JSON: {top_path}")
    print(f"Recommended patch: {patch_path}")
    print(f"Recommended full config: {full_cfg_path}")


if __name__ == "__main__":
    main()
