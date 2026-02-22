from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_FACTOR_DIAGNOSTIC_THRESHOLDS: dict[str, float] = {
    "ex_ante_mean_abs_max": 0.05,
    "ex_ante_max_abs_max": 0.35,
    "ex_post_beta_abs_max": 0.15,
    "ex_post_beta_yearly_abs_max": 0.35,
    "ex_post_r2_max": 0.15,
    "yearly_min_obs": 60.0,
}


def _to_float_map(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _safe_abs_max(values: list[float]) -> float | None:
    if not values:
        return None
    return float(max(abs(x) for x in values))


def _safe_abs_l1(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(abs(x) for x in values))


def _yearly_beta_sign_flips(by_year: list[dict[str, Any]], factor_name: str, min_obs: int) -> int:
    signs: list[int] = []
    for row in sorted(by_year, key=lambda x: int(x.get("year", 0))):
        n_obs = int(row.get("n_obs", 0) or 0)
        if n_obs < min_obs:
            continue
        betas = row.get("betas")
        if not isinstance(betas, dict):
            continue
        value = betas.get(factor_name)
        if value is None:
            continue
        try:
            s = _sign(float(value))
        except (TypeError, ValueError):
            continue
        if s != 0:
            signs.append(s)

    flips = 0
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            flips += 1
    return flips


def _build_recommendations(
    checks: dict[str, bool],
    metrics: dict[str, Any],
    factor_names: list[str],
) -> list[str]:
    recommendations: list[str] = []
    if not checks.get("ex_ante_mean_abs_ok", True):
        recommendations.append(
            "Reduce average ex-ante factor exposure: tighten neutralization target or increase beta lookback."
        )
    if not checks.get("ex_ante_max_abs_ok", True):
        recommendations.append(
            "Cap extreme ex-ante factor spikes: review quantiles/weight caps and turnover cap interaction."
        )
    if not checks.get("ex_post_beta_abs_ok", True):
        recommendations.append(
            "Residual ex-post factor beta is elevated: recalibrate neutralization frequency and factor set."
        )
    if not checks.get("ex_post_beta_yearly_abs_ok", True):
        recommendations.append(
            "Yearly factor beta instability detected: verify regime robustness and per-year constraints."
        )
    if not checks.get("ex_post_r2_ok", True):
        recommendations.append(
            "High ex-post R2 to factors: strategy may still be driven by market/style beta."
        )
    if not recommendations and factor_names:
        recommendations.append("Factor diagnostics are within configured thresholds.")
    if not factor_names:
        recommendations.append("No factor diagnostics available. Enable beta_neutralization factors in backtest config.")
    if metrics.get("yearly_sign_flip_counts"):
        flips = metrics["yearly_sign_flip_counts"]
        unstable = [name for name, count in flips.items() if int(count) >= 3]
        if unstable:
            recommendations.append(f"Frequent yearly beta sign flips in: {', '.join(sorted(unstable))}.")
    return recommendations


def build_factor_diagnostics_report(
    backtest_summary: dict[str, Any],
    factor_exposure_report: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    limits = dict(DEFAULT_FACTOR_DIAGNOSTIC_THRESHOLDS)
    if thresholds:
        limits.update({k: float(v) for k, v in thresholds.items()})

    factor_names = [str(x) for x in factor_exposure_report.get("factor_names", []) if str(x).strip()]

    ex_ante = factor_exposure_report.get("ex_ante", {})
    ex_post = factor_exposure_report.get("ex_post", {})
    ex_ante_mean = _to_float_map(ex_ante.get("mean"))
    ex_ante_max_abs = _to_float_map(ex_ante.get("max_abs"))

    full_sample = ex_post.get("full_sample", {})
    full_sample_betas = _to_float_map(full_sample.get("betas"))
    full_sample_r2 = full_sample.get("r2")
    try:
        full_sample_r2_float = float(full_sample_r2) if full_sample_r2 is not None else None
    except (TypeError, ValueError):
        full_sample_r2_float = None

    by_year_raw = ex_post.get("by_year", [])
    by_year = by_year_raw if isinstance(by_year_raw, list) else []
    min_obs = int(limits["yearly_min_obs"])

    yearly_beta_abs_values: list[float] = []
    for row in by_year:
        if not isinstance(row, dict):
            continue
        n_obs = int(row.get("n_obs", 0) or 0)
        if n_obs < min_obs:
            continue
        betas = _to_float_map(row.get("betas"))
        yearly_beta_abs_values.extend(abs(v) for v in betas.values())

    yearly_sign_flip_counts = {
        name: _yearly_beta_sign_flips(by_year=by_year, factor_name=name, min_obs=min_obs) for name in factor_names
    }

    metric_ex_ante_mean_abs_max = _safe_abs_max(list(ex_ante_mean.values()))
    metric_ex_ante_max_abs_worst = _safe_abs_max(list(ex_ante_max_abs.values()))
    metric_ex_post_beta_abs_max = _safe_abs_max(list(full_sample_betas.values()))
    metric_ex_post_beta_abs_l1 = _safe_abs_l1(list(full_sample_betas.values()))
    metric_ex_post_beta_yearly_abs_max = _safe_abs_max(yearly_beta_abs_values)

    checks = {
        "ex_ante_mean_abs_ok": (
            metric_ex_ante_mean_abs_max is None
            or metric_ex_ante_mean_abs_max <= float(limits["ex_ante_mean_abs_max"])
        ),
        "ex_ante_max_abs_ok": (
            metric_ex_ante_max_abs_worst is None
            or metric_ex_ante_max_abs_worst <= float(limits["ex_ante_max_abs_max"])
        ),
        "ex_post_beta_abs_ok": (
            metric_ex_post_beta_abs_max is None
            or metric_ex_post_beta_abs_max <= float(limits["ex_post_beta_abs_max"])
        ),
        "ex_post_beta_yearly_abs_ok": (
            metric_ex_post_beta_yearly_abs_max is None
            or metric_ex_post_beta_yearly_abs_max <= float(limits["ex_post_beta_yearly_abs_max"])
        ),
        "ex_post_r2_ok": (
            full_sample_r2_float is None or full_sample_r2_float <= float(limits["ex_post_r2_max"])
        ),
    }
    n_fail = sum(0 if ok else 1 for ok in checks.values())
    if not factor_names:
        status = "missing_factors"
    elif n_fail == 0:
        status = "pass"
    elif n_fail <= 2:
        status = "warning"
    else:
        status = "fail"

    metrics = {
        "full_sample": {
            "ex_ante_mean_abs_max": metric_ex_ante_mean_abs_max,
            "ex_ante_max_abs_worst": metric_ex_ante_max_abs_worst,
            "ex_post_beta_abs_max": metric_ex_post_beta_abs_max,
            "ex_post_beta_abs_l1": metric_ex_post_beta_abs_l1,
            "ex_post_r2": full_sample_r2_float,
        },
        "yearly": {
            "min_obs_filter": min_obs,
            "ex_post_beta_yearly_abs_max": metric_ex_post_beta_yearly_abs_max,
            "yearly_sign_flip_counts": yearly_sign_flip_counts,
        },
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "factor_names": factor_names,
        "thresholds": limits,
        "checks": checks,
        "metrics": metrics,
        "context": {
            "portfolio_mode": backtest_summary.get("portfolio_mode"),
            "allocation_method": backtest_summary.get("allocation_method"),
            "sharpe_ratio": backtest_summary.get("sharpe_ratio"),
            "weekly_sharpe_ratio": backtest_summary.get("weekly_sharpe_ratio"),
            "n_rebalances": backtest_summary.get("n_rebalances"),
            "start_date": backtest_summary.get("start_date"),
            "end_date": backtest_summary.get("end_date"),
        },
        "recommendations": _build_recommendations(
            checks=checks,
            metrics=metrics["yearly"],
            factor_names=factor_names,
        ),
    }


def run_factor_diagnostics_from_outputs(
    project_root: Path,
    output_subdir: str = "outputs/backtests",
    thresholds: dict[str, float] | None = None,
) -> tuple[dict[str, Any], Path]:
    out_dir = (project_root / output_subdir).resolve()
    summary_path = out_dir / "backtest_summary.json"
    factor_exposure_path = out_dir / "factor_exposure_report.json"
    diagnostic_path = out_dir / "factor_diagnostics_report.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing backtest summary: {summary_path}")
    if not factor_exposure_path.exists():
        raise FileNotFoundError(f"Missing factor exposure report: {factor_exposure_path}")

    backtest_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    factor_exposure_report = json.loads(factor_exposure_path.read_text(encoding="utf-8"))

    report = build_factor_diagnostics_report(
        backtest_summary=backtest_summary,
        factor_exposure_report=factor_exposure_report,
        thresholds=thresholds,
    )
    diagnostic_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report, diagnostic_path
