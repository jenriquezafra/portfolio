from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _initial_weights(n_assets: int, weight_max: float) -> np.ndarray:
    if n_assets == 0:
        return np.array([], dtype=float)
    base = np.full(n_assets, 1.0 / n_assets, dtype=float)
    if float(np.max(base)) <= weight_max:
        return base

    top_k = int(math.ceil(1.0 / weight_max))
    w = np.zeros(n_assets, dtype=float)
    w[:top_k] = 1.0 / top_k
    return w


def _fallback_weights(mu: np.ndarray, weight_max: float) -> np.ndarray:
    n_assets = len(mu)
    order = np.argsort(-mu)
    w = np.zeros(n_assets, dtype=float)
    remaining = 1.0

    for idx in order:
        if remaining <= 0:
            break
        alloc = min(weight_max, remaining)
        w[idx] = alloc
        remaining -= alloc
    if remaining > 1e-10:
        w[order[0]] += remaining
    return w / w.sum()


def optimize_mean_variance_long_only(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    prev_weights: pd.Series | None,
    risk_aversion_lambda: float,
    turnover_penalty_eta: float,
    weight_max: float,
    fully_invested: bool = True,
) -> pd.Series:
    tickers = expected_returns.index.astype(str).tolist()
    n_assets = len(tickers)
    if n_assets == 0:
        raise ValueError("Expected returns series is empty.")
    if weight_max <= 0 or weight_max > 1:
        raise ValueError("`weight_max` must be in (0, 1].")
    if fully_invested and weight_max * n_assets < 1.0:
        raise ValueError("Infeasible constraints: weight_max * n_assets < 1 for fully invested long-only portfolio.")

    mu = expected_returns.to_numpy(dtype=float)
    cov = covariance.reindex(index=tickers, columns=tickers).fillna(0.0).to_numpy(dtype=float)

    if prev_weights is None:
        w_prev = _initial_weights(n_assets=n_assets, weight_max=weight_max)
    else:
        w_prev = prev_weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
        if w_prev.sum() > 0:
            w_prev = w_prev / w_prev.sum()
        else:
            w_prev = _initial_weights(n_assets=n_assets, weight_max=weight_max)

    def objective(w: np.ndarray) -> float:
        mean_term = -float(np.dot(mu, w))
        risk_term = 0.5 * float(risk_aversion_lambda) * float(w.T @ cov @ w)
        turnover_term = float(turnover_penalty_eta) * float(np.sum((w - w_prev) ** 2))
        return mean_term + risk_term + turnover_term

    bounds = [(0.0, weight_max) for _ in range(n_assets)]
    constraints = []
    if fully_invested:
        constraints.append({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)})
    else:
        constraints.append({"type": "ineq", "fun": lambda w: float(1.0 - np.sum(w))})

    x0 = np.clip(w_prev, 0.0, weight_max)
    if x0.sum() == 0:
        x0 = _initial_weights(n_assets=n_assets, weight_max=weight_max)
    else:
        x0 = x0 / x0.sum()

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10, "disp": False},
    )

    if not result.success:
        weights = _fallback_weights(mu=mu, weight_max=weight_max)
    else:
        weights = np.clip(result.x, 0.0, weight_max)
        if fully_invested:
            total = float(weights.sum())
            if total <= 0:
                weights = _fallback_weights(mu=mu, weight_max=weight_max)
            else:
                weights = weights / total

    return pd.Series(weights, index=tickers, name="weight")


def apply_turnover_cap(
    target_weights: pd.Series,
    prev_weights: pd.Series | None,
    max_turnover_per_rebalance: float | None,
) -> tuple[pd.Series, float]:
    if prev_weights is None:
        prev = pd.Series(0.0, index=target_weights.index)
    else:
        prev = prev_weights.reindex(target_weights.index).fillna(0.0)

    raw_turnover = float((target_weights - prev).abs().sum())

    if (
        max_turnover_per_rebalance is None
        or max_turnover_per_rebalance <= 0
        or raw_turnover <= max_turnover_per_rebalance
        or raw_turnover == 0
    ):
        return target_weights, raw_turnover

    alpha = max_turnover_per_rebalance / raw_turnover
    capped = prev + alpha * (target_weights - prev)
    capped_turnover = float((capped - prev).abs().sum())
    return capped, capped_turnover


def _cap_and_normalize_positive(
    raw_strength: pd.Series,
    budget: float,
    max_weight: float,
) -> pd.Series:
    strengths = raw_strength.fillna(0.0).clip(lower=0.0)
    if strengths.empty:
        return strengths
    if budget < 0:
        raise ValueError("`budget` must be non-negative.")
    if max_weight <= 0:
        raise ValueError("`max_weight` must be positive.")
    if budget > 0 and max_weight * len(strengths) + 1e-12 < budget:
        raise ValueError("Infeasible constraints for side budget and max_weight.")
    if budget == 0:
        return pd.Series(0.0, index=strengths.index, dtype=float)

    if float(strengths.sum()) <= 0:
        strengths = pd.Series(1.0, index=strengths.index, dtype=float)

    remaining = strengths.copy()
    fixed = pd.Series(0.0, index=strengths.index, dtype=float)
    remaining_budget = float(budget)

    for _ in range(len(strengths) + 2):
        if remaining.empty or remaining_budget <= 1e-12:
            break

        alloc = remaining_budget * remaining / float(remaining.sum())
        hit_cap = alloc > (max_weight + 1e-12)
        if not hit_cap.any():
            fixed.loc[alloc.index] += alloc
            remaining_budget = 0.0
            break

        capped_names = alloc[hit_cap].index
        fixed.loc[capped_names] = max_weight
        remaining = remaining.drop(index=capped_names)
        remaining_budget = budget - float(fixed.sum())

    if remaining_budget > 1e-9:
        # Numerical fallback: spread residual budget uniformly among uncapped names.
        uncapped = fixed[fixed < (max_weight - 1e-12)].index
        if len(uncapped) > 0:
            extra = min(remaining_budget / len(uncapped), max_weight)
            fixed.loc[uncapped] += extra

    fixed = fixed.clip(lower=0.0, upper=max_weight)
    total = float(fixed.sum())
    if total <= 0:
        return pd.Series(0.0, index=strengths.index, dtype=float)
    return fixed * (budget / total)


def signal_to_long_only_weights(
    signal: pd.Series,
    volatility: pd.Series | None,
    weight_max: float,
    fully_invested: bool = True,
) -> pd.Series:
    if signal.empty:
        raise ValueError("`signal` is empty.")
    if weight_max <= 0 or weight_max > 1:
        raise ValueError("`weight_max` must be in (0, 1].")
    if fully_invested and weight_max * len(signal) < 1.0:
        raise ValueError("Infeasible constraints for long-only with full investment.")

    s = signal.astype(float).copy()
    if volatility is not None:
        vol = volatility.reindex(s.index).astype(float)
        med = float(vol[vol > 0].median()) if (vol > 0).any() else 1.0
        vol = vol.where(vol > 1e-12, med if med > 0 else 1.0).fillna(med if med > 0 else 1.0)
        s = s / vol

    strengths = s.clip(lower=0.0)
    if float(strengths.sum()) <= 0:
        strengths = pd.Series(1.0, index=s.index, dtype=float)

    budget = 1.0 if fully_invested else min(1.0, float(strengths.sum()))
    weights = _cap_and_normalize_positive(
        raw_strength=strengths,
        budget=budget,
        max_weight=weight_max,
    )
    return weights.rename("weight")


def signal_to_market_neutral_weights(
    signal: pd.Series,
    volatility: pd.Series | None,
    weight_max_abs: float,
    gross_exposure_target: float = 1.0,
    long_quantile: float = 0.20,
    short_quantile: float = 0.20,
) -> pd.Series:
    if signal.empty:
        raise ValueError("`signal` is empty.")
    if not (0.0 < long_quantile <= 0.5):
        raise ValueError("`long_quantile` must be in (0, 0.5].")
    if not (0.0 < short_quantile <= 0.5):
        raise ValueError("`short_quantile` must be in (0, 0.5].")
    if weight_max_abs <= 0:
        raise ValueError("`weight_max_abs` must be positive.")
    if gross_exposure_target <= 0:
        raise ValueError("`gross_exposure_target` must be positive.")

    s = signal.astype(float).copy()
    n_assets = len(s)
    long_n = max(1, int(np.floor(n_assets * long_quantile)))
    short_n = max(1, int(np.floor(n_assets * short_quantile)))
    if long_n + short_n > n_assets:
        raise ValueError("Quantile buckets overlap; reduce long/short quantiles.")

    ordered = s.sort_values()
    short_idx = ordered.index[:short_n]
    long_idx = ordered.index[-long_n:]

    if volatility is not None:
        vol = volatility.reindex(s.index).astype(float)
        med = float(vol[vol > 0].median()) if (vol > 0).any() else 1.0
        vol = vol.where(vol > 1e-12, med if med > 0 else 1.0).fillna(med if med > 0 else 1.0)
    else:
        vol = pd.Series(1.0, index=s.index, dtype=float)

    long_strength = s.loc[long_idx].clip(lower=0.0) / vol.loc[long_idx]
    short_strength = (-s.loc[short_idx]).clip(lower=0.0) / vol.loc[short_idx]
    if float(long_strength.sum()) <= 0:
        long_strength = s.loc[long_idx].abs() / vol.loc[long_idx]
    if float(short_strength.sum()) <= 0:
        short_strength = s.loc[short_idx].abs() / vol.loc[short_idx]
    if float(long_strength.sum()) <= 0:
        long_strength = pd.Series(1.0, index=long_idx, dtype=float)
    if float(short_strength.sum()) <= 0:
        short_strength = pd.Series(1.0, index=short_idx, dtype=float)

    side_budget = gross_exposure_target / 2.0
    if weight_max_abs * max(len(long_idx), len(short_idx)) + 1e-12 < side_budget:
        raise ValueError("Infeasible market-neutral side budgets for weight cap.")

    long_w = _cap_and_normalize_positive(
        raw_strength=long_strength,
        budget=side_budget,
        max_weight=weight_max_abs,
    )
    short_w = _cap_and_normalize_positive(
        raw_strength=short_strength,
        budget=side_budget,
        max_weight=weight_max_abs,
    )

    out = pd.Series(0.0, index=s.index, dtype=float, name="weight")
    out.loc[long_w.index] = long_w.values
    out.loc[short_w.index] = -short_w.values
    return out


def _project_market_neutral_weights(
    current: pd.Series,
    gross_exposure_target: float,
    weight_max_abs: float,
) -> pd.Series:
    clipped = current.clip(lower=-weight_max_abs, upper=weight_max_abs)
    long_raw = clipped.clip(lower=0.0)
    short_raw = (-clipped.clip(upper=0.0))

    if float(long_raw.sum()) <= 0:
        long_raw = pd.Series(1.0, index=current.index, dtype=float)
    if float(short_raw.sum()) <= 0:
        short_raw = pd.Series(1.0, index=current.index, dtype=float)

    side_budget = gross_exposure_target / 2.0
    long_w = _cap_and_normalize_positive(long_raw, budget=side_budget, max_weight=weight_max_abs)
    short_w = _cap_and_normalize_positive(short_raw, budget=side_budget, max_weight=weight_max_abs)
    projected = long_w - short_w
    return projected.reindex(current.index).fillna(0.0)


def factor_neutralize_market_neutral_weights(
    weights: pd.Series,
    asset_factor_exposures: pd.DataFrame,
    target_factor_exposures: pd.Series | dict[str, float] | None = None,
    gross_exposure_target: float = 1.0,
    weight_max_abs: float = 1.0,
    max_iter: int = 10,
    tol: float = 1e-4,
    ridge: float = 1e-6,
) -> pd.Series:
    if weights.empty:
        raise ValueError("`weights` is empty.")
    if gross_exposure_target <= 0:
        raise ValueError("`gross_exposure_target` must be positive.")
    if weight_max_abs <= 0:
        raise ValueError("`weight_max_abs` must be positive.")

    w = weights.astype(float).copy()
    exposures = asset_factor_exposures.reindex(index=w.index).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if exposures.empty or exposures.shape[1] == 0:
        return _project_market_neutral_weights(w, gross_exposure_target, weight_max_abs).rename("weight")

    factor_names = list(exposures.columns)
    if target_factor_exposures is None:
        target = pd.Series(0.0, index=factor_names, dtype=float)
    else:
        target = pd.Series(target_factor_exposures, dtype=float).reindex(factor_names).fillna(0.0)

    B = exposures.to_numpy(dtype=float)  # (n_assets, n_factors)
    if np.all(np.std(B, axis=0) <= 1e-12):
        return _project_market_neutral_weights(w, gross_exposure_target, weight_max_abs).rename("weight")

    curr = _project_market_neutral_weights(w, gross_exposure_target, weight_max_abs)
    target_vec = target.to_numpy(dtype=float)
    k = B.shape[1]
    bt_b = B.T @ B
    solve_mat = bt_b + float(ridge) * np.eye(k)

    for _ in range(max_iter):
        err = (B.T @ curr.to_numpy(dtype=float)) - target_vec
        if float(np.max(np.abs(err))) <= tol:
            break
        try:
            step = np.linalg.solve(solve_mat, err)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(solve_mat, err, rcond=None)[0]
        candidate = curr.to_numpy(dtype=float) - (B @ step)
        curr = _project_market_neutral_weights(
            pd.Series(candidate, index=curr.index, dtype=float),
            gross_exposure_target=gross_exposure_target,
            weight_max_abs=weight_max_abs,
        )

    return curr.rename("weight")


def beta_neutralize_market_neutral_weights(
    weights: pd.Series,
    asset_betas: pd.Series,
    target_beta: float = 0.0,
    gross_exposure_target: float = 1.0,
    weight_max_abs: float = 1.0,
    max_iter: int = 8,
    tol: float = 1e-4,
) -> pd.Series:
    exposures = pd.DataFrame({"beta": asset_betas.astype(float)})
    neutral = factor_neutralize_market_neutral_weights(
        weights=weights,
        asset_factor_exposures=exposures,
        target_factor_exposures={"beta": float(target_beta)},
        gross_exposure_target=gross_exposure_target,
        weight_max_abs=weight_max_abs,
        max_iter=max_iter,
        tol=tol,
    )
    return neutral.rename("weight")
