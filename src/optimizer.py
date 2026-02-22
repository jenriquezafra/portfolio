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
