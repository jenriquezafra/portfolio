from __future__ import annotations

import numpy as np
import pandas as pd

from src.optimizer import (
    beta_neutralize_market_neutral_weights,
    factor_neutralize_market_neutral_weights,
    signal_to_long_only_weights,
    signal_to_market_neutral_weights,
)


def test_signal_to_long_only_weights_respects_budget_and_cap() -> None:
    idx = ["A", "B", "C", "D", "E", "F"]
    signal = pd.Series([1.2, 0.8, 0.4, -0.2, 0.1, 0.0], index=idx)
    vol = pd.Series([0.2, 0.3, 0.5, 0.4, 0.6, 0.7], index=idx)
    weights = signal_to_long_only_weights(signal=signal, volatility=vol, weight_max=0.40, fully_invested=True)

    assert np.isclose(float(weights.sum()), 1.0, atol=1e-9)
    assert float(weights.min()) >= -1e-12
    assert float(weights.max()) <= 0.40 + 1e-9


def test_signal_to_market_neutral_weights_has_50_50_sides() -> None:
    idx = [f"T{i}" for i in range(20)]
    signal = pd.Series(np.linspace(-1.0, 1.0, len(idx)), index=idx)
    vol = pd.Series(np.linspace(0.1, 0.3, len(idx)), index=idx)
    weights = signal_to_market_neutral_weights(
        signal=signal,
        volatility=vol,
        weight_max_abs=0.20,
        gross_exposure_target=1.0,
        long_quantile=0.20,
        short_quantile=0.20,
    )

    long_sum = float(weights[weights > 0].sum())
    short_sum_abs = float((-weights[weights < 0]).sum())
    assert np.isclose(long_sum, 0.5, atol=1e-9)
    assert np.isclose(short_sum_abs, 0.5, atol=1e-9)
    assert np.isclose(float(weights.sum()), 0.0, atol=1e-9)
    assert float(weights.abs().max()) <= 0.20 + 1e-9


def test_beta_neutralization_reduces_beta_error() -> None:
    idx = [f"T{i}" for i in range(20)]
    signal = pd.Series(np.linspace(-2.0, 2.0, len(idx)), index=idx)
    vol = pd.Series(0.2, index=idx)
    betas = pd.Series(np.linspace(0.4, 1.6, len(idx)), index=idx)
    base = signal_to_market_neutral_weights(
        signal=signal,
        volatility=vol,
        weight_max_abs=0.10,
        gross_exposure_target=1.0,
        long_quantile=0.30,
        short_quantile=0.30,
    )

    beta_before = float((base * betas).sum())
    neutral = beta_neutralize_market_neutral_weights(
        weights=base,
        asset_betas=betas,
        target_beta=0.0,
        gross_exposure_target=1.0,
        weight_max_abs=0.10,
    )
    beta_after = float((neutral * betas).sum())

    assert abs(beta_after) <= abs(beta_before) + 1e-9
    assert np.isclose(float(neutral.sum()), 0.0, atol=1e-6)


def test_multifactor_neutralization_reduces_joint_exposure_error() -> None:
    idx = [f"T{i}" for i in range(30)]
    signal = pd.Series(np.linspace(-1.5, 1.5, len(idx)), index=idx)
    base = signal_to_market_neutral_weights(
        signal=signal,
        volatility=pd.Series(0.2, index=idx),
        weight_max_abs=0.10,
        gross_exposure_target=1.0,
        long_quantile=0.30,
        short_quantile=0.30,
    )
    exposures = pd.DataFrame(
        {
            "nasdaq_proxy": np.linspace(0.5, 1.5, len(idx)),
            "growth_proxy": np.linspace(-0.7, 0.9, len(idx)),
        },
        index=idx,
    )
    before = exposures.T.dot(base)
    neutral = factor_neutralize_market_neutral_weights(
        weights=base,
        asset_factor_exposures=exposures,
        target_factor_exposures={"nasdaq_proxy": 0.0, "growth_proxy": 0.0},
        gross_exposure_target=1.0,
        weight_max_abs=0.10,
    )
    after = exposures.T.dot(neutral)

    assert float(np.linalg.norm(after.values, ord=2)) <= float(np.linalg.norm(before.values, ord=2)) + 1e-9
    assert np.isclose(float(neutral.sum()), 0.0, atol=1e-6)
