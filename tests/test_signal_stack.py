from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals import (
    ENGINEERED_SIGNAL_COLUMNS,
    build_composite_signal,
    build_price_volume_signal_panel,
    normalize_signal_weights,
    parse_signal_stack_weights,
    standardize_cross_sectional_signal,
)


def _make_clean_prices(n_days: int = 180, n_tickers: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(303)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rows: list[dict[str, object]] = []
    for i, ticker in enumerate(tickers):
        drift = 0.0002 * (i - n_tickers / 2)
        noise = rng.normal(0.0, 0.012, size=n_days)
        ret = drift + noise
        prices = 100.0 * np.cumprod(1.0 + ret)
        volume = 1_000_000.0 + (50_000.0 * rng.normal(size=n_days))
        for d, px, vol in zip(dates, prices, volume):
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "adj_close": float(px),
                    "volume": float(max(vol, 1000.0)),
                }
            )
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def test_price_volume_signals_no_lookahead_under_future_mutation() -> None:
    clean = _make_clean_prices()
    panel = build_price_volume_signal_panel(clean)
    ticker = "T00"
    valid = panel[(panel["ticker"] == ticker) & panel[list(ENGINEERED_SIGNAL_COLUMNS)].notna().all(axis=1)]
    assert not valid.empty
    ref = valid.iloc[20]
    ref_date = pd.Timestamp(ref["date"])

    baseline_values = panel[(panel["ticker"] == ticker) & (panel["date"] == ref_date)][
        list(ENGINEERED_SIGNAL_COLUMNS)
    ].iloc[0]

    mutated = clean.copy()
    mask = (mutated["ticker"] == ticker) & (pd.to_datetime(mutated["date"]) > ref_date)
    mutated.loc[mask, "adj_close"] = mutated.loc[mask, "adj_close"] * 3.0
    panel_mut = build_price_volume_signal_panel(mutated)
    mutated_values = panel_mut[(panel_mut["ticker"] == ticker) & (panel_mut["date"] == ref_date)][
        list(ENGINEERED_SIGNAL_COLUMNS)
    ].iloc[0]

    assert np.allclose(
        baseline_values.to_numpy(dtype=float),
        mutated_values.to_numpy(dtype=float),
        atol=1e-12,
        equal_nan=True,
    )


def test_standardize_signal_handles_constant_and_nan() -> None:
    s = pd.Series([np.nan, 5.0, 5.0, 5.0], index=["a", "b", "c", "d"])
    out = standardize_cross_sectional_signal(s, winsor_quantile=0.05)
    assert np.allclose(out.to_numpy(dtype=float), np.zeros(len(out)), atol=1e-12)


def test_weight_normalization_and_zero_fallback() -> None:
    all_zero = normalize_signal_weights(
        {
            "model_prediction": 0.0,
            "momentum_residual": 0.0,
            "reversal_regime": 0.0,
            "vol_compression_breakout": 0.0,
            "liquidity_impulse": 0.0,
        },
        normalize_weights=True,
    )
    assert all_zero["model_prediction"] == 1.0
    assert all_zero["momentum_residual"] == 0.0

    parsed, normalize_flag = parse_signal_stack_weights(
        {
            "normalize_weights": True,
            "weights": {
                "model_prediction": 1.0,
                "momentum_residual": 0.5,
                "reversal_regime": 0.5,
                "vol_compression_breakout": 0.0,
                "liquidity_impulse": 0.0,
            },
        }
    )
    assert normalize_flag is True
    l1 = float(sum(abs(v) for v in parsed.values()))
    assert np.isclose(l1, 1.0, atol=1e-12)


def test_build_composite_signal_outputs_finite_values() -> None:
    idx = pd.Index(["A", "B", "C", "D"], name="ticker")
    model_signal = pd.Series([0.2, -0.1, 0.4, 0.0], index=idx, dtype=float)
    engineered = pd.DataFrame(
        {
            "momentum_residual_signal": [0.1, 0.2, np.nan, -0.1],
            "reversal_regime_signal": [0.0, -0.2, 0.3, 0.1],
            "vol_compression_breakout_signal": [0.1, np.nan, -0.1, 0.0],
            "liquidity_impulse_signal": [0.4, 0.0, -0.2, 0.1],
        },
        index=idx,
    )
    composite, _, weights = build_composite_signal(
        model_signal=model_signal,
        engineered_signals=engineered,
        weights={
            "model_prediction": 1.0,
            "momentum_residual": 0.4,
            "reversal_regime": 0.2,
            "vol_compression_breakout": 0.1,
            "liquidity_impulse": 0.3,
        },
        normalize_weights=True,
    )
    assert np.isfinite(composite.to_numpy(dtype=float)).all()
    assert np.isclose(sum(abs(v) for v in weights.values()), 1.0, atol=1e-12)
