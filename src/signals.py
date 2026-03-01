from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ENGINEERED_SIGNAL_COLUMNS = (
    "momentum_residual_signal",
    "reversal_regime_signal",
    "vol_compression_breakout_signal",
    "liquidity_impulse_signal",
)

SIGNAL_WEIGHT_KEYS = (
    "model_prediction",
    "momentum_residual",
    "reversal_regime",
    "vol_compression_breakout",
    "liquidity_impulse",
)


def _rolling_group_stat(
    series: pd.Series,
    group: pd.Series,
    window: int,
    min_periods: int,
    fn: str,
) -> pd.Series:
    return (
        series.groupby(group)
        .rolling(window=window, min_periods=min_periods)
        .agg(fn)
        .reset_index(level=0, drop=True)
    )


def _winsorize_series(series: pd.Series, quantile: float = 0.05) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        return pd.Series(0.0, index=series.index, dtype=float)
    q = float(np.clip(quantile, 0.0, 0.49))
    lower = float(s.quantile(q))
    upper = float(s.quantile(1.0 - q))
    if lower > upper:
        lower, upper = upper, lower
    return s.clip(lower=lower, upper=upper)


def cross_sectional_rank_zscore(series: pd.Series) -> pd.Series:
    ranked = pd.to_numeric(series, errors="coerce").rank(method="average")
    std = float(ranked.std(ddof=0)) if not ranked.dropna().empty else 0.0
    if std <= 1e-12:
        return pd.Series(0.0, index=series.index, dtype=float)
    out = (ranked - float(ranked.mean())) / std
    return out.fillna(0.0).astype(float)


def standardize_cross_sectional_signal(
    series: pd.Series,
    winsor_quantile: float = 0.05,
) -> pd.Series:
    clipped = _winsorize_series(series, quantile=winsor_quantile)
    return cross_sectional_rank_zscore(clipped).fillna(0.0).astype(float)


def normalize_signal_weights(
    weights: dict[str, float] | None,
    normalize_weights: bool = True,
) -> dict[str, float]:
    defaults = {key: 0.0 for key in SIGNAL_WEIGHT_KEYS}
    defaults["model_prediction"] = 1.0
    if weights is None:
        out = defaults.copy()
    else:
        out = {}
        for key in SIGNAL_WEIGHT_KEYS:
            out[key] = float(weights.get(key, defaults[key]))

    if not normalize_weights:
        if abs(sum(abs(out[k]) for k in SIGNAL_WEIGHT_KEYS)) <= 1e-12:
            out = defaults.copy()
        return out

    l1 = float(sum(abs(out[k]) for k in SIGNAL_WEIGHT_KEYS))
    if l1 <= 1e-12:
        return defaults.copy()
    return {k: float(v / l1) for k, v in out.items()}


def build_price_volume_signal_panel(clean_prices: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "adj_close"}
    missing = sorted(required - set(clean_prices.columns))
    if missing:
        raise ValueError(f"`clean_prices` is missing required columns: {missing}")

    if "volume" not in clean_prices.columns:
        df = clean_prices[["date", "ticker", "adj_close"]].copy()
        df["volume"] = 1.0
    else:
        df = clean_prices[["date", "ticker", "adj_close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype(str)
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    grouped = df.groupby("ticker", group_keys=False)
    df["ret_1d"] = grouped["adj_close"].pct_change(periods=1)
    df["ret_2d"] = grouped["adj_close"].pct_change(periods=2)
    df["ret_5d"] = grouped["adj_close"].pct_change(periods=5)
    df["mom_20_60"] = grouped["adj_close"].shift(20) / grouped["adj_close"].shift(60) - 1.0

    df["vol_20d"] = _rolling_group_stat(df["ret_1d"], df["ticker"], window=20, min_periods=20, fn="std")
    df["vol_60d"] = _rolling_group_stat(df["ret_1d"], df["ticker"], window=60, min_periods=60, fn="std")

    market_ret_1d = df.groupby("date", sort=True)["ret_1d"].mean().astype(float)
    market_ret_20 = (1.0 + market_ret_1d).rolling(window=20, min_periods=20).apply(np.prod, raw=True) - 1.0
    market_var_60 = market_ret_1d.rolling(window=60, min_periods=20).var(ddof=1)

    df["market_ret_1d"] = df["date"].map(market_ret_1d)
    df["market_ret_20"] = df["date"].map(market_ret_20)
    df["market_var_60"] = df["date"].map(market_var_60)

    df["ret_x_market"] = df["ret_1d"] * df["market_ret_1d"]
    mean_ret = _rolling_group_stat(df["ret_1d"], df["ticker"], window=60, min_periods=20, fn="mean")
    mean_market = _rolling_group_stat(df["market_ret_1d"], df["ticker"], window=60, min_periods=20, fn="mean")
    mean_ret_x_market = _rolling_group_stat(df["ret_x_market"], df["ticker"], window=60, min_periods=20, fn="mean")
    cov_ret_market = mean_ret_x_market - (mean_ret * mean_market)
    beta_to_market = cov_ret_market / df["market_var_60"].replace(0.0, np.nan)

    df["momentum_residual_signal"] = df["mom_20_60"] - (beta_to_market * df["market_ret_20"])

    vol_state = (df["vol_20d"] / df["vol_60d"]).replace([np.inf, -np.inf], np.nan).clip(lower=0.5, upper=1.5)
    reversal_raw = (-df["ret_1d"]) + (0.5 * -df["ret_2d"])
    df["reversal_regime_signal"] = reversal_raw * vol_state

    compression = (1.0 - (df["vol_20d"] / df["vol_60d"])).replace([np.inf, -np.inf], np.nan)
    breakout = df["ret_5d"]
    df["vol_compression_breakout_signal"] = compression * breakout

    df["dollar_volume_1d"] = df["adj_close"] * df["volume"]
    dv_mean_20 = _rolling_group_stat(df["dollar_volume_1d"], df["ticker"], window=20, min_periods=20, fn="mean")
    dv_std_60 = _rolling_group_stat(df["dollar_volume_1d"], df["ticker"], window=60, min_periods=20, fn="std")
    dv_shock = (df["dollar_volume_1d"] - dv_mean_20) / dv_std_60.replace(0.0, np.nan)
    direction = np.sign(df["ret_5d"]).replace(0.0, np.nan)
    df["liquidity_impulse_signal"] = dv_shock * direction

    out = df[["date", "ticker", *ENGINEERED_SIGNAL_COLUMNS]].copy()
    out[list(ENGINEERED_SIGNAL_COLUMNS)] = out[list(ENGINEERED_SIGNAL_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    return out


def build_composite_signal(
    model_signal: pd.Series,
    engineered_signals: pd.DataFrame,
    weights: dict[str, float] | None,
    normalize_weights: bool = True,
    winsor_quantile: float = 0.05,
) -> tuple[pd.Series, dict[str, pd.Series], dict[str, float]]:
    model_clean = (
        pd.to_numeric(model_signal, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )

    feats = engineered_signals.reindex(index=model_clean.index).copy()
    components: dict[str, pd.Series] = {
        "model_prediction": model_clean,
    }
    for col_name, key in [
        ("momentum_residual_signal", "momentum_residual"),
        ("reversal_regime_signal", "reversal_regime"),
        ("vol_compression_breakout_signal", "vol_compression_breakout"),
        ("liquidity_impulse_signal", "liquidity_impulse"),
    ]:
        raw = feats[col_name] if col_name in feats.columns else pd.Series(0.0, index=model_clean.index, dtype=float)
        components[key] = standardize_cross_sectional_signal(raw, winsor_quantile=winsor_quantile)

    resolved = normalize_signal_weights(weights=weights, normalize_weights=normalize_weights)
    composite = pd.Series(0.0, index=model_clean.index, dtype=float, name="composite_signal")
    for key in SIGNAL_WEIGHT_KEYS:
        composite = composite + (resolved[key] * components[key])
    composite = composite.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    return composite, components, resolved


def compute_signal_attribution_stats(
    components: dict[str, pd.Series],
    weights: dict[str, float],
    composite: pd.Series,
) -> dict[str, float]:
    weighted_model = components["model_prediction"] * float(weights["model_prediction"])
    weighted_momentum = components["momentum_residual"] * float(weights["momentum_residual"])
    weighted_reversal = components["reversal_regime"] * float(weights["reversal_regime"])
    weighted_vol = components["vol_compression_breakout"] * float(weights["vol_compression_breakout"])
    weighted_liq = components["liquidity_impulse"] * float(weights["liquidity_impulse"])

    return {
        "signal_model_component": float(weighted_model.abs().mean()),
        "signal_momentum_component": float(weighted_momentum.abs().mean()),
        "signal_reversal_component": float(weighted_reversal.abs().mean()),
        "signal_vol_breakout_component": float(weighted_vol.abs().mean()),
        "signal_liquidity_component": float(weighted_liq.abs().mean()),
        "signal_composite": float(composite.abs().mean()),
    }


def summarize_signal_stack_contributions(rebalance_log: pd.DataFrame) -> dict[str, dict[str, float]]:
    cols = [
        "signal_model_component",
        "signal_momentum_component",
        "signal_reversal_component",
        "signal_vol_breakout_component",
        "signal_liquidity_component",
        "signal_composite",
    ]
    out: dict[str, dict[str, float]] = {}
    for col in cols:
        if col not in rebalance_log.columns:
            continue
        series = pd.to_numeric(rebalance_log[col], errors="coerce").dropna()
        if series.empty:
            continue
        out[col] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "max": float(series.max()),
        }
    return out


def parse_signal_stack_weights(signal_stack_cfg: dict[str, Any]) -> tuple[dict[str, float], bool]:
    normalize_weights = bool(signal_stack_cfg.get("normalize_weights", True))
    weights_cfg = signal_stack_cfg.get("weights", {})
    if weights_cfg is None:
        weights_cfg = {}
    if not isinstance(weights_cfg, dict):
        raise ValueError("`backtest.signal_stack.weights` must be a mapping.")

    raw = {
        "model_prediction": float(weights_cfg.get("model_prediction", 1.0)),
        "momentum_residual": float(weights_cfg.get("momentum_residual", 0.0)),
        "reversal_regime": float(weights_cfg.get("reversal_regime", 0.0)),
        "vol_compression_breakout": float(weights_cfg.get("vol_compression_breakout", 0.0)),
        "liquidity_impulse": float(weights_cfg.get("liquidity_impulse", 0.0)),
    }
    return normalize_signal_weights(raw, normalize_weights=normalize_weights), normalize_weights
