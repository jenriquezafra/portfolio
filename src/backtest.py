from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import load_yaml
from src.optimizer import (
    apply_turnover_cap,
    factor_neutralize_market_neutral_weights,
    optimize_mean_variance_long_only,
    signal_to_long_only_weights,
    signal_to_market_neutral_weights,
)
from src.reporting import build_factor_diagnostics_report
from src.risk import build_daily_returns, estimate_covariance_matrix, pivot_returns


def _get_rebalance_dates(
    available_dates: pd.Series,
    frequency: str,
    every_n_days: int | None = None,
) -> list[pd.Timestamp]:
    dates = pd.Series(pd.to_datetime(available_dates).sort_values().unique())
    freq = frequency.lower()

    if freq == "daily":
        return dates.tolist()
    if freq == "weekly":
        return dates.groupby(dates.dt.to_period("W-FRI")).max().sort_values().tolist()
    if freq == "monthly":
        return dates.groupby(dates.dt.to_period("M")).max().sort_values().tolist()
    if freq == "every_n_days":
        if every_n_days is None or every_n_days <= 0:
            raise ValueError("`rebalance_every_n_days` must be positive for `every_n_days` frequency.")
        return dates.iloc[every_n_days - 1 :: every_n_days].tolist()

    raise ValueError(f"Unsupported rebalance frequency: {frequency}")


def _spearman_rank_corr(lhs: pd.Series | np.ndarray, rhs: pd.Series | np.ndarray) -> float | None:
    left = pd.Series(lhs).rank(method="average")
    right = pd.Series(rhs).rank(method="average")
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return None
    corr = left.corr(right, method="pearson")
    if pd.isna(corr):
        return None
    return float(corr)


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


def _compute_summary(
    daily_returns: pd.DataFrame,
    rebalance_log: pd.DataFrame,
    portfolio_mode: str,
    allocation_method: str,
    use_rank_zscore: bool,
    beta_neutralization_enabled: bool,
) -> dict[str, Any]:
    if daily_returns.empty:
        raise ValueError("Backtest produced no daily returns.")

    ret = daily_returns["portfolio_return_net"].astype(float)
    equity = (1.0 + ret).cumprod()
    n_days = len(ret)

    total_return = float(equity.iloc[-1] - 1.0)
    ann_return = float((equity.iloc[-1] ** (252.0 / n_days)) - 1.0) if n_days > 0 else 0.0
    ann_vol = float(ret.std(ddof=1) * np.sqrt(252.0)) if n_days > 1 else 0.0
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else None

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    max_drawdown = float(drawdown.min())

    weekly_returns = (
        daily_returns.set_index("date")["portfolio_return_net"]
        .resample("W-FRI")
        .apply(lambda x: float((1.0 + x).prod() - 1.0))
    )
    weekly_stats = _annualized_stats(weekly_returns, periods_per_year=52.0)

    summary: dict[str, Any] = {
        "start_date": str(daily_returns["date"].min().date()),
        "end_date": str(daily_returns["date"].max().date()),
        "n_days": int(n_days),
        "n_rebalances": int(rebalance_log["rebalance_date"].nunique()),
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "weekly_sharpe_ratio": weekly_stats["sharpe_ratio"],
        "max_drawdown": max_drawdown,
        "average_turnover": float(rebalance_log["turnover"].mean()) if not rebalance_log.empty else 0.0,
        "total_cost_bps_paid": float(rebalance_log["trade_cost_bps_paid"].sum()) if not rebalance_log.empty else 0.0,
        "average_gross_exposure": float(rebalance_log["gross_exposure"].mean()) if not rebalance_log.empty else 0.0,
        "average_net_exposure": float(rebalance_log["net_exposure"].mean()) if not rebalance_log.empty else 0.0,
        "portfolio_mode": portfolio_mode,
        "allocation_method": allocation_method,
        "signal_rank_zscore": bool(use_rank_zscore),
        "beta_neutralization_enabled": bool(beta_neutralization_enabled),
    }
    factor_cols = [col for col in rebalance_log.columns if col.startswith("ex_ante_factor_")]
    if factor_cols:
        mean_map: dict[str, float] = {}
        max_abs_map: dict[str, float] = {}
        for col in factor_cols:
            name = col.replace("ex_ante_factor_", "", 1)
            series = pd.to_numeric(rebalance_log[col], errors="coerce").dropna()
            if series.empty:
                continue
            mean_map[name] = float(series.mean())
            max_abs_map[name] = float(series.abs().max())
        if mean_map:
            summary["average_ex_ante_factor_exposure"] = mean_map
            summary["max_abs_ex_ante_factor_exposure"] = max_abs_map
            if len(mean_map) == 1:
                only_name = next(iter(mean_map.keys()))
                summary["average_ex_ante_beta_to_factor"] = float(mean_map[only_name])
                summary["max_abs_ex_ante_beta_to_factor"] = float(max_abs_map[only_name])
    return summary


def _rank_to_zscore(signal: pd.Series) -> pd.Series:
    ranked = signal.rank(method="average")
    std = float(ranked.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(0.0, index=signal.index, dtype=float)
    return ((ranked - float(ranked.mean())) / std).astype(float)


def _estimate_asset_volatility(
    returns_wide: pd.DataFrame,
    tickers: list[str],
    as_of_date: pd.Timestamp,
    lookback_days: int,
) -> pd.Series:
    history = returns_wide[returns_wide.index <= as_of_date]
    history = history.tail(max(2, int(lookback_days)))
    if history.empty:
        return pd.Series(1.0, index=tickers, dtype=float)

    vol = history[tickers].std(ddof=1).replace([np.inf, -np.inf], np.nan)
    median = float(vol[vol > 0].median()) if (vol > 0).any() else 1.0
    vol = vol.where(vol > 1e-12, median if median > 0 else 1.0).fillna(median if median > 0 else 1.0)
    return vol.astype(float)


def _build_universe_growth_proxy_returns(
    clean_prices: pd.DataFrame,
    quantile: float = 0.30,
) -> pd.Series:
    if not (0.0 < quantile <= 0.5):
        raise ValueError("`quantile` for growth proxy must be in (0, 0.5].")

    df = clean_prices[["date", "ticker", "adj_close"]].copy()
    df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(None)
    df = df.sort_values(["ticker", "date"])
    grouped = df.groupby("ticker", group_keys=False)
    df["ret_1d"] = grouped["adj_close"].pct_change(periods=1)
    momentum = grouped["adj_close"].shift(20) / grouped["adj_close"].shift(60) - 1.0
    # Use lagged score for same-day return to avoid look-ahead in factor construction.
    df["score"] = momentum.groupby(df["ticker"]).shift(1)
    df = df.dropna(subset=["ret_1d", "score"])

    records: list[tuple[pd.Timestamp, float]] = []
    for dt, frame in df.groupby("date"):
        n = len(frame)
        bucket = int(np.floor(n * quantile))
        if bucket < 1:
            continue
        ranked = frame.sort_values("score")
        low = float(ranked.head(bucket)["ret_1d"].mean())
        high = float(ranked.tail(bucket)["ret_1d"].mean())
        records.append((pd.Timestamp(dt), high - low))

    if not records:
        return pd.Series(dtype=float, name="growth_proxy")
    out = pd.DataFrame(records, columns=["date", "growth_proxy"]).dropna()
    return out.set_index("date")["growth_proxy"].astype(float)


def _read_factor_file(
    project_root: Path,
    factor_path_cfg: str,
    factor_col: str,
) -> pd.Series:
    factor_path = Path(factor_path_cfg)
    if not factor_path.is_absolute():
        factor_path = (project_root / factor_path).resolve()
    if factor_path.suffix.lower() == ".parquet":
        factor_df = pd.read_parquet(factor_path)
    else:
        factor_df = pd.read_csv(factor_path)
    if "date" not in factor_df.columns or factor_col not in factor_df.columns:
        raise ValueError(f"Factor file must include columns `date` and `{factor_col}`.")
    series = (
        factor_df.assign(date=pd.to_datetime(factor_df["date"], utc=False).dt.tz_localize(None))
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .set_index("date")[factor_col]
        .astype(float)
        .dropna()
    )
    return series


def _load_single_factor_returns(
    returns_wide: pd.DataFrame,
    clean_prices: pd.DataFrame,
    project_root: Path,
    factor_cfg: dict[str, Any],
) -> pd.Series:
    source = str(factor_cfg.get("source", factor_cfg.get("factor_source", "universe_equal_weight"))).lower()
    if source == "universe_equal_weight":
        return returns_wide.mean(axis=1, skipna=True).dropna()
    if source == "ticker":
        ticker = str(factor_cfg.get("ticker", factor_cfg.get("factor_ticker", ""))).upper()
        if not ticker:
            raise ValueError("`ticker` is required when factor source is `ticker`.")
        if ticker not in returns_wide.columns:
            raise ValueError(
                f"Ticker factor `{ticker}` not found in returns history. "
                "Use source=`file` or include ticker in universe."
            )
        return returns_wide[ticker].dropna().astype(float)
    if source == "ticker_spread":
        long_ticker = str(factor_cfg.get("long_ticker", "")).upper()
        short_ticker = str(factor_cfg.get("short_ticker", "")).upper()
        if not long_ticker or not short_ticker:
            raise ValueError("`long_ticker` and `short_ticker` are required for `ticker_spread`.")
        if long_ticker not in returns_wide.columns or short_ticker not in returns_wide.columns:
            raise ValueError("Spread tickers not found in returns history.")
        scale = float(factor_cfg.get("scale", 1.0))
        return (returns_wide[long_ticker] - returns_wide[short_ticker]).astype(float) * scale
    if source == "universe_growth_proxy":
        quantile = float(factor_cfg.get("quantile", 0.30))
        return _build_universe_growth_proxy_returns(clean_prices=clean_prices, quantile=quantile)
    if source == "file":
        factor_path_cfg = factor_cfg.get("path", factor_cfg.get("factor_path"))
        if not isinstance(factor_path_cfg, str):
            raise ValueError("`path` must be provided when factor source is `file`.")
        factor_col = str(factor_cfg.get("return_column", factor_cfg.get("factor_return_column", "return")))
        return _read_factor_file(
            project_root=project_root,
            factor_path_cfg=factor_path_cfg,
            factor_col=factor_col,
        )
    raise ValueError(
        "Unsupported factor source. Use one of: universe_equal_weight, ticker, ticker_spread, "
        "universe_growth_proxy, file."
    )


def _load_factor_returns_frame(
    returns_wide: pd.DataFrame,
    clean_prices: pd.DataFrame,
    project_root: Path,
    beta_cfg: dict[str, Any],
) -> pd.DataFrame:
    raw_factors = beta_cfg.get("factors")
    factors: list[dict[str, Any]]
    if isinstance(raw_factors, list) and raw_factors:
        factors = []
        for item in raw_factors:
            if not isinstance(item, dict):
                raise ValueError("Each entry in `beta_neutralization.factors` must be a mapping.")
            factors.append(item)
    else:
        # Backward-compatible single factor config.
        factors = [
            {
                "name": str(beta_cfg.get("factor_name", "factor")),
                "source": beta_cfg.get("factor_source", "universe_equal_weight"),
                "ticker": beta_cfg.get("factor_ticker"),
                "path": beta_cfg.get("factor_path"),
                "return_column": beta_cfg.get("factor_return_column", "return"),
            }
        ]

    out: dict[str, pd.Series] = {}
    for i, factor_cfg in enumerate(factors, start=1):
        name = str(factor_cfg.get("name", f"factor_{i}")).strip()
        if not name:
            name = f"factor_{i}"
        series = _load_single_factor_returns(
            returns_wide=returns_wide,
            clean_prices=clean_prices,
            project_root=project_root,
            factor_cfg=factor_cfg,
        )
        out[name] = series.rename(name)

    frame = pd.concat(out.values(), axis=1, join="outer").sort_index()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return frame


def _estimate_asset_factor_exposures(
    returns_wide: pd.DataFrame,
    factor_returns: pd.DataFrame,
    tickers: list[str],
    as_of_date: pd.Timestamp,
    lookback_days: int,
) -> pd.DataFrame:
    history = returns_wide[returns_wide.index <= as_of_date][tickers].tail(max(20, int(lookback_days)))
    factor_hist = factor_returns[factor_returns.index <= as_of_date].tail(max(20, int(lookback_days)))
    common_idx = history.index.intersection(factor_hist.index)
    if len(common_idx) < 20:
        return pd.DataFrame(0.0, index=tickers, columns=factor_returns.columns, dtype=float)

    factors_aligned = factor_hist.reindex(common_idx).astype(float)
    k = len(factors_aligned.columns)
    min_obs = max(20, k + 5)

    exposures = pd.DataFrame(0.0, index=tickers, columns=factors_aligned.columns, dtype=float)
    for ticker in tickers:
        asset = history[ticker].reindex(common_idx).astype(float)
        pair = pd.concat([asset.rename("asset"), factors_aligned], axis=1).dropna()
        if len(pair) < min_obs:
            continue
        y = pair["asset"].to_numpy(dtype=float)
        x = pair[factors_aligned.columns].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(pair)), x])
        try:
            coef = np.linalg.lstsq(design, y, rcond=None)[0]
            exposures.loc[ticker, factors_aligned.columns] = coef[1:]
        except np.linalg.LinAlgError:
            continue
    return exposures


def _beta_anchor_key(rebalance_date: pd.Timestamp, frequency: str) -> str:
    freq = frequency.lower()
    if freq == "rebalance":
        return str(pd.Timestamp(rebalance_date).date())
    if freq == "weekly":
        return str(pd.Timestamp(rebalance_date).to_period("W-FRI"))
    raise ValueError("`beta_neutralization.frequency` must be `rebalance` or `weekly`.")


def _resolve_target_factor_exposures(
    beta_cfg: dict[str, Any],
    factor_names: list[str],
) -> pd.Series:
    target = pd.Series(0.0, index=factor_names, dtype=float)
    target_map = beta_cfg.get("target_factor_exposures")
    if isinstance(target_map, dict):
        mapped = pd.Series(target_map, dtype=float).reindex(factor_names).fillna(0.0)
        target.loc[mapped.index] = mapped.values
        return target

    raw_factors = beta_cfg.get("factors")
    if isinstance(raw_factors, list):
        for factor in raw_factors:
            if not isinstance(factor, dict):
                continue
            name = str(factor.get("name", "")).strip()
            if name and name in target.index and factor.get("target_exposure") is not None:
                target.loc[name] = float(factor.get("target_exposure"))
        return target

    if len(factor_names) == 1:
        target.loc[factor_names[0]] = float(beta_cfg.get("target_beta", 0.0))
    return target


def _fit_factor_model(
    returns_series: pd.Series,
    factor_frame: pd.DataFrame,
) -> dict[str, Any]:
    factors = factor_frame.columns.tolist()
    merged = pd.concat([returns_series.rename("y"), factor_frame], axis=1).dropna()
    k = len(factors)
    min_obs = max(20, k + 5)
    if len(merged) < min_obs:
        return {
            "n_obs": int(len(merged)),
            "alpha_daily": None,
            "r2": None,
            "betas": {name: None for name in factors},
        }

    y = merged["y"].to_numpy(dtype=float)
    x = merged[factors].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(merged)), x])
    try:
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {
            "n_obs": int(len(merged)),
            "alpha_daily": None,
            "r2": None,
            "betas": {name: None for name in factors},
        }

    yhat = design @ coef
    resid = y - yhat
    var_y = float(np.var(y, ddof=1)) if len(y) > 1 else 0.0
    var_e = float(np.var(resid, ddof=1)) if len(y) > 1 else 0.0
    r2 = None
    if var_y > 0:
        r2_val = 1.0 - (var_e / var_y)
        r2 = float(r2_val)
    betas = {name: float(coef[i + 1]) for i, name in enumerate(factors)}
    return {
        "n_obs": int(len(merged)),
        "alpha_daily": float(coef[0]),
        "r2": r2,
        "betas": betas,
    }


def _compute_factor_exposure_report(
    daily_returns: pd.DataFrame,
    rebalance_log: pd.DataFrame,
    factor_returns: pd.DataFrame | None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "factor_names": [],
        "ex_ante": {"mean": {}, "max_abs": {}},
        "ex_post": {"full_sample": {}, "by_year": []},
    }
    if factor_returns is None or factor_returns.empty or daily_returns.empty:
        return report

    factor_names = factor_returns.columns.tolist()
    report["factor_names"] = factor_names

    factor_cols = [f"ex_ante_factor_{name}" for name in factor_names]
    for name, col in zip(factor_names, factor_cols):
        if col not in rebalance_log.columns:
            continue
        series = pd.to_numeric(rebalance_log[col], errors="coerce").dropna()
        if series.empty:
            continue
        report["ex_ante"]["mean"][name] = float(series.mean())
        report["ex_ante"]["max_abs"][name] = float(series.abs().max())

    merged = daily_returns[["date", "portfolio_return_net"]].copy()
    merged["date"] = pd.to_datetime(merged["date"], utc=False).dt.tz_localize(None)
    factors = factor_returns.copy()
    factor_idx = pd.to_datetime(factors.index, utc=False)
    if getattr(factor_idx, "tz", None) is not None:
        factor_idx = factor_idx.tz_localize(None)
    factors.index = factor_idx
    merged = merged.merge(factors.reset_index().rename(columns={"index": "date"}), on="date", how="inner")
    if merged.empty:
        return report

    full = _fit_factor_model(
        returns_series=merged["portfolio_return_net"].astype(float),
        factor_frame=merged[factor_names].astype(float),
    )
    report["ex_post"]["full_sample"] = full

    merged["year"] = pd.to_datetime(merged["date"]).dt.year
    by_year: list[dict[str, Any]] = []
    for year, frame in merged.groupby("year"):
        res = _fit_factor_model(
            returns_series=frame["portfolio_return_net"].astype(float),
            factor_frame=frame[factor_names].astype(float),
        )
        row = {"year": int(year), **res}
        by_year.append(row)
    report["ex_post"]["by_year"] = by_year
    return report


def _build_subperiod_report(
    daily_returns: pd.DataFrame,
    predictions: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    if daily_returns.empty:
        return {"returns_by_year": [], "returns_three_blocks": [], "ic_by_year": []}

    returns = daily_returns.copy()
    returns["year"] = pd.to_datetime(returns["date"]).dt.year
    by_year: list[dict[str, Any]] = []
    for year, frame in returns.groupby("year"):
        stats = _annualized_stats(frame["portfolio_return_net"], periods_per_year=252.0)
        by_year.append(
            {
                "year": int(year),
                "n_days": int(stats["n_obs"]),
                "total_return": stats["total_return"],
                "annualized_return": stats["annualized_return"],
                "annualized_volatility": stats["annualized_volatility"],
                "sharpe_ratio": stats["sharpe_ratio"],
            }
        )

    unique_dates = pd.Series(pd.to_datetime(returns["date"]).sort_values().unique())
    split_dates = [part.tolist() for part in np.array_split(unique_dates.to_numpy(), 3) if len(part) > 0]
    by_blocks: list[dict[str, Any]] = []
    for idx, block in enumerate(split_dates, start=1):
        block_dates = pd.to_datetime(block)
        mask = returns["date"].isin(block_dates)
        frame = returns.loc[mask]
        stats = _annualized_stats(frame["portfolio_return_net"], periods_per_year=252.0)
        by_blocks.append(
            {
                "block": idx,
                "start_date": str(pd.Timestamp(block_dates.min()).date()),
                "end_date": str(pd.Timestamp(block_dates.max()).date()),
                "n_days": int(stats["n_obs"]),
                "annualized_return": stats["annualized_return"],
                "annualized_volatility": stats["annualized_volatility"],
                "sharpe_ratio": stats["sharpe_ratio"],
            }
        )

    ic_by_year: list[dict[str, Any]] = []
    if target_column in predictions.columns:
        cs_ic_rows: list[tuple[pd.Timestamp, float]] = []
        for dt, frame in predictions.groupby("date"):
            ic = _spearman_rank_corr(frame["prediction"], frame[target_column])
            if ic is not None and np.isfinite(ic):
                cs_ic_rows.append((pd.Timestamp(dt), float(ic)))
        if cs_ic_rows:
            cs_ic_df = pd.DataFrame(cs_ic_rows, columns=["date", "cs_ic"])
            cs_ic_df["year"] = cs_ic_df["date"].dt.year
            for year, frame in cs_ic_df.groupby("year"):
                ic_mean = float(frame["cs_ic"].mean())
                ic_std = float(frame["cs_ic"].std(ddof=1)) if len(frame) > 1 else None
                ic_ir = float(ic_mean / ic_std) if ic_std is not None and ic_std > 0 else None
                ic_by_year.append(
                    {
                        "year": int(year),
                        "n_rebalances": int(len(frame)),
                        "ic_mean": ic_mean,
                        "ic_median": float(frame["cs_ic"].median()),
                        "ic_std": ic_std,
                        "ic_ir": ic_ir,
                        "ic_positive_rate": float((frame["cs_ic"] > 0).mean()),
                    }
                )

    return {
        "returns_by_year": by_year,
        "returns_three_blocks": by_blocks,
        "ic_by_year": ic_by_year,
    }


def run_backtest(
    config_data_path: Path,
    config_backtest_path: Path,
    config_execution_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], Path, Path, Path, Path]:
    data_cfg = load_yaml(config_data_path)
    back_cfg = load_yaml(config_backtest_path)
    exec_cfg = load_yaml(config_execution_path)

    data_section = data_cfg.get("data")
    labels_section = data_cfg.get("labels", {})
    backtest_section = back_cfg.get("backtest")
    risk_controls = exec_cfg.get("risk_controls", {})

    if not isinstance(data_section, dict):
        raise ValueError("Missing `data` section in config_data.yaml")
    if not isinstance(backtest_section, dict):
        raise ValueError("Missing `backtest` section in config_backtest.yaml")
    if not isinstance(risk_controls, dict):
        raise ValueError("Missing `risk_controls` section in config_execution.yaml")

    clean_rel = data_section.get("output_clean_path")
    target_column = labels_section.get("target_column", "fwd_return_5d")

    rebalance_frequency = backtest_section.get("rebalance_frequency", "monthly")
    rebalance_every_n_days = backtest_section.get("rebalance_every_n_days")
    risk_lookback_days = backtest_section.get("risk_lookback_days", 60)
    risk_shrinkage = backtest_section.get("risk_shrinkage", 0.10)

    signal_cfg = backtest_section.get("signal_transform", {})
    portfolio_cfg = backtest_section.get("portfolio", {})
    beta_neut_cfg = portfolio_cfg.get("beta_neutralization", {})
    costs_cfg = backtest_section.get("costs", {})
    constraints_cfg = backtest_section.get("constraints", {})
    objective_cfg = backtest_section.get("objective", {})

    if not isinstance(signal_cfg, dict):
        raise ValueError("`backtest.signal_transform` must be a mapping.")
    if not isinstance(portfolio_cfg, dict):
        raise ValueError("`backtest.portfolio` must be a mapping.")
    if not isinstance(beta_neut_cfg, dict):
        raise ValueError("`backtest.portfolio.beta_neutralization` must be a mapping.")

    long_only_default = bool(constraints_cfg.get("long_only", True))
    portfolio_mode = str(portfolio_cfg.get("mode", "long_only" if long_only_default else "market_neutral")).lower()
    allocation_method = str(objective_cfg.get("allocation_method", "mean_variance")).lower()
    use_rank_zscore = bool(signal_cfg.get("cross_sectional_rank_zscore", False))
    long_quantile = float(portfolio_cfg.get("long_quantile", 0.20))
    short_quantile = float(portfolio_cfg.get("short_quantile", 0.20))
    gross_exposure_target = float(portfolio_cfg.get("gross_exposure_target", 1.0))
    vol_lookback_days = int(portfolio_cfg.get("vol_lookback_days", risk_lookback_days))

    risk_aversion_lambda = objective_cfg.get("risk_aversion_lambda", 10.0)
    turnover_penalty_eta = objective_cfg.get("turnover_penalty_eta", 5.0)
    fully_invested = constraints_cfg.get("fully_invested", True)
    weight_max = constraints_cfg.get("weight_max", 0.20)

    beta_neutralization_enabled = bool(beta_neut_cfg.get("enabled", False))
    beta_frequency = str(beta_neut_cfg.get("frequency", "weekly")).lower()
    beta_lookback_days = int(beta_neut_cfg.get("lookback_days", 126))

    bps_per_side = costs_cfg.get("bps_per_side", 5.0)
    slippage_bps = costs_cfg.get("slippage_bps", 2.0)
    total_cost_bps = float(bps_per_side) + float(slippage_bps)

    max_turnover = risk_controls.get("max_turnover_per_rebalance")
    if max_turnover is not None:
        max_turnover = float(max_turnover)

    if portfolio_mode not in {"long_only", "market_neutral"}:
        raise ValueError("`backtest.portfolio.mode` must be one of: long_only, market_neutral.")
    if allocation_method not in {"mean_variance", "score_over_vol"}:
        raise ValueError("`backtest.objective.allocation_method` must be one of: mean_variance, score_over_vol.")
    if allocation_method == "mean_variance" and portfolio_mode != "long_only":
        raise ValueError("mean_variance currently supports only `portfolio.mode=long_only`.")
    if beta_neutralization_enabled and portfolio_mode != "market_neutral":
        raise ValueError("beta neutralization is currently supported only in `market_neutral` mode.")
    if not isinstance(clean_rel, str):
        raise ValueError("`data.output_clean_path` must be a string path.")
    if not isinstance(target_column, str):
        raise ValueError("`labels.target_column` must be a string.")
    if not isinstance(rebalance_frequency, str):
        raise ValueError("`backtest.rebalance_frequency` must be a string.")
    if rebalance_every_n_days is not None and not isinstance(rebalance_every_n_days, int):
        raise ValueError("`backtest.rebalance_every_n_days` must be null or integer.")

    project_root = config_data_path.parents[1]
    clean_path = (project_root / clean_rel).resolve()
    predictions_path = (project_root / "outputs/models/predictions_oos.parquet").resolve()

    clean_prices = pd.read_parquet(clean_path)
    predictions = pd.read_parquet(predictions_path)
    predictions["date"] = pd.to_datetime(predictions["date"], utc=False).dt.tz_localize(None)
    predictions["ticker"] = predictions["ticker"].astype(str)

    returns_wide = pivot_returns(build_daily_returns(clean_prices))
    returns_dates = pd.Series(returns_wide.index.to_list())

    factor_returns: pd.DataFrame | None = None
    factor_target_exposures: pd.Series | None = None
    if beta_neutralization_enabled:
        factor_returns = _load_factor_returns_frame(
            returns_wide=returns_wide,
            clean_prices=clean_prices,
            project_root=project_root,
            beta_cfg=beta_neut_cfg,
        )
        if factor_returns.empty:
            raise ValueError("Factor returns are empty. Check beta_neutralization factor configuration.")
        factor_target_exposures = _resolve_target_factor_exposures(
            beta_cfg=beta_neut_cfg,
            factor_names=factor_returns.columns.tolist(),
        )

    available_prediction_dates = pd.Series(predictions["date"].sort_values().unique())
    pred_date_list = available_prediction_dates.tolist()
    if rebalance_frequency.lower() == "every_n_days" and rebalance_every_n_days is not None:
        # If predictions are already sparse (e.g., model was trained on the same cadence), keep them as-is.
        gaps = available_prediction_dates.diff().dropna().dt.days
        median_gap = float(gaps.median()) if not gaps.empty else 0.0
        if median_gap >= max(1.0, float(rebalance_every_n_days) - 1.0):
            rebalance_dates = pred_date_list
        else:
            rebalance_dates = _get_rebalance_dates(
                available_dates=available_prediction_dates,
                frequency=rebalance_frequency,
                every_n_days=rebalance_every_n_days,
            )
    else:
        rebalance_dates = _get_rebalance_dates(
            available_dates=available_prediction_dates,
            frequency=rebalance_frequency,
            every_n_days=rebalance_every_n_days,
        )

    if not rebalance_dates:
        raise ValueError("No rebalance dates available after applying frequency filter.")

    all_tickers = sorted(clean_prices["ticker"].astype(str).unique().tolist())
    prev_weights = pd.Series(0.0, index=all_tickers, name="weight")
    weights_records: list[dict[str, Any]] = []
    rebalance_logs: list[dict[str, Any]] = []
    daily_records: list[dict[str, Any]] = []

    cached_beta_anchor: str | None = None
    cached_factor_exposures: pd.DataFrame | None = None

    for idx, rebalance_date in enumerate(rebalance_dates):
        pred_slice = predictions[predictions["date"] == rebalance_date].copy()
        if pred_slice.empty:
            continue

        pred_slice = pred_slice.sort_values("ticker")
        pred_slice = pred_slice[pred_slice["ticker"].isin(returns_wide.columns)]
        if pred_slice.empty:
            continue

        tickers = pred_slice["ticker"].tolist()
        mu_raw = pred_slice.set_index("ticker")["prediction"].astype(float)
        signal = _rank_to_zscore(mu_raw) if use_rank_zscore else mu_raw.copy()
        vol_est = _estimate_asset_volatility(
            returns_wide=returns_wide,
            tickers=tickers,
            as_of_date=pd.Timestamp(rebalance_date),
            lookback_days=vol_lookback_days,
        )

        prev_sub = prev_weights.reindex(tickers).fillna(0.0)
        ex_ante_factor_map: dict[str, float] = {}

        if allocation_method == "mean_variance":
            cov_daily = estimate_covariance_matrix(
                returns_wide=returns_wide,
                tickers=tickers,
                as_of_date=rebalance_date,
                lookback_days=int(risk_lookback_days),
                shrinkage=float(risk_shrinkage),
            )
            # Predictions are H-day forward returns; scale daily covariance to the same horizon.
            horizon_cov = cov_daily * float(max(1, int(data_cfg.get("labels", {}).get("horizon_days", 5))))
            target = optimize_mean_variance_long_only(
                expected_returns=signal,
                covariance=horizon_cov,
                prev_weights=prev_sub,
                risk_aversion_lambda=float(risk_aversion_lambda),
                turnover_penalty_eta=float(turnover_penalty_eta),
                weight_max=float(weight_max),
                fully_invested=bool(fully_invested),
            )
        else:
            if portfolio_mode == "long_only":
                target = signal_to_long_only_weights(
                    signal=signal,
                    volatility=vol_est,
                    weight_max=float(weight_max),
                    fully_invested=bool(fully_invested),
                )
            else:
                target = signal_to_market_neutral_weights(
                    signal=signal,
                    volatility=vol_est,
                    weight_max_abs=float(weight_max),
                    gross_exposure_target=float(gross_exposure_target),
                    long_quantile=float(long_quantile),
                    short_quantile=float(short_quantile),
                )
                if beta_neutralization_enabled and factor_returns is not None:
                    anchor = _beta_anchor_key(pd.Timestamp(rebalance_date), beta_frequency)
                    cache_missing_tickers = cached_factor_exposures is None or not set(tickers).issubset(
                        set(cached_factor_exposures.index)
                    )
                    if cached_factor_exposures is None or anchor != cached_beta_anchor or cache_missing_tickers:
                        cached_factor_exposures = _estimate_asset_factor_exposures(
                            returns_wide=returns_wide,
                            factor_returns=factor_returns,
                            tickers=tickers,
                            as_of_date=pd.Timestamp(rebalance_date),
                            lookback_days=beta_lookback_days,
                        )
                        cached_beta_anchor = anchor

                    factor_exposures = cached_factor_exposures.reindex(index=tickers).fillna(0.0)
                    assert factor_target_exposures is not None
                    target = factor_neutralize_market_neutral_weights(
                        weights=target,
                        asset_factor_exposures=factor_exposures,
                        target_factor_exposures=factor_target_exposures.to_dict(),
                        gross_exposure_target=float(gross_exposure_target),
                        weight_max_abs=float(weight_max),
                    )

        final_w, turnover = apply_turnover_cap(
            target_weights=target,
            prev_weights=prev_sub,
            max_turnover_per_rebalance=max_turnover,
        )
        if beta_neutralization_enabled and factor_returns is not None and cached_factor_exposures is not None:
            factor_exposures = cached_factor_exposures.reindex(index=tickers).fillna(0.0)
            ex_ante = factor_exposures.T.dot(final_w.reindex(tickers).fillna(0.0))
            ex_ante_factor_map = {str(name): float(val) for name, val in ex_ante.items()}

        trade_cost = turnover * total_cost_bps / 10000.0

        next_rebalance = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else returns_dates.max()
        hold_dates = returns_dates[(returns_dates > rebalance_date) & (returns_dates <= next_rebalance)]
        if len(hold_dates) == 0:
            continue

        returns_slice = returns_wide.loc[hold_dates, tickers].fillna(0.0)
        gross_series = returns_slice.dot(final_w.reindex(tickers).fillna(0.0))

        net_series = gross_series.copy()
        first_day = hold_dates.iloc[0]
        net_series.loc[first_day] = net_series.loc[first_day] - trade_cost

        for d in hold_dates:
            daily_records.append(
                {
                    "date": pd.Timestamp(d),
                    "rebalance_date": pd.Timestamp(rebalance_date),
                    "portfolio_return_gross": float(gross_series.loc[d]),
                    "portfolio_return_net": float(net_series.loc[d]),
                }
            )

        for t in tickers:
            weights_records.append(
                {
                    "rebalance_date": pd.Timestamp(rebalance_date),
                    "ticker": t,
                    "weight": float(final_w.loc[t]),
                    "predicted_return": float(mu_raw.loc[t]),
                    "signal_score": float(signal.loc[t]),
                }
            )

        log_row: dict[str, Any] = {
            "rebalance_date": pd.Timestamp(rebalance_date),
            "n_assets": int(len(tickers)),
            "n_longs": int((final_w > 0).sum()),
            "n_shorts": int((final_w < 0).sum()),
            "gross_exposure": float(final_w.abs().sum()),
            "net_exposure": float(final_w.sum()),
            "turnover": float(turnover),
            "trade_cost_bps_paid": float(turnover * total_cost_bps),
            "trade_cost_return": float(trade_cost),
        }
        for name, value in ex_ante_factor_map.items():
            log_row[f"ex_ante_factor_{name}"] = float(value)
        if len(ex_ante_factor_map) == 1:
            only_value = float(next(iter(ex_ante_factor_map.values())))
            log_row["ex_ante_beta_to_factor"] = only_value
        rebalance_logs.append(log_row)

        prev_weights = pd.Series(0.0, index=all_tickers, name="weight")
        prev_weights.loc[final_w.index] = final_w.values

    daily_returns = pd.DataFrame(daily_records).sort_values("date").reset_index(drop=True)
    weights_history = pd.DataFrame(weights_records).sort_values(["rebalance_date", "ticker"]).reset_index(drop=True)
    rebalance_log = pd.DataFrame(rebalance_logs).sort_values("rebalance_date").reset_index(drop=True)
    summary = _compute_summary(
        daily_returns=daily_returns,
        rebalance_log=rebalance_log,
        portfolio_mode=portfolio_mode,
        allocation_method=allocation_method,
        use_rank_zscore=use_rank_zscore,
        beta_neutralization_enabled=beta_neutralization_enabled,
    )
    subperiod_report = _build_subperiod_report(
        daily_returns=daily_returns,
        predictions=predictions,
        target_column=target_column,
    )
    factor_exposure_report = _compute_factor_exposure_report(
        daily_returns=daily_returns,
        rebalance_log=rebalance_log,
        factor_returns=factor_returns,
    )
    factor_diagnostics_report = build_factor_diagnostics_report(
        backtest_summary=summary,
        factor_exposure_report=factor_exposure_report,
    )
    full_factor_model = factor_exposure_report.get("ex_post", {}).get("full_sample", {})
    if isinstance(full_factor_model, dict):
        betas = full_factor_model.get("betas")
        if isinstance(betas, dict) and betas:
            summary["ex_post_factor_betas"] = betas
            summary["ex_post_factor_alpha_daily"] = full_factor_model.get("alpha_daily")
            summary["ex_post_factor_r2"] = full_factor_model.get("r2")

    out_dir = (project_root / "outputs/backtests").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_path = out_dir / "daily_returns.parquet"
    weights_path = out_dir / "weights_history.parquet"
    rebalance_log_path = out_dir / "rebalance_log.parquet"
    summary_path = out_dir / "backtest_summary.json"
    subperiod_path = out_dir / "subperiod_report.json"
    factor_exposure_path = out_dir / "factor_exposure_report.json"
    factor_diagnostics_path = out_dir / "factor_diagnostics_report.json"

    daily_returns.to_parquet(daily_path, index=False)
    weights_history.to_parquet(weights_path, index=False)
    rebalance_log.to_parquet(rebalance_log_path, index=False)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    with subperiod_path.open("w", encoding="utf-8") as fh:
        json.dump(subperiod_report, fh, indent=2, sort_keys=True)
    with factor_exposure_path.open("w", encoding="utf-8") as fh:
        json.dump(factor_exposure_report, fh, indent=2, sort_keys=True)
    with factor_diagnostics_path.open("w", encoding="utf-8") as fh:
        json.dump(factor_diagnostics_report, fh, indent=2, sort_keys=True)

    return daily_returns, weights_history, rebalance_log, summary, daily_path, weights_path, rebalance_log_path, summary_path
