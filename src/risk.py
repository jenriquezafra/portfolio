from __future__ import annotations

import numpy as np
import pandas as pd


def build_daily_returns(clean_prices: pd.DataFrame) -> pd.DataFrame:
    df = clean_prices[["date", "ticker", "adj_close"]].copy()
    df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(None)
    df = df.sort_values(["ticker", "date"])
    df["ret_1d"] = df.groupby("ticker", group_keys=False)["adj_close"].pct_change(periods=1)
    df = df.dropna(subset=["ret_1d"])
    return df[["date", "ticker", "ret_1d"]].reset_index(drop=True)


def pivot_returns(returns_long: pd.DataFrame) -> pd.DataFrame:
    pivot = returns_long.pivot(index="date", columns="ticker", values="ret_1d").sort_index()
    pivot.columns = pivot.columns.astype(str)
    return pivot


def estimate_covariance_matrix(
    returns_wide: pd.DataFrame,
    tickers: list[str],
    as_of_date: pd.Timestamp,
    lookback_days: int,
    shrinkage: float = 0.10,
) -> pd.DataFrame:
    if lookback_days <= 1:
        raise ValueError("`lookback_days` must be greater than 1.")
    if not (0.0 <= shrinkage <= 1.0):
        raise ValueError("`shrinkage` must be between 0.0 and 1.0.")
    if not tickers:
        raise ValueError("Cannot estimate covariance with empty ticker set.")

    as_of_date = pd.Timestamp(as_of_date).tz_localize(None)
    history = returns_wide.loc[returns_wide.index <= as_of_date].copy()
    if history.empty:
        eye = np.eye(len(tickers)) * 1e-4
        return pd.DataFrame(eye, index=tickers, columns=tickers)

    history = history.tail(lookback_days)
    x = history.reindex(columns=tickers).fillna(0.0).to_numpy(dtype=float)

    if x.shape[0] <= 1:
        eye = np.eye(len(tickers)) * 1e-4
        return pd.DataFrame(eye, index=tickers, columns=tickers)

    sample_cov = np.cov(x, rowvar=False)
    if sample_cov.ndim == 0:
        sample_cov = np.array([[float(sample_cov)]], dtype=float)

    diag = np.diag(np.diag(sample_cov))
    shrunk = (1.0 - shrinkage) * sample_cov + shrinkage * diag
    shrunk = np.nan_to_num(shrunk, nan=0.0, posinf=0.0, neginf=0.0)
    shrunk = shrunk + np.eye(len(tickers)) * 1e-8
    return pd.DataFrame(shrunk, index=tickers, columns=tickers)
