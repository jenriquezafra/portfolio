from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data import load_yaml

REQUIRED_RAW_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
FEATURE_COLUMNS = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "momentum_20_60"]


def _validate_raw_schema(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Raw prices parquet is missing required columns: {missing}")


def clean_prices(
    raw_prices: pd.DataFrame,
    min_history_days: int = 252,
    drop_rows_without_adj_close: bool = True,
) -> pd.DataFrame:
    _validate_raw_schema(raw_prices)

    df = raw_prices.copy()
    df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype(str).str.upper()

    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "ticker"])
    df = df.sort_values(["ticker", "date"]).drop_duplicates(subset=["date", "ticker"], keep="last")

    if drop_rows_without_adj_close:
        df = df.dropna(subset=["adj_close"])

    df = df[df["adj_close"] > 0]

    for col in ["close", "open", "high", "low"]:
        mask = df[col].isna()
        df.loc[mask, col] = df.loc[mask, "adj_close"]

    df["volume"] = df["volume"].fillna(0.0).clip(lower=0.0)
    df = df.dropna(subset=["open", "high", "low", "close", "adj_close"])

    counts = df.groupby("ticker").size()
    eligible_tickers = counts[counts >= min_history_days].index
    df = df[df["ticker"].isin(eligible_tickers)]

    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_feature_panel(
    clean_prices_df: pd.DataFrame,
    horizon_days: int = 5,
    target_column: str = "fwd_return_5d",
) -> pd.DataFrame:
    if horizon_days <= 0:
        raise ValueError("`horizon_days` must be a positive integer.")

    df = clean_prices_df.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker", group_keys=False)

    df["ret_1d"] = grouped["adj_close"].pct_change(periods=1)
    df["ret_5d"] = grouped["adj_close"].pct_change(periods=5)
    df["ret_20d"] = grouped["adj_close"].pct_change(periods=20)

    daily_ret = grouped["adj_close"].pct_change(periods=1)
    df["vol_20d"] = (
        daily_ret.groupby(df["ticker"])
        .rolling(window=20, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )

    # 60-to-20-day momentum excludes the latest month to reduce short-term reversal effects.
    df["momentum_20_60"] = grouped["adj_close"].shift(20) / grouped["adj_close"].shift(60) - 1.0
    df[target_column] = grouped["adj_close"].shift(-horizon_days) / df["adj_close"] - 1.0

    keep_cols = [
        "date",
        "ticker",
        "adj_close",
        "volume",
        *FEATURE_COLUMNS,
        target_column,
    ]
    panel = (
        df[keep_cols]
        .dropna(subset=[*FEATURE_COLUMNS, target_column])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    return panel


def run_build_panel(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    config = load_yaml(config_path)
    data_cfg = config.get("data")
    labels_cfg = config.get("labels", {})
    pre_cfg = config.get("preprocessing", {})

    if not isinstance(data_cfg, dict):
        raise ValueError("Missing `data` section in config_data.yaml")

    raw_path_cfg = data_cfg.get("output_raw_path")
    clean_path_cfg = data_cfg.get("output_clean_path")
    panel_path_cfg = data_cfg.get("output_panel_path")
    if not isinstance(raw_path_cfg, str):
        raise ValueError("`data.output_raw_path` must be a string path.")
    if not isinstance(clean_path_cfg, str):
        raise ValueError("`data.output_clean_path` must be a string path.")
    if not isinstance(panel_path_cfg, str):
        raise ValueError("`data.output_panel_path` must be a string path.")

    horizon_days = labels_cfg.get("horizon_days", 5)
    target_column = labels_cfg.get("target_column", "fwd_return_5d")
    min_history_days = pre_cfg.get("min_history_days", 252)
    drop_rows_without_adj_close = pre_cfg.get("drop_rows_without_adj_close", True)

    if not isinstance(horizon_days, int):
        raise ValueError("`labels.horizon_days` must be an integer.")
    if not isinstance(target_column, str):
        raise ValueError("`labels.target_column` must be a string.")
    if not isinstance(min_history_days, int):
        raise ValueError("`preprocessing.min_history_days` must be an integer.")
    if not isinstance(drop_rows_without_adj_close, bool):
        raise ValueError("`preprocessing.drop_rows_without_adj_close` must be a boolean.")

    project_root = config_path.parents[1]
    raw_path = (project_root / raw_path_cfg).resolve()
    clean_path = (project_root / clean_path_cfg).resolve()
    panel_path = (project_root / panel_path_cfg).resolve()

    raw_prices = pd.read_parquet(raw_path)
    clean_df = clean_prices(
        raw_prices=raw_prices,
        min_history_days=min_history_days,
        drop_rows_without_adj_close=drop_rows_without_adj_close,
    )
    panel_df = build_feature_panel(
        clean_prices_df=clean_df,
        horizon_days=horizon_days,
        target_column=target_column,
    )

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(clean_path, index=False)
    panel_df.to_parquet(panel_path, index=False)

    return clean_df, panel_df, clean_path, panel_path
