from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import load_yaml

REQUIRED_RAW_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
FEATURE_COLUMNS = [
    "ret_1d",
    "ret_2d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_63d",
    "vol_5d",
    "vol_20d",
    "vol_60d",
    "vol_ratio_20_60",
    "downside_vol_20d",
    "upside_vol_20d",
    "momentum_20_60",
    "momentum_5_20",
    "momentum_x_vol_20",
    "reversal_1_5",
    "overnight_gap_1d",
    "intraday_return_1d",
    "high_low_range_1d",
    "dollar_volume_log_20d",
    "dollar_volume_z_60d",
]


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
    target_mode: str = "absolute",
    market_context_df: pd.DataFrame | None = None,
    drop_target_na: bool = True,
) -> pd.DataFrame:
    if horizon_days <= 0:
        raise ValueError("`horizon_days` must be a positive integer.")
    if target_mode not in {"absolute", "cross_sectional_demeaned"}:
        raise ValueError("`target_mode` must be one of: `absolute`, `cross_sectional_demeaned`.")

    df = clean_prices_df.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker", group_keys=False)

    df["ret_1d"] = grouped["adj_close"].pct_change(periods=1)
    df["ret_2d"] = grouped["adj_close"].pct_change(periods=2)
    df["ret_5d"] = grouped["adj_close"].pct_change(periods=5)
    df["ret_10d"] = grouped["adj_close"].pct_change(periods=10)
    df["ret_20d"] = grouped["adj_close"].pct_change(periods=20)
    df["ret_63d"] = grouped["adj_close"].pct_change(periods=63)

    daily_ret = grouped["adj_close"].pct_change(periods=1)
    df["vol_5d"] = (
        daily_ret.groupby(df["ticker"])
        .rolling(window=5, min_periods=5)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["vol_20d"] = (
        daily_ret.groupby(df["ticker"])
        .rolling(window=20, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["vol_60d"] = (
        daily_ret.groupby(df["ticker"])
        .rolling(window=60, min_periods=60)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["vol_ratio_20_60"] = df["vol_20d"] / df["vol_60d"]

    downside_ret = daily_ret.clip(upper=0.0)
    upside_ret = daily_ret.clip(lower=0.0)
    df["downside_vol_20d"] = (
        downside_ret.groupby(df["ticker"])
        .rolling(window=20, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["upside_vol_20d"] = (
        upside_ret.groupby(df["ticker"])
        .rolling(window=20, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )

    # 60-to-20-day momentum excludes the latest month to reduce short-term reversal effects.
    df["momentum_20_60"] = grouped["adj_close"].shift(20) / grouped["adj_close"].shift(60) - 1.0
    # 20-to-5-day momentum excludes the latest week.
    df["momentum_5_20"] = grouped["adj_close"].shift(5) / grouped["adj_close"].shift(20) - 1.0
    df["momentum_x_vol_20"] = df["momentum_20_60"] * df["vol_20d"]
    df["reversal_1_5"] = df["ret_1d"] - df["ret_5d"]

    prev_close = grouped["close"].shift(1)
    df["overnight_gap_1d"] = df["open"] / prev_close - 1.0
    df["intraday_return_1d"] = df["close"] / df["open"] - 1.0
    df["high_low_range_1d"] = (df["high"] - df["low"]) / df["adj_close"]

    df["dollar_volume_1d"] = df["adj_close"] * df["volume"]
    df["dollar_volume_20d"] = (
        df.groupby("ticker", group_keys=False)["dollar_volume_1d"]
        .rolling(window=20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["dollar_volume_log_20d"] = np.log1p(df["dollar_volume_20d"])
    dv_mean_60d = (
        df.groupby("ticker", group_keys=False)["dollar_volume_log_20d"]
        .rolling(window=60, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    dv_std_60d = (
        df.groupby("ticker", group_keys=False)["dollar_volume_log_20d"]
        .rolling(window=60, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["dollar_volume_z_60d"] = (df["dollar_volume_log_20d"] - dv_mean_60d) / dv_std_60d

    # Clean numerical artifacts before dropping NaNs.
    numeric_cols = [
        "vol_ratio_20_60",
        "overnight_gap_1d",
        "intraday_return_1d",
        "high_low_range_1d",
        "dollar_volume_z_60d",
    ]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df["fwd_return_raw"] = grouped["adj_close"].shift(-horizon_days) / df["adj_close"] - 1.0

    extra_feature_cols: list[str] = []
    if market_context_df is not None and not market_context_df.empty:
        if "date" not in market_context_df.columns:
            raise ValueError("`market_context_df` must include a `date` column.")
        context = market_context_df.copy()
        context["date"] = pd.to_datetime(context["date"], utc=False).dt.tz_localize(None)
        context = context.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        extra_feature_cols = [col for col in context.columns if col != "date"]
        if extra_feature_cols:
            df = df.merge(context[["date", *extra_feature_cols]], on="date", how="left")

    model_feature_cols = [*FEATURE_COLUMNS, *extra_feature_cols]
    if target_mode == "absolute":
        df[target_column] = df["fwd_return_raw"]
    else:
        # De-mean on the modelable universe (rows with non-null features/label inputs) for each date.
        modelable_mask = df[[*model_feature_cols, "fwd_return_raw"]].notna().all(axis=1)
        cross_mean = df["fwd_return_raw"].where(modelable_mask).groupby(df["date"], group_keys=False).transform("mean")
        df[target_column] = df["fwd_return_raw"] - cross_mean

    keep_cols = [
        "date",
        "ticker",
        "adj_close",
        "volume",
        *model_feature_cols,
        "fwd_return_raw",
        target_column,
    ]
    keep_cols = list(dict.fromkeys(keep_cols))
    drop_subset = [*model_feature_cols, target_column] if drop_target_na else [*model_feature_cols]
    panel = df[keep_cols].dropna(subset=drop_subset).sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel


def _load_market_context_from_config(
    project_root: Path,
    market_context_cfg: dict[str, Any],
) -> pd.DataFrame | None:
    enabled = bool(market_context_cfg.get("enabled", False))
    if not enabled:
        return None

    path_cfg = market_context_cfg.get("path")
    if not isinstance(path_cfg, str):
        raise ValueError("`market_context.path` must be a string when market context is enabled.")
    context_path = Path(path_cfg)
    if not context_path.is_absolute():
        context_path = (project_root / context_path).resolve()

    if context_path.suffix.lower() == ".parquet":
        context = pd.read_parquet(context_path)
    else:
        context = pd.read_csv(context_path)

    if "date" not in context.columns:
        raise ValueError("Market context source must include a `date` column.")

    columns_cfg = market_context_cfg.get("columns")
    if columns_cfg is None:
        selected_cols = [col for col in context.columns if col != "date"]
    else:
        if not isinstance(columns_cfg, list) or not all(isinstance(col, str) for col in columns_cfg):
            raise ValueError("`market_context.columns` must be a list[str].")
        selected_cols = list(columns_cfg)

    if not selected_cols:
        raise ValueError("`market_context.columns` resolved to an empty list.")
    missing = [col for col in selected_cols if col not in context.columns]
    if missing:
        raise ValueError(f"Market context source is missing configured columns: {missing}")

    lag_days = market_context_cfg.get("lag_days", 1)
    if not isinstance(lag_days, int) or lag_days <= 0:
        raise ValueError("`market_context.lag_days` must be a positive integer.")

    prefix = str(market_context_cfg.get("feature_prefix", "mkt")).strip()
    if not prefix:
        raise ValueError("`market_context.feature_prefix` must be a non-empty string.")

    context = context[["date", *selected_cols]].copy()
    context["date"] = pd.to_datetime(context["date"], utc=False).dt.tz_localize(None)
    context = context.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    for col in selected_cols:
        context[col] = pd.to_numeric(context[col], errors="coerce")
    context[selected_cols] = context[selected_cols].shift(lag_days)

    rename_map = {col: f"{prefix}_{col}_lag{lag_days}" for col in selected_cols}
    return context.rename(columns=rename_map)


def run_build_panel(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    config = load_yaml(config_path)
    data_cfg = config.get("data")
    labels_cfg = config.get("labels", {})
    pre_cfg = config.get("preprocessing", {})
    market_context_cfg = config.get("market_context", {})

    if not isinstance(data_cfg, dict):
        raise ValueError("Missing `data` section in config_data.yaml")
    if not isinstance(market_context_cfg, dict):
        raise ValueError("`market_context` must be a mapping.")

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
    target_mode = labels_cfg.get("target_mode", "absolute")
    min_history_days = pre_cfg.get("min_history_days", 252)
    drop_rows_without_adj_close = pre_cfg.get("drop_rows_without_adj_close", True)

    if not isinstance(horizon_days, int):
        raise ValueError("`labels.horizon_days` must be an integer.")
    if not isinstance(target_column, str):
        raise ValueError("`labels.target_column` must be a string.")
    if not isinstance(target_mode, str):
        raise ValueError("`labels.target_mode` must be a string.")
    if not isinstance(min_history_days, int):
        raise ValueError("`preprocessing.min_history_days` must be an integer.")
    if not isinstance(drop_rows_without_adj_close, bool):
        raise ValueError("`preprocessing.drop_rows_without_adj_close` must be a boolean.")

    project_root = config_path.parents[1]
    raw_path = (project_root / raw_path_cfg).resolve()
    clean_path = (project_root / clean_path_cfg).resolve()
    panel_path = (project_root / panel_path_cfg).resolve()
    market_context_df = _load_market_context_from_config(
        project_root=project_root,
        market_context_cfg=market_context_cfg,
    )

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
        target_mode=target_mode,
        market_context_df=market_context_df,
    )

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(clean_path, index=False)
    panel_df.to_parquet(panel_path, index=False)

    return clean_df, panel_df, clean_path, panel_path
