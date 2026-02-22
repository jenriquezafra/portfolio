from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yfinance as yf
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def load_data_settings(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("Missing `data` section in config_data.yaml")
    return data_cfg


def _normalize_downloaded_prices(raw: pd.DataFrame, universe: Sequence[str]) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("No data returned by data provider.")

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        level1 = set(raw.columns.get_level_values(1))

        if "Adj Close" in level0:
            long_df = raw.stack(level=1, future_stack=True).rename_axis(index=["date", "ticker"]).reset_index()
        elif "Adj Close" in level1:
            swapped = raw.swaplevel(axis=1).sort_index(axis=1)
            long_df = swapped.stack(level=1, future_stack=True).rename_axis(index=["date", "ticker"]).reset_index()
        else:
            raise ValueError("Expected an `Adj Close` column from yfinance output.")
    else:
        ticker = universe[0]
        long_df = raw.copy()
        long_df["ticker"] = ticker
        long_df = long_df.reset_index().rename(columns={"Date": "date"})

    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    long_df = long_df.rename(columns=rename_map)

    required_cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    missing = [col for col in required_cols if col not in long_df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    out = long_df[required_cols].copy()
    out["date"] = pd.to_datetime(out["date"], utc=False).dt.tz_localize(None)
    out["ticker"] = out["ticker"].astype(str)
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    return out


def fetch_prices_from_yfinance(
    universe: Sequence[str],
    start_date: str,
    end_date: str | None,
    frequency: str,
) -> pd.DataFrame:
    if not universe:
        raise ValueError("Universe is empty.")

    raw = yf.download(
        tickers=list(universe),
        start=start_date,
        end=end_date,
        interval=frequency,
        auto_adjust=False,
        actions=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    return _normalize_downloaded_prices(raw=raw, universe=universe)


def run_fetch_data(config_path: Path) -> tuple[pd.DataFrame, Path]:
    data_cfg = load_data_settings(config_path)

    universe = data_cfg.get("universe", [])
    start_date = data_cfg.get("start_date")
    end_date = data_cfg.get("end_date")
    frequency = data_cfg.get("frequency", "1d")
    output_raw_path = data_cfg.get("output_raw_path")

    if not isinstance(universe, list) or not all(isinstance(x, str) for x in universe):
        raise ValueError("`data.universe` must be a list[str].")
    if not isinstance(start_date, str):
        raise ValueError("`data.start_date` must be a string (YYYY-MM-DD).")
    if end_date is not None and not isinstance(end_date, str):
        raise ValueError("`data.end_date` must be null or a string (YYYY-MM-DD).")
    if not isinstance(frequency, str):
        raise ValueError("`data.frequency` must be a string.")
    if not isinstance(output_raw_path, str):
        raise ValueError("`data.output_raw_path` must be a string path.")

    prices = fetch_prices_from_yfinance(
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
    )

    project_root = config_path.parents[1]
    output_path = (project_root / output_raw_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(output_path, index=False)

    return prices, output_path
