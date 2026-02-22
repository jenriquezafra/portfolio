from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features import build_feature_panel, run_build_panel


def _make_clean_prices(n_days: int = 140, ticker: str = "AAA") -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    adj_close = 100.0 + np.arange(n_days, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": adj_close,
            "high": adj_close,
            "low": adj_close,
            "close": adj_close,
            "adj_close": adj_close,
            "volume": 1_000_000.0,
        }
    )


def test_build_feature_panel_with_market_context_columns() -> None:
    clean = _make_clean_prices()
    dates = pd.to_datetime(clean["date"].drop_duplicates().sort_values().reset_index(drop=True))
    context = pd.DataFrame(
        {
            "date": dates,
            "mkt_nasdaq_proxy_lag1": np.arange(len(dates), dtype=float) - 1.0,
            "mkt_growth_proxy_lag1": (np.arange(len(dates), dtype=float) - 1.0) * 0.5,
        }
    )
    panel = build_feature_panel(
        clean_prices_df=clean,
        horizon_days=5,
        target_column="fwd_return_test",
        target_mode="absolute",
        market_context_df=context,
    )

    assert not panel.empty
    assert "mkt_nasdaq_proxy_lag1" in panel.columns
    assert "mkt_growth_proxy_lag1" in panel.columns
    assert panel["mkt_nasdaq_proxy_lag1"].notna().all()
    assert panel["mkt_growth_proxy_lag1"].notna().all()


def test_run_build_panel_market_context_is_lagged_one_day(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "external" / "factors").mkdir(parents=True, exist_ok=True)

    raw = _make_clean_prices()
    raw_path = project_root / "data" / "raw" / "prices.parquet"
    raw.to_parquet(raw_path, index=False)

    dates = pd.to_datetime(raw["date"].drop_duplicates().sort_values().reset_index(drop=True))
    factor = pd.DataFrame(
        {
            "date": dates,
            "nasdaq_proxy": np.arange(len(dates), dtype=float),
            "growth_proxy": np.arange(len(dates), dtype=float) * 2.0,
        }
    )
    factor_path = project_root / "data" / "external" / "factors" / "factors.parquet"
    factor.to_parquet(factor_path, index=False)

    config_data = {
        "data": {
            "output_raw_path": "data/raw/prices.parquet",
            "output_clean_path": "data/processed/prices_clean.parquet",
            "output_panel_path": "data/processed/panel.parquet",
        },
        "labels": {
            "horizon_days": 5,
            "target_column": "fwd_return_5d_resid",
            "target_mode": "cross_sectional_demeaned",
        },
        "preprocessing": {
            "min_history_days": 1,
            "drop_rows_without_adj_close": True,
        },
        "market_context": {
            "enabled": True,
            "path": "data/external/factors/factors.parquet",
            "columns": ["nasdaq_proxy", "growth_proxy"],
            "feature_prefix": "mkt",
            "lag_days": 1,
        },
    }
    cfg_path = project_root / "configs" / "config_data.yaml"
    cfg_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

    _, panel, _, _ = run_build_panel(config_path=cfg_path)
    assert "mkt_nasdaq_proxy_lag1" in panel.columns
    assert "mkt_growth_proxy_lag1" in panel.columns
    assert not panel.empty

    sample = panel.sort_values("date").iloc[0]
    sample_date = pd.Timestamp(sample["date"])
    date_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    i = date_idx[sample_date]
    assert i >= 1

    expected_nasdaq = float(factor.iloc[i - 1]["nasdaq_proxy"])
    expected_growth = float(factor.iloc[i - 1]["growth_proxy"])
    assert np.isclose(float(sample["mkt_nasdaq_proxy_lag1"]), expected_nasdaq, atol=1e-12)
    assert np.isclose(float(sample["mkt_growth_proxy_lag1"]), expected_growth, atol=1e-12)
