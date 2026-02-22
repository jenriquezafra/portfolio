from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import build_feature_panel
from src.model_xgb import train_walk_forward_xgb


def _make_clean_prices(n_days: int = 120, ticker: str = "AAA") -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    # Smooth increasing series keeps expected forward-return checks deterministic.
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


def test_build_feature_panel_target_is_future_return() -> None:
    clean = _make_clean_prices(n_days=140, ticker="AAA")
    horizon_days = 5
    target_col = "fwd_return_test"
    panel = build_feature_panel(
        clean_prices_df=clean,
        horizon_days=horizon_days,
        target_column=target_col,
        target_mode="absolute",
    )

    assert not panel.empty

    by_ticker = clean.sort_values(["ticker", "date"]).reset_index(drop=True)
    sample = panel.iloc[10]
    sample_date = pd.Timestamp(sample["date"])
    idx_map = {pd.Timestamp(d): i for i, d in enumerate(by_ticker["date"])}
    i = idx_map[sample_date]
    px_t = float(by_ticker.iloc[i]["adj_close"])
    px_t_h = float(by_ticker.iloc[i + horizon_days]["adj_close"])
    expected = (px_t_h / px_t) - 1.0

    assert np.isclose(float(sample[target_col]), expected, atol=1e-12)

    # Guard against future leakage: no label should exist for the last H dates.
    last_valid_date = pd.Timestamp(by_ticker.iloc[-(horizon_days + 1)]["date"])
    assert pd.Timestamp(panel["date"].max()) <= last_valid_date


def test_walk_forward_training_respects_horizon_and_purge_gap() -> None:
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2021-01-01", periods=90)
    tickers = [f"T{i}" for i in range(6)]

    rows: list[dict[str, object]] = []
    for d in dates:
        for t in tickers:
            f1 = float(rng.normal())
            f2 = float(rng.normal())
            target = 0.2 * f1 - 0.1 * f2 + float(rng.normal(scale=0.05))
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "f1": f1,
                    "f2": f2,
                    "target": target,
                }
            )
    panel = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)

    horizon_days = 5
    _, training_log, _, _ = train_walk_forward_xgb(
        panel=panel,
        features=["f1", "f2"],
        target_column="target",
        model_params={
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "objective": "reg:squarederror",
            "random_state": 7,
            "n_jobs": 1,
        },
        train_window_days=30,
        validation_window_days=15,
        horizon_days=horizon_days,
        rebalance_frequency="every_n_days",
        rebalance_every_n_days=5,
    )

    assert not training_log.empty

    unique_dates = pd.Series(pd.to_datetime(panel["date"]).sort_values().unique())
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(unique_dates)}

    for row in training_log.itertuples(index=False):
        rebalance_idx = date_to_idx[pd.Timestamp(row.rebalance_date)]
        label_end_idx = date_to_idx[pd.Timestamp(row.train_label_end_date)]
        assert rebalance_idx - label_end_idx >= horizon_days

        if pd.notna(row.validation_start_date):
            train_end_idx = date_to_idx[pd.Timestamp(row.train_end_date)]
            val_start_idx = date_to_idx[pd.Timestamp(row.validation_start_date)]
            assert val_start_idx - train_end_idx >= horizon_days
