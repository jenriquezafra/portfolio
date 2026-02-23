from __future__ import annotations

import numpy as np
import pandas as pd

from src.model_xgb import train_walk_forward_xgb


def test_train_walk_forward_with_cross_sectional_rank_target_transform() -> None:
    rng = np.random.default_rng(2026)
    dates = pd.bdate_range("2021-01-01", periods=90)
    tickers = [f"T{i}" for i in range(10)]

    rows: list[dict[str, object]] = []
    for d in dates:
        for idx, ticker in enumerate(tickers):
            f1 = float(rng.normal())
            f2 = float(rng.normal())
            # Add mild cross-sectional structure by ticker bucket.
            target = 0.15 * f1 - 0.10 * f2 + 0.005 * idx + float(rng.normal(scale=0.03))
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "f1": f1,
                    "f2": f2,
                    "target": target,
                }
            )
    panel = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)

    predictions, training_log, _, summary = train_walk_forward_xgb(
        panel=panel,
        features=["f1", "f2"],
        target_column="target",
        model_params={
            "n_estimators": 20,
            "max_depth": 2,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": 1,
        },
        train_window_days=35,
        validation_window_days=15,
        horizon_days=5,
        rebalance_frequency="every_n_days",
        rebalance_every_n_days=5,
        training_target_transform="cross_sectional_rank",
    )

    assert not predictions.empty
    assert not training_log.empty
    assert summary["training_target_transform"] == "cross_sectional_rank"
