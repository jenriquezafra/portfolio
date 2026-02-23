from __future__ import annotations

import numpy as np
import pandas as pd

from src.model_xgb import predict_latest_live_xgb


def test_predict_latest_live_xgb_uses_latest_panel_date() -> None:
    rng = np.random.default_rng(2026)
    dates = pd.bdate_range("2024-01-02", periods=50)
    tickers = [f"T{i}" for i in range(8)]

    rows: list[dict[str, object]] = []
    for d in dates:
        for idx, ticker in enumerate(tickers):
            f1 = float(rng.normal())
            f2 = float(rng.normal())
            target = 0.20 * f1 - 0.08 * f2 + 0.004 * idx + float(rng.normal(scale=0.03))
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

    # Mimic realistic horizon labels: latest rows have no realized target yet.
    cutoff = pd.Timestamp(dates[-5])
    panel.loc[panel["date"] > cutoff, "target"] = np.nan

    predictions, importances, summary = predict_latest_live_xgb(
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
        training_target_transform="cross_sectional_rank",
    )

    assert not predictions.empty
    assert predictions["date"].nunique() == 1
    assert pd.Timestamp(predictions["date"].iloc[0]) == pd.Timestamp(panel["date"].max())
    assert int(summary["n_live_assets"]) == len(tickers)
    assert summary["training_target_transform"] == "cross_sectional_rank"
    assert not importances.empty
