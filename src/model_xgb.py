from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from xgboost import XGBRegressor

from src.data import load_yaml


def _as_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize(None)


def _get_rebalance_dates(
    unique_dates: pd.Series,
    frequency: str,
    every_n_days: int | None = None,
) -> list[pd.Timestamp]:
    dates = pd.Series(pd.to_datetime(unique_dates).sort_values().unique())
    freq = frequency.lower()

    if freq == "daily":
        return dates.tolist()
    if freq == "weekly":
        return (
            dates.groupby(dates.dt.to_period("W-FRI"))
            .max()
            .sort_values()
            .tolist()
        )
    if freq == "monthly":
        return dates.groupby(dates.dt.to_period("M")).max().sort_values().tolist()
    if freq == "every_n_days":
        if every_n_days is None or every_n_days <= 0:
            raise ValueError("`rebalance_every_n_days` must be a positive integer for `every_n_days` frequency.")
        return dates.iloc[every_n_days - 1 :: every_n_days].tolist()

    raise ValueError(f"Unsupported rebalance frequency: {frequency}")


def _validate_panel(panel: pd.DataFrame, features: list[str], target_column: str) -> None:
    required = {"date", "ticker", target_column, *features}
    missing = [col for col in required if col not in panel.columns]
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")


def _make_model(params: dict[str, Any]) -> XGBRegressor:
    model_params = dict(params)
    model_params.setdefault("n_jobs", -1)
    return XGBRegressor(**model_params)


def train_walk_forward_xgb(
    panel: pd.DataFrame,
    features: list[str],
    target_column: str,
    model_params: dict[str, Any],
    train_window_days: int,
    validation_window_days: int,
    horizon_days: int,
    rebalance_frequency: str,
    rebalance_every_n_days: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if train_window_days <= 0:
        raise ValueError("`train_window_days` must be positive.")
    if validation_window_days < 0:
        raise ValueError("`validation_window_days` must be non-negative.")
    if horizon_days <= 0:
        raise ValueError("`horizon_days` must be positive.")

    _validate_panel(panel, features=features, target_column=target_column)

    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(None)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    unique_dates = pd.Series(df["date"].sort_values().unique())
    rebalance_dates = _get_rebalance_dates(
        unique_dates=unique_dates,
        frequency=rebalance_frequency,
        every_n_days=rebalance_every_n_days,
    )
    date_to_index = {d: idx for idx, d in enumerate(unique_dates)}

    prediction_frames: list[pd.DataFrame] = []
    logs: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []

    for rebalance_date in rebalance_dates:
        rebalance_date = _as_timestamp(rebalance_date)
        rebalance_idx = date_to_index.get(rebalance_date)
        if rebalance_idx is None or rebalance_idx < horizon_days:
            continue

        # Target at date t is known only once future horizon is realized, so we cap training labels at t-horizon.
        train_label_end = unique_dates.iloc[rebalance_idx - horizon_days]
        train_start = train_label_end - pd.Timedelta(days=train_window_days)
        val_start = train_label_end - pd.Timedelta(days=validation_window_days)

        train_mask = (df["date"] > train_start) & (df["date"] <= train_label_end)
        test_mask = df["date"] == rebalance_date

        train_df = df.loc[train_mask, ["date", "ticker", *features, target_column]].dropna()
        test_df = df.loc[test_mask, ["date", "ticker", *features, target_column]].dropna()
        if train_df.empty or test_df.empty:
            continue

        model = _make_model(model_params)
        x_train = train_df[features].to_numpy()
        y_train = train_df[target_column].to_numpy()
        model.fit(x_train, y_train)

        test_pred = model.predict(test_df[features].to_numpy())

        pred_frame = test_df[["date", "ticker", target_column]].copy()
        pred_frame["prediction"] = test_pred
        pred_frame["train_start_date"] = train_start
        pred_frame["train_label_end_date"] = train_label_end
        prediction_frames.append(pred_frame)

        val_mask = (train_df["date"] > val_start) & (train_df["date"] <= train_label_end)
        val_df = train_df.loc[val_mask]
        val_ic = None
        if len(val_df) >= 2:
            val_pred = model.predict(val_df[features].to_numpy())
            val_ic = pd.Series(val_pred).corr(val_df[target_column].reset_index(drop=True), method="spearman")

        logs.append(
            {
                "rebalance_date": rebalance_date,
                "train_start_date": train_start,
                "train_label_end_date": train_label_end,
                "train_rows": int(len(train_df)),
                "validation_rows": int(len(val_df)),
                "test_rows": int(len(test_df)),
                "test_tickers": int(test_df["ticker"].nunique()),
                "validation_ic_spearman": None if val_ic is None or pd.isna(val_ic) else float(val_ic),
            }
        )

        if hasattr(model, "feature_importances_"):
            importance_frames.append(
                pd.DataFrame(
                    {
                        "rebalance_date": rebalance_date,
                        "feature": features,
                        "importance": model.feature_importances_,
                    }
                )
            )

    if not prediction_frames:
        raise ValueError("No walk-forward predictions were produced. Check panel coverage and configs.")

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    training_log = pd.DataFrame(logs).sort_values("rebalance_date").reset_index(drop=True)
    importances = (
        pd.concat(importance_frames, ignore_index=True)
        if importance_frames
        else pd.DataFrame(columns=["rebalance_date", "feature", "importance"])
    )

    oos_ic = predictions["prediction"].corr(predictions[target_column], method="spearman")
    summary: dict[str, Any] = {
        "n_rebalances": int(training_log["rebalance_date"].nunique()),
        "n_predictions": int(len(predictions)),
        "date_start": str(predictions["date"].min().date()),
        "date_end": str(predictions["date"].max().date()),
        "oos_ic_spearman": None if pd.isna(oos_ic) else float(oos_ic),
        "avg_validation_ic_spearman": None
        if training_log["validation_ic_spearman"].dropna().empty
        else float(training_log["validation_ic_spearman"].dropna().mean()),
        "model_params": model_params,
        "features": features,
        "target_column": target_column,
        "rebalance_frequency": rebalance_frequency,
        "rebalance_every_n_days": rebalance_every_n_days,
        "train_window_days": train_window_days,
        "validation_window_days": validation_window_days,
        "horizon_days": horizon_days,
    }
    return predictions, training_log, importances, summary


def run_train(
    config_data_path: Path,
    config_model_path: Path,
    config_backtest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], Path, Path, Path, Path]:
    data_cfg = load_yaml(config_data_path)
    model_cfg = load_yaml(config_model_path)
    backtest_cfg = load_yaml(config_backtest_path)

    data_section = data_cfg.get("data")
    labels_section = data_cfg.get("labels", {})
    model_section = model_cfg.get("model")
    backtest_section = backtest_cfg.get("backtest")

    if not isinstance(data_section, dict):
        raise ValueError("Missing `data` section in config_data.yaml")
    if not isinstance(model_section, dict):
        raise ValueError("Missing `model` section in config_model.yaml")
    if not isinstance(backtest_section, dict):
        raise ValueError("Missing `backtest` section in config_backtest.yaml")

    panel_rel = data_section.get("output_panel_path")
    horizon_days = labels_section.get("horizon_days", 5)
    target_column = labels_section.get("target_column", "fwd_return_5d")
    features = model_section.get("features", [])
    model_params = model_section.get("params", {})

    train_window_days = backtest_section.get("train_window_days", 756)
    validation_window_days = backtest_section.get("validation_window_days", 252)
    rebalance_frequency = backtest_section.get("rebalance_frequency", "monthly")
    rebalance_every_n_days = backtest_section.get("rebalance_every_n_days")

    if not isinstance(panel_rel, str):
        raise ValueError("`data.output_panel_path` must be a string path.")
    if not isinstance(horizon_days, int):
        raise ValueError("`labels.horizon_days` must be an integer.")
    if not isinstance(target_column, str):
        raise ValueError("`labels.target_column` must be a string.")
    if not isinstance(features, list) or not all(isinstance(col, str) for col in features):
        raise ValueError("`model.features` must be a list[str].")
    if not isinstance(model_params, dict):
        raise ValueError("`model.params` must be a mapping.")
    if not isinstance(train_window_days, int):
        raise ValueError("`backtest.train_window_days` must be an integer.")
    if not isinstance(validation_window_days, int):
        raise ValueError("`backtest.validation_window_days` must be an integer.")
    if not isinstance(rebalance_frequency, str):
        raise ValueError("`backtest.rebalance_frequency` must be a string.")
    if rebalance_every_n_days is not None and not isinstance(rebalance_every_n_days, int):
        raise ValueError("`backtest.rebalance_every_n_days` must be null or an integer.")

    project_root = config_data_path.parents[1]
    panel_path = (project_root / panel_rel).resolve()
    panel_df = pd.read_parquet(panel_path)

    predictions, training_log, importances, summary = train_walk_forward_xgb(
        panel=panel_df,
        features=features,
        target_column=target_column,
        model_params=model_params,
        train_window_days=train_window_days,
        validation_window_days=validation_window_days,
        horizon_days=horizon_days,
        rebalance_frequency=rebalance_frequency,
        rebalance_every_n_days=rebalance_every_n_days,
    )

    out_dir = (project_root / "outputs/models").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = out_dir / "predictions_oos.parquet"
    log_path = out_dir / "training_log.parquet"
    importance_path = out_dir / "feature_importance.parquet"
    summary_path = out_dir / "train_summary.json"

    predictions.to_parquet(predictions_path, index=False)
    training_log.to_parquet(log_path, index=False)
    importances.to_parquet(importance_path, index=False)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    return (
        predictions,
        training_log,
        importances,
        summary,
        predictions_path,
        log_path,
        importance_path,
        summary_path,
    )
