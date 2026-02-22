from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
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


def _spearman_rank_corr(lhs: pd.Series | np.ndarray, rhs: pd.Series | np.ndarray) -> float | None:
    left = pd.Series(lhs).rank(method="average")
    right = pd.Series(rhs).rank(method="average")
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return None
    corr = left.corr(right, method="pearson")
    if pd.isna(corr):
        return None
    return float(corr)


def _top_bottom_spread(
    predictions: pd.Series | np.ndarray,
    realized: pd.Series | np.ndarray,
    quantile: float = 0.20,
) -> float | None:
    if quantile <= 0.0 or quantile >= 0.5:
        raise ValueError("`quantile` must be in (0.0, 0.5).")

    frame = pd.DataFrame(
        {
            "prediction": pd.Series(predictions).astype(float),
            "realized": pd.Series(realized).astype(float),
        }
    ).dropna()
    if frame.empty:
        return None

    bucket = int(np.floor(len(frame) * quantile))
    if bucket < 1:
        return None

    ranked = frame.sort_values("prediction")
    bottom = ranked.head(bucket)["realized"].mean()
    top = ranked.tail(bucket)["realized"].mean()
    spread = top - bottom
    if pd.isna(spread):
        return None
    return float(spread)


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

        # Label at date t requires future horizon; the latest known label is t-horizon.
        train_label_end_idx = rebalance_idx - horizon_days
        train_label_end = unique_dates.iloc[train_label_end_idx]
        if validation_window_days > 0:
            val_end_idx = train_label_end_idx
            val_start_idx = max(0, val_end_idx - validation_window_days + 1)

            # Purge train labels that could overlap with the validation label horizon.
            train_end_idx = val_start_idx - horizon_days
            if train_end_idx < 0:
                continue
            train_start_idx = max(0, train_end_idx - train_window_days + 1)
            if train_end_idx < train_start_idx:
                continue

            val_dates = unique_dates.iloc[val_start_idx : val_end_idx + 1]
            validation_start_date = val_dates.iloc[0]
            validation_end_date = val_dates.iloc[-1]
        else:
            train_end_idx = train_label_end_idx
            train_start_idx = max(0, train_end_idx - train_window_days + 1)
            validation_start_date = pd.NaT
            validation_end_date = pd.NaT
            val_dates = pd.Series([], dtype="datetime64[ns]")

        train_dates = unique_dates.iloc[train_start_idx : train_end_idx + 1]
        train_start = train_dates.iloc[0]
        train_end = train_dates.iloc[-1]

        train_mask = df["date"].isin(train_dates)
        val_mask = df["date"].isin(val_dates) if validation_window_days > 0 else pd.Series(False, index=df.index)
        test_mask = df["date"] == rebalance_date

        train_df = df.loc[train_mask, ["date", "ticker", *features, target_column]].dropna()
        val_df = df.loc[val_mask, ["date", "ticker", *features, target_column]].dropna()
        test_df = df.loc[test_mask, ["date", "ticker", *features, target_column]].dropna()
        if train_df.empty or test_df.empty:
            continue
        if validation_window_days > 0 and val_df.empty:
            continue

        current_params = dict(model_params)
        fit_kwargs: dict[str, Any] = {}
        if not val_df.empty:
            current_params.setdefault("early_stopping_rounds", 50)
            current_params.setdefault("eval_metric", "rmse")
            fit_kwargs["eval_set"] = [(val_df[features].to_numpy(), val_df[target_column].to_numpy())]
            fit_kwargs["verbose"] = False
        else:
            current_params.pop("early_stopping_rounds", None)

        model = _make_model(current_params)
        x_train = train_df[features].to_numpy()
        y_train = train_df[target_column].to_numpy()
        model.fit(x_train, y_train, **fit_kwargs)

        test_pred = model.predict(test_df[features].to_numpy())

        pred_frame = test_df[["date", "ticker", target_column]].copy()
        pred_frame["prediction"] = test_pred
        pred_frame["train_start_date"] = train_start
        pred_frame["train_end_date"] = train_end
        pred_frame["validation_start_date"] = validation_start_date
        pred_frame["validation_end_date"] = validation_end_date
        pred_frame["train_label_end_date"] = train_label_end
        prediction_frames.append(pred_frame)

        val_ic = None
        if len(val_df) >= 2:
            val_pred = model.predict(val_df[features].to_numpy())
            val_ic = _spearman_rank_corr(val_pred, val_df[target_column].to_numpy())

        oos_cs_ic = None
        if len(test_df) >= 2:
            oos_cs_ic = _spearman_rank_corr(test_pred, test_df[target_column].to_numpy())
        oos_top_bottom = _top_bottom_spread(test_pred, test_df[target_column].to_numpy(), quantile=0.20)

        best_iteration = getattr(model, "best_iteration", None)
        best_iteration_int = int(best_iteration) if best_iteration is not None else None

        logs.append(
            {
                "rebalance_date": rebalance_date,
                "train_start_date": train_start,
                "train_end_date": train_end,
                "train_label_end_date": train_label_end,
                "validation_start_date": validation_start_date,
                "validation_end_date": validation_end_date,
                "train_rows": int(len(train_df)),
                "validation_rows": int(len(val_df)),
                "test_rows": int(len(test_df)),
                "test_tickers": int(test_df["ticker"].nunique()),
                "validation_ic_spearman": None if val_ic is None or pd.isna(val_ic) else float(val_ic),
                "oos_cs_ic_spearman": None if oos_cs_ic is None or pd.isna(oos_cs_ic) else float(oos_cs_ic),
                "oos_top_bottom_spread": None
                if oos_top_bottom is None or pd.isna(oos_top_bottom)
                else float(oos_top_bottom),
                "best_iteration": best_iteration_int,
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

    oos_ic = _spearman_rank_corr(predictions["prediction"], predictions[target_column])
    cs_ic = training_log["oos_cs_ic_spearman"].dropna()
    cs_ic_mean = None if cs_ic.empty else float(cs_ic.mean())
    cs_ic_std = None if cs_ic.empty else float(cs_ic.std(ddof=1))
    cs_ic_ir = None
    if cs_ic_mean is not None and cs_ic_std is not None and cs_ic_std > 0:
        cs_ic_ir = float(cs_ic_mean / cs_ic_std)

    top_bottom = training_log["oos_top_bottom_spread"].dropna()
    top_bottom_mean = None if top_bottom.empty else float(top_bottom.mean())
    top_bottom_std = None if top_bottom.empty else float(top_bottom.std(ddof=1))
    top_bottom_tstat = None
    if top_bottom_mean is not None and top_bottom_std is not None and top_bottom_std > 0:
        top_bottom_tstat = float(top_bottom_mean / (top_bottom_std / np.sqrt(len(top_bottom))))

    summary: dict[str, Any] = {
        "n_rebalances": int(training_log["rebalance_date"].nunique()),
        "n_predictions": int(len(predictions)),
        "date_start": str(predictions["date"].min().date()),
        "date_end": str(predictions["date"].max().date()),
        "oos_ic_spearman": oos_ic,
        "oos_cs_ic_mean": cs_ic_mean,
        "oos_cs_ic_median": None if cs_ic.empty else float(cs_ic.median()),
        "oos_cs_ic_std": cs_ic_std,
        "oos_cs_ic_ir": cs_ic_ir,
        "oos_cs_ic_positive_rate": None if cs_ic.empty else float((cs_ic > 0).mean()),
        "oos_top_bottom_mean": top_bottom_mean,
        "oos_top_bottom_median": None if top_bottom.empty else float(top_bottom.median()),
        "oos_top_bottom_positive_rate": None if top_bottom.empty else float((top_bottom > 0).mean()),
        "oos_top_bottom_tstat": top_bottom_tstat,
        "avg_validation_ic_spearman": None
        if training_log["validation_ic_spearman"].dropna().empty
        else float(training_log["validation_ic_spearman"].dropna().mean()),
        "validation_scheme": "time_series_purged",
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
