from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build rebalance execution history table and diagnostics from outputs/execution summaries."
    )
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=Path("outputs/execution"),
        help="Directory containing rebalance_*_summary.json files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/execution/rebalance_history.csv"),
        help="CSV output path for per-rebalance history rows.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/execution/rebalance_history_summary.json"),
        help="JSON output path for aggregate diagnostics.",
    )
    return parser.parse_args()


SUMMARY_FILENAME_RE = re.compile(r"^rebalance_\d{4}-\d{2}-\d{2}_.+_summary\.json$")


def _list_rebalance_summary_paths(execution_dir: Path) -> list[Path]:
    candidates = sorted(execution_dir.glob("rebalance_*_summary.json"))
    return [p for p in candidates if SUMMARY_FILENAME_RE.match(p.name)]


def _resolve_project_path(path_like: Any) -> Path | None:
    if not isinstance(path_like, str) or not path_like.strip():
        return None
    path = Path(path_like)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_weight_stats(summary: dict[str, Any]) -> dict[str, Any]:
    rebalance_date_raw = summary.get("rebalance_date")
    rebalance_date = None if rebalance_date_raw is None else pd.Timestamp(rebalance_date_raw).tz_localize(None)
    meta = summary.get("weights_source_meta", {})
    if not isinstance(meta, dict):
        return {
            "weights_stats_source": None,
            "target_weight_gross": None,
            "target_weight_net": None,
            "target_weight_max": None,
            "target_weight_min": None,
            "target_positions_count": None,
        }

    weight_series: pd.Series | None = None
    source_name: str | None = None

    weights_csv_path = _resolve_project_path(meta.get("weights_csv"))
    if weights_csv_path is not None and weights_csv_path.exists():
        weights_df = pd.read_csv(weights_csv_path)
        if "ticker" in weights_df.columns and "weight" in weights_df.columns:
            cleaned = weights_df[["ticker", "weight"]].copy()
            cleaned["ticker"] = cleaned["ticker"].astype(str)
            cleaned["weight"] = pd.to_numeric(cleaned["weight"], errors="coerce")
            cleaned = cleaned.dropna(subset=["weight"])
            weight_series = cleaned.groupby("ticker")["weight"].sum().astype(float)
            source_name = "weights_csv"

    if weight_series is None:
        history_path = _resolve_project_path(meta.get("weights_history_path"))
        if history_path is not None and history_path.exists() and rebalance_date is not None:
            weights_history = pd.read_parquet(history_path)
            if {"rebalance_date", "ticker", "weight"}.issubset(set(weights_history.columns)):
                weights_history["rebalance_date"] = (
                    pd.to_datetime(weights_history["rebalance_date"], utc=False).dt.tz_localize(None)
                )
                slice_df = weights_history[weights_history["rebalance_date"] == rebalance_date]
                if not slice_df.empty:
                    cleaned = slice_df[["ticker", "weight"]].copy()
                    cleaned["ticker"] = cleaned["ticker"].astype(str)
                    cleaned["weight"] = pd.to_numeric(cleaned["weight"], errors="coerce")
                    cleaned = cleaned.dropna(subset=["weight"])
                    weight_series = cleaned.groupby("ticker")["weight"].sum().astype(float)
                    source_name = "weights_history"

    if weight_series is None or weight_series.empty:
        return {
            "weights_stats_source": source_name,
            "target_weight_gross": None,
            "target_weight_net": None,
            "target_weight_max": None,
            "target_weight_min": None,
            "target_positions_count": None,
        }

    return {
        "weights_stats_source": source_name,
        "target_weight_gross": float(weight_series.abs().sum()),
        "target_weight_net": float(weight_series.sum()),
        "target_weight_max": float(weight_series.max()),
        "target_weight_min": float(weight_series.min()),
        "target_positions_count": int(weight_series.size),
    }


def _load_order_stats(summary: dict[str, Any]) -> dict[str, Any]:
    orders_path = _resolve_project_path(summary.get("orders_file"))
    if orders_path is None or not orders_path.exists():
        return {
            "orders_rows": None,
            "planned_buy_notional": None,
            "planned_sell_notional": None,
            "planned_net_notional": None,
        }

    try:
        orders_df = pd.read_csv(orders_path)
    except Exception:
        return {
            "orders_rows": None,
            "planned_buy_notional": None,
            "planned_sell_notional": None,
            "planned_net_notional": None,
        }

    if not {"delta_qty", "price"}.issubset(set(orders_df.columns)):
        return {
            "orders_rows": int(len(orders_df)),
            "planned_buy_notional": None,
            "planned_sell_notional": None,
            "planned_net_notional": None,
        }

    qty = pd.to_numeric(orders_df["delta_qty"], errors="coerce").fillna(0.0)
    px = pd.to_numeric(orders_df["price"], errors="coerce").fillna(0.0)
    signed_notional = qty * px
    buy_notional = float(signed_notional[signed_notional > 0].sum())
    sell_notional = float((-signed_notional[signed_notional < 0]).sum())
    net_notional = float(signed_notional.sum())
    return {
        "orders_rows": int(len(orders_df)),
        "planned_buy_notional": buy_notional,
        "planned_sell_notional": sell_notional,
        "planned_net_notional": net_notional,
    }


def _load_run_all_context(
    summary: dict[str, Any],
    recommendation_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    meta = summary.get("weights_source_meta", {})
    if not isinstance(meta, dict):
        return {
            "run_all_signal_gate_multiplier": None,
            "run_all_signal_gate_enabled": None,
            "run_all_turnover_cap_applied": None,
            "run_all_used_existing_positions": None,
        }

    recommendation_path = _resolve_project_path(meta.get("recommendation_path"))
    if recommendation_path is None or not recommendation_path.exists():
        return {
            "run_all_signal_gate_multiplier": None,
            "run_all_signal_gate_enabled": None,
            "run_all_turnover_cap_applied": None,
            "run_all_used_existing_positions": None,
        }

    key = str(recommendation_path)
    if key not in recommendation_cache:
        try:
            recommendation_cache[key] = json.loads(recommendation_path.read_text(encoding="utf-8"))
        except Exception:
            recommendation_cache[key] = {}

    recommendation = recommendation_cache[key]
    live_summary = recommendation.get("live_weights_summary", {})
    if not isinstance(live_summary, dict):
        live_summary = {}
    return {
        "run_all_signal_gate_multiplier": _to_float(live_summary.get("signal_gate_multiplier")),
        "run_all_signal_gate_enabled": live_summary.get("signal_quality_gate_enabled"),
        "run_all_turnover_cap_applied": live_summary.get("turnover_cap_applied"),
        "run_all_used_existing_positions": live_summary.get("used_existing_positions"),
    }


def _build_history_row(
    summary_path: Path,
    summary: dict[str, Any],
    recommendation_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    weight_stats = _load_weight_stats(summary=summary)
    order_stats = _load_order_stats(summary=summary)
    run_all_stats = _load_run_all_context(summary=summary, recommendation_cache=recommendation_cache)

    equity = _to_float(summary.get("account_equity"))
    cash = _to_float(summary.get("account_cash"))
    planned_net_notional = _to_float(order_stats.get("planned_net_notional"))
    estimated_cash_after = None
    estimated_cash_after_fraction = None
    if equity is not None and cash is not None and planned_net_notional is not None:
        estimated_cash_after = float(cash - planned_net_notional)
        if abs(equity) > 1e-12:
            estimated_cash_after_fraction = float(estimated_cash_after / equity)

    pre_cash_fraction = None
    if equity is not None and cash is not None and abs(equity) > 1e-12:
        pre_cash_fraction = float(cash / equity)

    min_cash_buffer = _to_float(summary.get("min_cash_buffer"))
    investable_target = None
    if min_cash_buffer is not None:
        investable_target = float(max(0.0, 1.0 - min_cash_buffer))

    target_weight_gross = _to_float(weight_stats.get("target_weight_gross"))
    target_under_investment = None
    if investable_target is not None and target_weight_gross is not None:
        target_under_investment = float(max(0.0, investable_target - target_weight_gross))

    out: dict[str, Any] = {
        "summary_path": str(summary_path),
        "timestamp_utc": summary.get("timestamp_utc"),
        "rebalance_date": summary.get("rebalance_date"),
        "broker": summary.get("broker"),
        "mode": summary.get("mode"),
        "apply": summary.get("apply"),
        "portfolio_mode": summary.get("portfolio_mode"),
        "n_orders": summary.get("n_orders"),
        "portfolio_turnover_estimate": _to_float(summary.get("portfolio_turnover_estimate")),
        "account_equity_pre": equity,
        "account_cash_pre": cash,
        "pre_cash_fraction": pre_cash_fraction,
        "min_cash_buffer": min_cash_buffer,
        "investable_target_fraction": investable_target,
        "target_under_investment_fraction": target_under_investment,
        "weights_source": (summary.get("weights_source_meta", {}) or {}).get("weights_source"),
        "orders_rows": order_stats.get("orders_rows"),
        "planned_buy_notional": _to_float(order_stats.get("planned_buy_notional")),
        "planned_sell_notional": _to_float(order_stats.get("planned_sell_notional")),
        "planned_net_notional": planned_net_notional,
        "estimated_cash_after_orders": estimated_cash_after,
        "estimated_cash_after_fraction": estimated_cash_after_fraction,
    }
    out.update(weight_stats)
    out.update(run_all_stats)
    return out


def _build_aggregate_summary(history_df: pd.DataFrame) -> dict[str, Any]:
    if history_df.empty:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "empty",
            "n_rows": 0,
        }

    ordered = history_df.sort_values("timestamp_utc").reset_index(drop=True)
    first = ordered.iloc[0]
    last = ordered.iloc[-1]

    equity_start = _to_float(first.get("account_equity_pre"))
    equity_end = _to_float(last.get("account_equity_pre"))
    equity_change_abs = None
    equity_change_pct = None
    if equity_start is not None and equity_end is not None and abs(equity_start) > 1e-12:
        equity_change_abs = float(equity_end - equity_start)
        equity_change_pct = float((equity_end / equity_start) - 1.0)

    latest_reasons: list[str] = []
    latest_target_under = _to_float(last.get("target_under_investment_fraction"))
    if latest_target_under is not None and latest_target_under > 1e-6:
        latest_reasons.append("target_weights_below_investable_fraction")
    gate_multiplier = _to_float(last.get("run_all_signal_gate_multiplier"))
    if gate_multiplier is not None and gate_multiplier < 0.999999:
        latest_reasons.append("signal_quality_gate_de_risking")
    used_prev = last.get("run_all_used_existing_positions")
    if pd.notna(used_prev) and bool(used_prev):
        latest_reasons.append("live_recommendation_used_existing_positions")

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "ok",
        "n_rows": int(len(ordered)),
        "first_timestamp_utc": first.get("timestamp_utc"),
        "last_timestamp_utc": last.get("timestamp_utc"),
        "first_rebalance_date": first.get("rebalance_date"),
        "last_rebalance_date": last.get("rebalance_date"),
        "equity_start": equity_start,
        "equity_end": equity_end,
        "equity_change_abs": equity_change_abs,
        "equity_change_pct": equity_change_pct,
        "applied_rebalances": int((ordered["apply"] == True).sum()),
        "dry_run_rebalances": int((ordered["apply"] == False).sum()),
        "avg_orders_per_rebalance": float(pd.to_numeric(ordered["n_orders"], errors="coerce").mean()),
        "avg_target_weight_gross": float(pd.to_numeric(ordered["target_weight_gross"], errors="coerce").mean()),
        "latest_pre_cash_fraction": _to_float(last.get("pre_cash_fraction")),
        "latest_estimated_cash_after_fraction": _to_float(last.get("estimated_cash_after_fraction")),
        "latest_target_weight_gross": _to_float(last.get("target_weight_gross")),
        "latest_run_all_signal_gate_multiplier": gate_multiplier,
        "latest_run_all_turnover_cap_applied": last.get("run_all_turnover_cap_applied"),
        "latest_run_all_used_existing_positions": used_prev,
        "latest_possible_underinvestment_reasons": latest_reasons,
    }


def main() -> None:
    args = parse_args()

    execution_dir = args.execution_dir
    if not execution_dir.is_absolute():
        execution_dir = (PROJECT_ROOT / execution_dir).resolve()
    output_csv = args.output_csv
    if not output_csv.is_absolute():
        output_csv = (PROJECT_ROOT / output_csv).resolve()
    output_summary = args.output_summary
    if not output_summary.is_absolute():
        output_summary = (PROJECT_ROOT / output_summary).resolve()

    if not execution_dir.exists():
        raise FileNotFoundError(f"Missing execution directory: {execution_dir}")

    summary_paths = _list_rebalance_summary_paths(execution_dir)
    if not summary_paths:
        raise FileNotFoundError(
            f"No summary files found in {execution_dir}. Expected files like rebalance_YYYY-MM-DD_..._summary.json"
        )

    recommendation_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            _build_history_row(
                summary_path=summary_path,
                summary=payload,
                recommendation_cache=recommendation_cache,
            )
        )

    history_df = pd.DataFrame(rows)
    history_df["timestamp_utc"] = pd.to_datetime(history_df["timestamp_utc"], utc=True, errors="coerce")
    history_df = history_df.sort_values("timestamp_utc").reset_index(drop=True)
    history_df["timestamp_utc"] = history_df["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    history_df.to_csv(output_csv, index=False)
    summary_payload = _build_aggregate_summary(history_df=history_df)
    output_summary.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    print("Execution history report generated")
    print(f"Rows: {len(history_df)}")
    print(f"History CSV: {output_csv}")
    print(f"Summary JSON: {output_summary}")
    print(f"Latest timestamp: {summary_payload.get('last_timestamp_utc')}")
    print(f"Latest pre-cash fraction: {summary_payload.get('latest_pre_cash_fraction')}")
    print(f"Latest target gross weight: {summary_payload.get('latest_target_weight_gross')}")
    print(f"Latest signal gate multiplier: {summary_payload.get('latest_run_all_signal_gate_multiplier')}")
    print(f"Latest underinvestment reasons: {summary_payload.get('latest_possible_underinvestment_reasons')}")


if __name__ == "__main__":
    main()
