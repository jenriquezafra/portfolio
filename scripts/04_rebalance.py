from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_yaml
from src.execution.broker_base import AccountSnapshot, OrderRequest
from src.execution.ibkr import IBKRBroker
from src.execution.paper import PaperBroker
from src.optimizer import apply_turnover_cap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and optionally execute rebalance orders.")
    parser.add_argument("--config-data", type=Path, default=Path("configs/config_data.yaml"))
    parser.add_argument("--config-backtest", type=Path, default=Path("configs/config_backtest.yaml"))
    parser.add_argument("--config-execution", type=Path, default=Path("configs/config_execution.yaml"))
    parser.add_argument(
        "--weights-source",
        type=str,
        default="backtest",
        choices=["backtest", "run_all"],
        help="Source for target weights: backtest history or latest run_all live recommendation.",
    )
    parser.add_argument(
        "--recommendation-path",
        type=Path,
        default=Path("outputs/run_all/recommendation.json"),
        help="Used when --weights-source=run_all.",
    )
    parser.add_argument(
        "--weights-csv",
        type=Path,
        default=None,
        help="Optional explicit weights CSV path (overrides recommendation artifact path).",
    )
    parser.add_argument(
        "--max-signal-age-business-days",
        type=int,
        default=3,
        help="Reject run_all weights if live signal is older than this many business days vs latest clean prices date.",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Use a specific rebalance date from weights_history (YYYY-MM-DD). Default: latest available.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="If provided, sends orders to the configured broker. Default is dry-run.",
    )
    return parser.parse_args()


def _resolve_env(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _resolve_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        var = obj[2:-1]
        return os.getenv(var)
    return obj


def _parse_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off", ""}:
            return False
        raise ValueError(f"Invalid boolean value: {value!r}")
    return bool(value)


def _parse_order_type(value: Any) -> str:
    normalized = str(value or "MKT").strip().upper()
    if normalized not in {"MKT", "LMT"}:
        raise ValueError(f"Unsupported `execution.order_type`: {value!r}. Use `MKT` or `LMT`.")
    return normalized


def _compute_limit_price(
    *,
    side: str,
    reference_price: float,
    limit_price_offset_bps: float,
    limit_price_round_decimals: int,
) -> float:
    if reference_price <= 0:
        raise ValueError(f"Cannot compute limit price from non-positive reference price: {reference_price}")
    offset = float(limit_price_offset_bps) / 10000.0
    if side == "BUY":
        price = float(reference_price) * (1.0 + offset)
    elif side == "SELL":
        price = float(reference_price) * (1.0 - offset)
    else:
        raise ValueError(f"Unsupported side for limit price: {side!r}")
    rounded = round(float(price), int(limit_price_round_decimals))
    if rounded <= 0:
        raise ValueError(
            "Computed non-positive limit price. "
            f"reference={reference_price} offset_bps={limit_price_offset_bps} side={side}"
        )
    return float(rounded)


def _load_target_weights(weights_path: Path, as_of_date: str | None) -> tuple[pd.Timestamp, pd.Series]:
    weights = pd.read_parquet(weights_path)
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"], utc=False).dt.tz_localize(None)
    weights["ticker"] = weights["ticker"].astype(str)

    if as_of_date is None:
        target_date = pd.Timestamp(weights["rebalance_date"].max())
    else:
        target_date = pd.Timestamp(as_of_date).tz_localize(None)
        if target_date not in set(weights["rebalance_date"]):
            available = weights["rebalance_date"].drop_duplicates().sort_values()
            prior = available[available <= target_date]
            if prior.empty:
                raise ValueError(f"No rebalance_date <= {target_date.date()} found in weights history.")
            target_date = pd.Timestamp(prior.iloc[-1])

    target = (
        weights[weights["rebalance_date"] == target_date]
        .set_index("ticker")["weight"]
        .astype(float)
        .sort_index()
    )
    if target.empty:
        raise ValueError(f"No target weights found for {target_date.date()}.")
    return target_date, target


def _business_days_between(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end <= start:
        return 0
    return max(0, int(len(pd.bdate_range(start=start, end=end)) - 1))


def _load_target_weights_from_run_all(
    recommendation_path: Path,
    weights_csv: Path | None,
) -> tuple[pd.Timestamp, pd.Series, dict[str, Any]]:
    if not recommendation_path.exists():
        raise FileNotFoundError(f"Missing recommendation file: {recommendation_path}")
    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))

    signal_date_raw = recommendation.get("live_signal_date")
    if signal_date_raw is None:
        raise ValueError(
            "Recommendation file does not include `live_signal_date`. "
            "Re-run scripts/06_run_all.py to generate live recommendation artifacts."
        )
    signal_date = pd.Timestamp(signal_date_raw).tz_localize(None)

    if weights_csv is None:
        artifacts = recommendation.get("artifacts", {})
        if not isinstance(artifacts, dict):
            raise ValueError("Invalid `artifacts` block in recommendation file.")
        csv_raw = artifacts.get("recommended_weights_csv")
        if not isinstance(csv_raw, str):
            raise ValueError("Recommendation file missing `artifacts.recommended_weights_csv`.")
        weights_csv = Path(csv_raw)
    if not weights_csv.is_absolute():
        weights_csv = (PROJECT_ROOT / weights_csv).resolve()
    if not weights_csv.exists():
        raise FileNotFoundError(f"Missing weights CSV: {weights_csv}")

    weights_df = pd.read_csv(weights_csv)
    if "ticker" not in weights_df.columns or "weight" not in weights_df.columns:
        raise ValueError("Weights CSV must contain `ticker` and `weight` columns.")
    target = weights_df[["ticker", "weight"]].copy()
    target["ticker"] = target["ticker"].astype(str)
    target["weight"] = pd.to_numeric(target["weight"], errors="coerce")
    target = target.dropna(subset=["weight"]).set_index("ticker")["weight"].astype(float).sort_index()
    if target.empty:
        raise ValueError("Weights CSV resolved to an empty target vector.")

    meta = {
        "weights_source": "run_all",
        "recommendation_path": str(recommendation_path),
        "weights_csv": str(weights_csv),
        "recommended_strategy": recommendation.get("recommended_strategy"),
        "live_signal_date": str(signal_date.date()),
        "backtest_rebalance_date": recommendation.get("rebalance_date"),
    }
    return signal_date, target, meta


def _snapshot_to_series(snapshot: AccountSnapshot) -> pd.Series:
    if not snapshot.positions:
        return pd.Series(dtype=float)
    return pd.Series({sym: float(pos.quantity) for sym, pos in snapshot.positions.items()}, dtype=float)


def _compute_current_weights(
    positions_qty: pd.Series,
    prices: dict[str, float],
    equity: float,
    symbols: list[str],
) -> pd.Series:
    if equity <= 0:
        return pd.Series(0.0, index=symbols, dtype=float)
    values = pd.Series({s: float(positions_qty.get(s, 0.0)) * float(prices.get(s, 0.0)) for s in symbols}, dtype=float)
    return (values / float(equity)).fillna(0.0)


def _effective_turnover_cap(
    max_turnover: float | None,
    current_qty: pd.Series,
) -> tuple[float | None, bool]:
    if max_turnover is None:
        return None, False
    qty = pd.to_numeric(current_qty, errors="coerce").fillna(0.0)
    has_open_positions = bool((qty.abs() > 1e-12).any())
    if not has_open_positions:
        return None, False
    return float(max_turnover), True


def _validate_weights(weights: pd.Series, max_position_weight: float, max_gross_exposure: float) -> None:
    return _validate_weights_with_mode(
        weights=weights,
        max_position_weight=max_position_weight,
        max_gross_exposure=max_gross_exposure,
        allow_short=False,
        max_net_exposure=1.0,
    )


def _validate_weights_with_mode(
    weights: pd.Series,
    max_position_weight: float,
    max_gross_exposure: float,
    allow_short: bool,
    max_net_exposure: float,
) -> None:
    if allow_short:
        if float(weights.abs().max()) > float(max_position_weight) + 1e-9:
            raise ValueError(
                f"Target abs(weight) exceeds max_position_weight={max_position_weight:.4f}. "
                f"Max found: {float(weights.abs().max()):.4f}"
            )
        if abs(float(weights.sum())) > float(max_net_exposure) + 1e-9:
            raise ValueError(
                f"Net exposure {float(weights.sum()):.4f} exceeds max_net_exposure={max_net_exposure:.4f}"
            )
    else:
        if (weights < -1e-12).any():
            bad = weights[weights < -1e-12]
            raise ValueError(f"Negative target weights are not allowed: {bad.to_dict()}")
        if float(weights.max()) > float(max_position_weight) + 1e-9:
            raise ValueError(
                f"Target weight exceeds max_position_weight={max_position_weight:.4f}. "
                f"Max found: {float(weights.max()):.4f}"
            )

    gross = float(weights.abs().sum())
    if gross > float(max_gross_exposure) + 1e-9:
        raise ValueError(f"Gross exposure {gross:.4f} exceeds max_gross_exposure={max_gross_exposure:.4f}")


def _prepare_target_weights(
    target_weights_raw: pd.Series,
    symbols: list[str],
    portfolio_mode: str,
    min_cash_buffer: float,
) -> pd.Series:
    target = target_weights_raw.reindex(symbols).fillna(0.0).astype(float)
    investable_fraction = max(0.0, 1.0 - float(min_cash_buffer))

    if portfolio_mode == "long_only":
        target = target.clip(lower=0.0)
        total = float(target.sum())
        if total > investable_fraction + 1e-12 and total > 1e-12:
            # Only scale down when requested longs exceed allowed invested fraction.
            # Keep partial-cash recommendations unchanged to avoid unintended concentration.
            target = target * (investable_fraction / total)
        return target

    if portfolio_mode == "market_neutral":
        # Keep signed exposures from backtest and only scale if cash buffer is requested.
        return target * investable_fraction

    raise ValueError("Unsupported portfolio mode. Expected `long_only` or `market_neutral`.")


def _compute_target_shares(
    symbols: list[str],
    target_weights: pd.Series,
    prices: dict[str, float],
    equity: float,
) -> pd.Series:
    target_shares = {}
    for sym in symbols:
        w = float(target_weights.get(sym, 0.0))
        px = float(prices[sym])
        if px <= 0:
            raise ValueError(f"Invalid non-positive price for {sym}: {px}")
        target_shares[sym] = int((w * equity) / px)
    return pd.Series(target_shares, dtype=float)


def _make_order_requests(
    current_qty: pd.Series,
    target_qty: pd.Series,
    prices: dict[str, float],
    order_type: str,
    tif: str,
    limit_price_offset_bps: float,
    limit_price_round_decimals: int,
) -> tuple[pd.DataFrame, list[OrderRequest]]:
    symbols = sorted(set(current_qty.index).union(set(target_qty.index)))
    rows: list[dict[str, Any]] = []
    requests: list[OrderRequest] = []
    order_type_normalized = _parse_order_type(order_type)

    for sym in symbols:
        cur = int(round(float(current_qty.get(sym, 0.0))))
        tgt = int(round(float(target_qty.get(sym, 0.0))))
        delta = tgt - cur
        if delta == 0:
            continue
        side = "BUY" if delta > 0 else "SELL"
        limit_price: float | None = None
        if order_type_normalized == "LMT":
            if sym not in prices:
                raise ValueError(f"Missing price for limit order symbol: {sym}")
            limit_price = _compute_limit_price(
                side=side,
                reference_price=float(prices[sym]),
                limit_price_offset_bps=limit_price_offset_bps,
                limit_price_round_decimals=limit_price_round_decimals,
            )
        rows.append(
            {
                "ticker": sym,
                "current_qty": cur,
                "target_qty": tgt,
                "delta_qty": delta,
                "side": side,
                "order_type": order_type_normalized,
                "tif": tif,
                "limit_price": limit_price,
            }
        )
        requests.append(
            OrderRequest(
                symbol=sym,
                quantity=int(delta),
                order_type=order_type_normalized,
                tif=tif,
                limit_price=limit_price,
            )
        )

    orders_df = pd.DataFrame(rows).sort_values(["side", "ticker"]).reset_index(drop=True) if rows else pd.DataFrame()
    return orders_df, requests


def _build_broker(execution_cfg: dict, project_root: Path, as_of_date: pd.Timestamp | None):
    exec_section = execution_cfg.get("execution", {})
    broker_name = str(exec_section.get("broker", "paper")).lower()

    if broker_name == "paper":
        paper_cfg = execution_cfg.get("paper", {})
        state_rel = paper_cfg.get("state_path", "outputs/execution/paper_state.json")
        prices_rel = paper_cfg.get("prices_path", "data/processed/prices_clean.parquet")
        initial_cash = float(paper_cfg.get("initial_cash", 100000))
        return PaperBroker(
            state_path=(project_root / str(state_rel)).resolve(),
            prices_path=(project_root / str(prices_rel)).resolve(),
            initial_cash=initial_cash,
            as_of_date=as_of_date,
        )

    if broker_name == "ibkr":
        ibkr_cfg = execution_cfg.get("ibkr", {})
        host = str(ibkr_cfg.get("host") or "127.0.0.1")
        port = int(ibkr_cfg.get("port") or 7497)
        client_id = int(ibkr_cfg.get("client_id") or 101)
        account = ibkr_cfg.get("account")
        readonly = _parse_bool(ibkr_cfg.get("readonly"), default=True)
        market_data_type = ibkr_cfg.get("market_data_type", "delayed")
        return IBKRBroker(
            host=host,
            port=port,
            client_id=client_id,
            account=account,
            readonly=readonly,
            market_data_type=market_data_type,
        )

    raise ValueError(f"Unsupported broker: {broker_name}")


def main() -> None:
    args = parse_args()
    load_dotenv(dotenv_path=(PROJECT_ROOT / ".env"), override=False)

    config_data = load_yaml(args.config_data.resolve())
    config_backtest = load_yaml(args.config_backtest.resolve())
    config_execution = _resolve_env(load_yaml(args.config_execution.resolve()))

    execution_section = config_execution.get("execution", {})
    risk_controls = config_execution.get("risk_controls", {})
    backtest_section = config_backtest.get("backtest", {})
    if not isinstance(backtest_section, dict):
        raise ValueError("Missing `backtest` section in config_backtest.yaml")
    portfolio_cfg = backtest_section.get("portfolio", {})
    if not isinstance(portfolio_cfg, dict):
        raise ValueError("`backtest.portfolio` must be a mapping.")
    portfolio_mode = str(portfolio_cfg.get("mode", "long_only")).lower()
    allow_shorting = portfolio_mode == "market_neutral"

    order_type = _parse_order_type(execution_section.get("order_type", "MKT"))
    tif = str(execution_section.get("tif", "DAY"))
    limit_price_offset_bps = float(execution_section.get("limit_price_offset_bps", 10.0))
    limit_price_round_decimals = int(execution_section.get("limit_price_round_decimals", 2))
    if limit_price_offset_bps < 0:
        raise ValueError("`execution.limit_price_offset_bps` must be >= 0.")
    if limit_price_round_decimals < 0:
        raise ValueError("`execution.limit_price_round_decimals` must be >= 0.")
    min_cash_buffer = float(risk_controls.get("min_cash_buffer", 0.0))
    max_turnover = risk_controls.get("max_turnover_per_rebalance")
    max_turnover = None if max_turnover is None else float(max_turnover)
    max_position_weight = float(risk_controls.get("max_position_weight", 1.0))
    max_gross_exposure = float(risk_controls.get("max_gross_exposure", 1.0))
    max_net_exposure = float(risk_controls.get("max_net_exposure", 1.0))
    reject_if_missing_prices = _parse_bool(risk_controls.get("reject_if_missing_prices"), default=True)
    kill_switch_enabled = _parse_bool(risk_controls.get("kill_switch_enabled"), default=True)

    as_of_date = None if args.as_of_date is None else pd.Timestamp(args.as_of_date).tz_localize(None)
    if args.max_signal_age_business_days <= 0:
        raise ValueError("`--max-signal-age-business-days` must be positive.")

    weights_source_meta: dict[str, Any] = {}
    if args.weights_source == "backtest":
        weights_path = (PROJECT_ROOT / "outputs/backtests/weights_history.parquet").resolve()
        rebalance_date, target_weights_raw = _load_target_weights(weights_path=weights_path, as_of_date=args.as_of_date)
        weights_source_meta = {
            "weights_source": "backtest",
            "weights_history_path": str(weights_path),
        }
    else:
        if args.as_of_date is not None:
            raise ValueError("`--as-of-date` is only supported with `--weights-source=backtest`.")
        explicit_csv = None if args.weights_csv is None else args.weights_csv.resolve()
        recommendation_path = args.recommendation_path.resolve()
        rebalance_date, target_weights_raw, weights_source_meta = _load_target_weights_from_run_all(
            recommendation_path=recommendation_path,
            weights_csv=explicit_csv,
        )

        data_section = config_data.get("data", {})
        clean_path_cfg = data_section.get("output_clean_path") if isinstance(data_section, dict) else None
        if isinstance(clean_path_cfg, str):
            clean_path = (PROJECT_ROOT / clean_path_cfg).resolve()
            if clean_path.exists():
                clean_dates = pd.read_parquet(clean_path, columns=["date"])
                clean_dates["date"] = pd.to_datetime(clean_dates["date"], utc=False).dt.tz_localize(None)
                latest_clean_date = pd.Timestamp(clean_dates["date"].max())
                signal_age_bdays = _business_days_between(
                    start_date=rebalance_date,
                    end_date=latest_clean_date,
                )
                weights_source_meta["latest_clean_prices_date"] = str(latest_clean_date.date())
                weights_source_meta["signal_age_business_days"] = int(signal_age_bdays)
                if signal_age_bdays > int(args.max_signal_age_business_days):
                    raise ValueError(
                        "Live signal is stale for paper rebalance: "
                        f"signal_date={rebalance_date.date()} latest_clean_date={latest_clean_date.date()} "
                        f"age_business_days={signal_age_bdays} max_allowed={args.max_signal_age_business_days}."
                    )

    broker = _build_broker(config_execution, project_root=PROJECT_ROOT, as_of_date=as_of_date or rebalance_date)
    broker.connect()
    try:
        snapshot = broker.get_account_snapshot()
        symbols = sorted(set(target_weights_raw.index.astype(str)).union(set(snapshot.positions.keys())))
        prices = broker.get_last_prices(symbols)

        missing_symbols = [s for s in symbols if s not in prices]
        if missing_symbols and reject_if_missing_prices:
            missing_target = sorted([s for s in missing_symbols if s in set(target_weights_raw.index.astype(str))])
            if missing_target:
                raise ValueError(
                    "Missing prices for symbols present in target weights: "
                    f"{missing_target}. This usually means `outputs/backtests/weights_history.parquet` "
                    "was generated with a different universe than the current prices data. "
                    "Re-run scripts/03_backtest.py after updating data/panel/train."
                )
            raise ValueError(f"Missing prices for symbols: {missing_symbols}")

        symbols = [s for s in symbols if s in prices]
        if not symbols:
            raise ValueError("No symbols with valid prices were available for rebalance.")

        current_qty = _snapshot_to_series(snapshot).reindex(symbols).fillna(0.0)
        current_weights = _compute_current_weights(
            positions_qty=current_qty,
            prices=prices,
            equity=float(snapshot.equity),
            symbols=symbols,
        )
        effective_max_turnover, turnover_cap_applied = _effective_turnover_cap(
            max_turnover=max_turnover,
            current_qty=current_qty,
        )
        if max_turnover is not None:
            weights_source_meta["turnover_cap_configured"] = float(max_turnover)
            weights_source_meta["turnover_cap_effective"] = (
                None if effective_max_turnover is None else float(effective_max_turnover)
            )
            weights_source_meta["turnover_cap_applied"] = bool(turnover_cap_applied)
            if not turnover_cap_applied:
                weights_source_meta["turnover_cap_skipped_reason"] = "no_open_positions"

        target_weights = _prepare_target_weights(
            target_weights_raw=target_weights_raw,
            symbols=symbols,
            portfolio_mode=portfolio_mode,
            min_cash_buffer=min_cash_buffer,
        )
        target_weights, turnover = apply_turnover_cap(
            target_weights=target_weights,
            prev_weights=current_weights,
            max_turnover_per_rebalance=effective_max_turnover,
        )
        _validate_weights_with_mode(
            weights=target_weights,
            max_position_weight=max_position_weight,
            max_gross_exposure=max_gross_exposure,
            allow_short=allow_shorting,
            max_net_exposure=max_net_exposure,
        )

        target_qty = _compute_target_shares(
            symbols=symbols,
            target_weights=target_weights,
            prices=prices,
            equity=float(snapshot.equity),
        )

        orders_df, requests = _make_order_requests(
            current_qty=current_qty,
            target_qty=target_qty,
            prices=prices,
            order_type=order_type,
            tif=tif,
            limit_price_offset_bps=limit_price_offset_bps,
            limit_price_round_decimals=limit_price_round_decimals,
        )

        if not orders_df.empty:
            orders_df["price"] = orders_df["ticker"].map(prices)
            orders_df["trade_notional"] = orders_df["delta_qty"] * orders_df["price"]
            orders_df["current_weight"] = orders_df["ticker"].map(current_weights)
            orders_df["target_weight"] = orders_df["ticker"].map(target_weights)
            orders_df["abs_weight_change"] = (orders_df["target_weight"] - orders_df["current_weight"]).abs()

        exec_dir = (PROJECT_ROOT / "outputs/execution").resolve()
        exec_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        mode = str(execution_section.get("mode", "paper"))
        broker_name = str(execution_section.get("broker", "paper"))
        prefix = f"rebalance_{rebalance_date.date()}_{ts}"
        orders_path = exec_dir / f"{prefix}_orders.csv"
        summary_path = exec_dir / f"{prefix}_summary.json"
        latest_orders_path = exec_dir / "rebalance_latest_orders.csv"
        latest_summary_path = exec_dir / "rebalance_latest_summary.json"

        if not orders_df.empty:
            orders_df.to_csv(orders_path, index=False)
            orders_df.to_csv(latest_orders_path, index=False)
        else:
            pd.DataFrame(columns=["ticker", "delta_qty"]).to_csv(orders_path, index=False)
            pd.DataFrame(columns=["ticker", "delta_qty"]).to_csv(latest_orders_path, index=False)

        can_apply = bool(args.apply)
        if can_apply and kill_switch_enabled and os.getenv("KILL_SWITCH", "0").strip() == "1":
            raise RuntimeError("Kill-switch is enabled via env KILL_SWITCH=1. Refusing to send orders.")

        if can_apply and broker_name == "ibkr":
            ibkr_cfg = config_execution.get("ibkr", {})
            if _parse_bool(ibkr_cfg.get("readonly"), default=True):
                raise RuntimeError("IBKR is readonly=true. Set readonly=false to allow order submission.")

        broker_ids: list[str] = []
        if can_apply and requests:
            broker_ids = broker.place_orders(requests)

        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": mode,
            "broker": broker_name,
            "apply": bool(can_apply),
            "rebalance_date": str(rebalance_date.date()),
            "account_equity": float(snapshot.equity),
            "account_cash": float(snapshot.cash),
            "n_symbols": int(len(symbols)),
            "n_orders": int(len(requests)),
            "portfolio_turnover_estimate": float(turnover),
            "max_position_weight": max_position_weight,
            "max_gross_exposure": max_gross_exposure,
            "max_net_exposure": max_net_exposure,
            "min_cash_buffer": min_cash_buffer,
            "portfolio_mode": portfolio_mode,
            "allow_shorting": allow_shorting,
            "order_type": order_type,
            "tif": tif,
            "limit_price_offset_bps": limit_price_offset_bps,
            "limit_price_round_decimals": limit_price_round_decimals,
            "orders_file": str(orders_path),
            "orders_file_latest": str(latest_orders_path),
            "broker_order_ids": broker_ids,
            "weights_source_meta": weights_source_meta,
        }
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        with latest_summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)

        print("Rebalance plan ready")
        print(f"Mode/Broker: {mode}/{broker_name}")
        print(f"Portfolio mode: {portfolio_mode} (allow shorting={allow_shorting})")
        print(f"Apply orders: {can_apply}")
        print(f"Rebalance date used: {rebalance_date.date()}")
        print(f"Equity: {snapshot.equity:,.2f}")
        print(
            "Turnover cap effective: "
            f"{'none' if effective_max_turnover is None else f'{effective_max_turnover:.4f}'}"
        )
        print(f"Estimated turnover: {turnover:.6f}")
        print(f"Orders: {len(requests)}")
        print(f"Orders file: {orders_path}")
        print(f"Latest orders file: {latest_orders_path}")
        print(f"Summary file: {summary_path}")
        print(f"Latest summary file: {latest_summary_path}")
        if not orders_df.empty:
            print("Top order changes:")
            top = orders_df.sort_values("abs_weight_change", ascending=False).head(10)
            for _, row in top.iterrows():
                print(
                    f"  - {row['ticker']}: {row['side']} {int(row['delta_qty'])} @ {row['price']:.2f} "
                    f"(w {row['current_weight']:.4f} -> {row['target_weight']:.4f})"
                )
    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
