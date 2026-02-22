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


def _validate_weights(weights: pd.Series, max_position_weight: float, max_gross_exposure: float) -> None:
    if (weights < -1e-12).any():
        bad = weights[weights < -1e-12]
        raise ValueError(f"Negative target weights are not allowed: {bad.to_dict()}")

    if float(weights.max()) > float(max_position_weight) + 1e-9:
        raise ValueError(
            f"Target weight exceeds max_position_weight={max_position_weight:.4f}. Max found: {float(weights.max()):.4f}"
        )

    gross = float(weights.abs().sum())
    if gross > float(max_gross_exposure) + 1e-9:
        raise ValueError(f"Gross exposure {gross:.4f} exceeds max_gross_exposure={max_gross_exposure:.4f}")


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
    order_type: str,
    tif: str,
) -> tuple[pd.DataFrame, list[OrderRequest]]:
    symbols = sorted(set(current_qty.index).union(set(target_qty.index)))
    rows: list[dict[str, Any]] = []
    requests: list[OrderRequest] = []

    for sym in symbols:
        cur = int(round(float(current_qty.get(sym, 0.0))))
        tgt = int(round(float(target_qty.get(sym, 0.0))))
        delta = tgt - cur
        if delta == 0:
            continue
        side = "BUY" if delta > 0 else "SELL"
        rows.append(
            {
                "ticker": sym,
                "current_qty": cur,
                "target_qty": tgt,
                "delta_qty": delta,
                "side": side,
                "order_type": order_type,
                "tif": tif,
            }
        )
        requests.append(OrderRequest(symbol=sym, quantity=int(delta), order_type=order_type, tif=tif))

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
        readonly = bool(ibkr_cfg.get("readonly", True))
        return IBKRBroker(host=host, port=port, client_id=client_id, account=account, readonly=readonly)

    raise ValueError(f"Unsupported broker: {broker_name}")


def main() -> None:
    args = parse_args()
    load_dotenv(dotenv_path=(PROJECT_ROOT / ".env"), override=False)

    config_data = load_yaml(args.config_data.resolve())
    config_backtest = load_yaml(args.config_backtest.resolve())
    config_execution = _resolve_env(load_yaml(args.config_execution.resolve()))

    execution_section = config_execution.get("execution", {})
    risk_controls = config_execution.get("risk_controls", {})

    order_type = str(execution_section.get("order_type", "MKT"))
    tif = str(execution_section.get("tif", "DAY"))
    min_cash_buffer = float(risk_controls.get("min_cash_buffer", 0.0))
    max_turnover = risk_controls.get("max_turnover_per_rebalance")
    max_turnover = None if max_turnover is None else float(max_turnover)
    max_position_weight = float(risk_controls.get("max_position_weight", 1.0))
    max_gross_exposure = float(risk_controls.get("max_gross_exposure", 1.0))
    reject_if_missing_prices = bool(risk_controls.get("reject_if_missing_prices", True))
    kill_switch_enabled = bool(risk_controls.get("kill_switch_enabled", True))

    as_of_date = None if args.as_of_date is None else pd.Timestamp(args.as_of_date).tz_localize(None)
    weights_path = (PROJECT_ROOT / "outputs/backtests/weights_history.parquet").resolve()
    rebalance_date, target_weights_raw = _load_target_weights(weights_path=weights_path, as_of_date=args.as_of_date)

    broker = _build_broker(config_execution, project_root=PROJECT_ROOT, as_of_date=as_of_date or rebalance_date)
    broker.connect()
    try:
        snapshot = broker.get_account_snapshot()
        symbols = sorted(set(target_weights_raw.index.astype(str)).union(set(snapshot.positions.keys())))
        prices = broker.get_last_prices(symbols)

        missing_symbols = [s for s in symbols if s not in prices]
        if missing_symbols and reject_if_missing_prices:
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

        target_weights = target_weights_raw.reindex(symbols).fillna(0.0)
        if target_weights.sum() > 0:
            target_weights = target_weights / target_weights.sum()

        investable_fraction = max(0.0, 1.0 - min_cash_buffer)
        target_weights = target_weights * investable_fraction
        target_weights, turnover = apply_turnover_cap(
            target_weights=target_weights,
            prev_weights=current_weights,
            max_turnover_per_rebalance=max_turnover,
        )
        _validate_weights(target_weights, max_position_weight=max_position_weight, max_gross_exposure=max_gross_exposure)

        target_qty = _compute_target_shares(
            symbols=symbols,
            target_weights=target_weights,
            prices=prices,
            equity=float(snapshot.equity),
        )

        orders_df, requests = _make_order_requests(
            current_qty=current_qty,
            target_qty=target_qty,
            order_type=order_type,
            tif=tif,
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

        if not orders_df.empty:
            orders_df.to_csv(orders_path, index=False)
        else:
            pd.DataFrame(columns=["ticker", "delta_qty"]).to_csv(orders_path, index=False)

        can_apply = bool(args.apply)
        if can_apply and kill_switch_enabled and os.getenv("KILL_SWITCH", "0").strip() == "1":
            raise RuntimeError("Kill-switch is enabled via env KILL_SWITCH=1. Refusing to send orders.")

        if can_apply and broker_name == "ibkr":
            ibkr_cfg = config_execution.get("ibkr", {})
            if bool(ibkr_cfg.get("readonly", True)):
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
            "min_cash_buffer": min_cash_buffer,
            "orders_file": str(orders_path),
            "broker_order_ids": broker_ids,
        }
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)

        print("Rebalance plan ready")
        print(f"Mode/Broker: {mode}/{broker_name}")
        print(f"Apply orders: {can_apply}")
        print(f"Rebalance date used: {rebalance_date.date()}")
        print(f"Equity: {snapshot.equity:,.2f}")
        print(f"Estimated turnover: {turnover:.6f}")
        print(f"Orders: {len(requests)}")
        print(f"Orders file: {orders_path}")
        print(f"Summary file: {summary_path}")
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
