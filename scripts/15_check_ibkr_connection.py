from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_yaml
from src.execution.ibkr import IBKRBroker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for IBKR paper connection (no order placement).")
    parser.add_argument("--config-execution", type=Path, default=Path("configs/config_execution.yaml"))
    parser.add_argument(
        "--symbols",
        type=str,
        default="SPY,QQQ",
        help="Comma-separated tickers for market data check.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/execution/ibkr_connection_latest.json"),
        help="JSON output path for connection report.",
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


def _parse_symbols(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def main() -> None:
    args = parse_args()
    load_dotenv(dotenv_path=(PROJECT_ROOT / ".env"), override=False)

    config_execution = _resolve_env(load_yaml(args.config_execution.resolve()))
    ibkr_cfg = config_execution.get("ibkr", {})

    host = str(ibkr_cfg.get("host") or "127.0.0.1")
    port = int(ibkr_cfg.get("port") or 7497)
    client_id = int(ibkr_cfg.get("client_id") or 101)
    account = ibkr_cfg.get("account")
    readonly = _parse_bool(ibkr_cfg.get("readonly"), default=True)
    market_data_type = ibkr_cfg.get("market_data_type", "delayed")

    broker = IBKRBroker(
        host=host,
        port=port,
        client_id=client_id,
        account=account,
        readonly=readonly,
        market_data_type=market_data_type,
    )

    symbols = _parse_symbols(args.symbols)
    output_path = args.output if args.output.is_absolute() else (PROJECT_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    broker.connect()
    try:
        snapshot = broker.get_account_snapshot()
        if not symbols:
            symbols = sorted(snapshot.positions.keys())[:5]
        prices = broker.get_last_prices(symbols) if symbols else {}
    finally:
        broker.disconnect()

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "ok",
        "host": host,
        "port": port,
        "client_id": client_id,
        "account": account,
        "readonly": readonly,
        "market_data_type": market_data_type,
        "equity": float(snapshot.equity),
        "cash": float(snapshot.cash),
        "positions_count": int(len(snapshot.positions)),
        "checked_symbols": symbols,
        "prices": prices,
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("IBKR connection check passed")
    print(f"Endpoint: {host}:{port} (client_id={client_id})")
    print(f"Account: {account or '<auto>'}")
    print(f"Readonly: {readonly}")
    print(f"Equity: {snapshot.equity:,.2f}")
    print(f"Cash: {snapshot.cash:,.2f}")
    print(f"Positions: {len(snapshot.positions)}")
    print(f"Prices resolved: {len(prices)}/{len(symbols)}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
