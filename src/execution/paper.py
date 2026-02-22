from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.execution.broker_base import AccountSnapshot, BrokerBase, OrderRequest, Position


class PaperBroker(BrokerBase):
    def __init__(
        self,
        state_path: Path,
        prices_path: Path,
        initial_cash: float = 100000.0,
        as_of_date: pd.Timestamp | None = None,
    ) -> None:
        self.state_path = state_path
        self.prices_path = prices_path
        self.initial_cash = float(initial_cash)
        self.as_of_date = None if as_of_date is None else pd.Timestamp(as_of_date).tz_localize(None)
        self._connected = False
        self._prices_df: pd.DataFrame | None = None

    def connect(self) -> None:
        self._connected = True
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self._write_state({"cash": self.initial_cash, "positions": {}, "last_updated": None})

    def disconnect(self) -> None:
        self._connected = False

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Paper broker is not connected.")

    def _load_prices(self) -> pd.DataFrame:
        if self._prices_df is None:
            df = pd.read_parquet(self.prices_path)
            df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(None)
            df["ticker"] = df["ticker"].astype(str)
            df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
            self._prices_df = df
        return self._prices_df

    def _write_state(self, state: dict) -> None:
        with self.state_path.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)

    def _read_state(self) -> dict:
        if not self.state_path.exists():
            return {"cash": self.initial_cash, "positions": {}, "last_updated": None}
        with self.state_path.open("r", encoding="utf-8") as fh:
            state = json.load(fh)
        return state

    def _effective_date(self, date: pd.Timestamp | None = None) -> pd.Timestamp:
        if date is not None:
            return pd.Timestamp(date).tz_localize(None)
        if self.as_of_date is not None:
            return self.as_of_date
        prices = self._load_prices()
        return pd.Timestamp(prices["date"].max())

    def get_last_prices(self, symbols: list[str]) -> dict[str, float]:
        self._ensure_connected()
        symbols = sorted({str(s) for s in symbols})
        if not symbols:
            return {}

        df = self._load_prices()
        as_of = self._effective_date()

        filtered = df[(df["ticker"].isin(symbols)) & (df["date"] <= as_of)]
        if filtered.empty:
            return {}

        last = (
            filtered.sort_values(["ticker", "date"])
            .groupby("ticker", as_index=False)
            .tail(1)[["ticker", "adj_close"]]
        )
        prices = {row["ticker"]: float(row["adj_close"]) for _, row in last.iterrows() if float(row["adj_close"]) > 0}

        # Fallback for symbols held in paper state but no longer present in prices parquet
        # (e.g., universe changed). Use avg_cost only to allow position unwind.
        missing = [sym for sym in symbols if sym not in prices]
        if missing:
            state = self._read_state()
            pos_raw = state.get("positions", {})
            for sym in missing:
                pos = pos_raw.get(sym)
                if not isinstance(pos, dict):
                    continue
                qty = float(pos.get("quantity", 0.0) or 0.0)
                avg_cost = pos.get("avg_cost")
                if abs(qty) <= 1e-12 or avg_cost is None:
                    continue
                px = float(avg_cost)
                if px > 0:
                    prices[sym] = px
        return prices

    def get_account_snapshot(self) -> AccountSnapshot:
        self._ensure_connected()
        state = self._read_state()
        cash = float(state.get("cash", self.initial_cash))
        pos_raw = state.get("positions", {})
        positions: dict[str, Position] = {}
        symbols = []
        for sym, data in pos_raw.items():
            qty = float(data.get("quantity", 0.0))
            avg_cost = data.get("avg_cost")
            if abs(qty) > 0:
                positions[str(sym)] = Position(symbol=str(sym), quantity=qty, avg_cost=avg_cost)
                symbols.append(str(sym))

        prices = self.get_last_prices(symbols) if symbols else {}
        market_value = sum(pos.quantity * prices.get(sym, 0.0) for sym, pos in positions.items())
        equity = cash + market_value
        return AccountSnapshot(equity=float(equity), cash=float(cash), positions=positions)

    def place_orders(self, orders: list[OrderRequest]) -> list[str]:
        self._ensure_connected()
        if not orders:
            return []

        state = self._read_state()
        cash = float(state.get("cash", self.initial_cash))
        positions = state.get("positions", {})

        symbols = sorted({order.symbol for order in orders})
        prices = self.get_last_prices(symbols)
        missing = [s for s in symbols if s not in prices]
        if missing:
            raise ValueError(f"Missing paper prices for symbols: {missing}")

        broker_ids: list[str] = []
        for order in orders:
            qty = int(order.quantity)
            if qty == 0:
                continue

            symbol = str(order.symbol)
            price = float(prices[symbol])
            notional = qty * price

            existing = positions.get(symbol, {"quantity": 0.0, "avg_cost": 0.0})
            prev_qty = float(existing.get("quantity", 0.0))
            prev_avg = float(existing.get("avg_cost", 0.0))
            new_qty = prev_qty + qty

            if qty > 0 and cash < notional:
                raise ValueError(f"Not enough cash to buy {qty} {symbol} at {price:.2f}.")

            if abs(new_qty) < 1e-10:
                positions.pop(symbol, None)
            else:
                if prev_qty >= 0 and qty > 0:
                    new_avg = ((prev_qty * prev_avg) + (qty * price)) / new_qty if new_qty != 0 else 0.0
                elif prev_qty > 0 and qty < 0:
                    new_avg = prev_avg
                else:
                    new_avg = price
                positions[symbol] = {"quantity": float(new_qty), "avg_cost": float(new_avg)}

            cash -= notional
            broker_ids.append(f"paper-{uuid4().hex[:10]}")

        new_state = {
            "cash": float(cash),
            "positions": positions,
            "last_updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._write_state(new_state)
        return broker_ids
