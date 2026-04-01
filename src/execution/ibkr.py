from __future__ import annotations

from src.execution.broker_base import AccountSnapshot, BrokerBase, OrderRequest, Position


def _parse_bool(value: bool | str | int | float | None, *, default: bool) -> bool:
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


def _parse_market_data_type(value: int | str | None, *, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, int):
        if value in {1, 2, 3, 4}:
            return int(value)
        raise ValueError(f"Invalid IBKR market_data_type: {value!r}. Expected one of: 1,2,3,4.")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized.isdigit():
            parsed = int(normalized)
            if parsed in {1, 2, 3, 4}:
                return parsed
        mapping = {
            "live": 1,
            "frozen": 2,
            "delayed": 3,
            "delayed_frozen": 4,
            "delayed-frozen": 4,
        }
        if normalized in mapping:
            return mapping[normalized]
        raise ValueError(
            f"Invalid IBKR market_data_type: {value!r}. "
            "Use one of: live, frozen, delayed, delayed_frozen, 1,2,3,4."
        )
    raise ValueError(f"Invalid IBKR market_data_type type: {type(value)!r}")


def _to_float_or_none(value: object) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _value_by_tag_currency(
    value_map: dict[tuple[str, str], float],
    *,
    tag: str,
    preferred_currency: str = "USD",
) -> float:
    direct = value_map.get((tag, preferred_currency))
    if direct is not None:
        return float(direct)
    for (k_tag, _k_ccy), val in value_map.items():
        if k_tag == tag:
            return float(val)
    return 0.0


class IBKRBroker(BrokerBase):
    def __init__(
        self,
        host: str,
        port: int,
        client_id: int,
        account: str | None = None,
        readonly: bool | str | int | float | None = True,
        market_data_type: int | str | None = 3,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.client_id = int(client_id)
        self.account = account
        self.readonly = _parse_bool(readonly, default=True)
        self.market_data_type = _parse_market_data_type(market_data_type, default=3)
        self._ib = None
        self._Stock = None
        self._MarketOrder = None
        self._LimitOrder = None

    def connect(self) -> None:
        try:
            from ib_insync import IB, LimitOrder, MarketOrder, Stock
        except Exception as exc:
            raise RuntimeError("ib_insync is required for IBKR execution.") from exc

        self._Stock = Stock
        self._MarketOrder = MarketOrder
        self._LimitOrder = LimitOrder
        self._ib = IB()
        self._ib.connect(host=self.host, port=self.port, clientId=self.client_id, timeout=10)

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    def _ensure_connected(self) -> None:
        if self._ib is None or not self._ib.isConnected():
            raise RuntimeError("IBKR broker is not connected.")

    def get_account_snapshot(self) -> AccountSnapshot:
        self._ensure_connected()
        ib = self._ib

        account_values = ib.accountSummary(account=self.account or "")
        value_map: dict[tuple[str, str], float] = {}
        for v in account_values:
            parsed = _to_float_or_none(getattr(v, "value", None))
            if parsed is None:
                continue
            tag = str(getattr(v, "tag", ""))
            currency = str(getattr(v, "currency", ""))
            if not tag:
                continue
            value_map[(tag, currency)] = float(parsed)

        equity = _value_by_tag_currency(value_map, tag="NetLiquidation", preferred_currency="USD")
        cash = _value_by_tag_currency(value_map, tag="TotalCashValue", preferred_currency="USD")

        positions: dict[str, Position] = {}
        for pos in ib.positions(account=self.account or ""):
            symbol = pos.contract.symbol
            qty = float(pos.position)
            if abs(qty) > 0:
                positions[symbol] = Position(symbol=symbol, quantity=qty, avg_cost=float(pos.avgCost))

        return AccountSnapshot(equity=float(equity), cash=float(cash), positions=positions)

    def get_last_prices(self, symbols: list[str]) -> dict[str, float]:
        self._ensure_connected()
        ib = self._ib
        Stock = self._Stock
        contracts = [Stock(symbol, "SMART", "USD") for symbol in sorted(set(symbols))]
        if not contracts:
            return {}

        ib.qualifyContracts(*contracts)
        requested_types: list[int] = [self.market_data_type]
        if self.market_data_type in {1, 2}:
            requested_types.extend([3, 4])
        elif self.market_data_type == 3:
            requested_types.append(4)

        seen_types: set[int] = set()
        for mkt_type in requested_types:
            if mkt_type in seen_types:
                continue
            seen_types.add(mkt_type)
            ib.reqMarketDataType(int(mkt_type))
            tickers = ib.reqTickers(*contracts)

            prices: dict[str, float] = {}
            for ticker in tickers:
                symbol = ticker.contract.symbol
                price = ticker.marketPrice()
                if price is None or price <= 0:
                    price = ticker.last
                if price is None or price <= 0:
                    price = ticker.close
                if price is not None and price > 0:
                    prices[symbol] = float(price)

            if prices:
                return prices
        return {}

    def place_orders(self, orders: list[OrderRequest]) -> list[str]:
        self._ensure_connected()
        if self.readonly:
            raise RuntimeError("IBKR broker is configured as readonly; refusing to place orders.")
        if not orders:
            return []

        ib = self._ib
        Stock = self._Stock
        MarketOrder = self._MarketOrder
        LimitOrder = self._LimitOrder

        broker_ids: list[str] = []
        for req in orders:
            qty = int(req.quantity)
            if qty == 0:
                continue

            action = "BUY" if qty > 0 else "SELL"
            contract = Stock(req.symbol, "SMART", "USD")
            ib.qualifyContracts(contract)
            order_type = str(req.order_type or "MKT").strip().upper()
            if order_type == "MKT":
                order = MarketOrder(action, abs(qty), tif=req.tif)
            elif order_type == "LMT":
                if req.limit_price is None or float(req.limit_price) <= 0:
                    raise ValueError(f"LMT order for {req.symbol} requires positive `limit_price`.")
                order = LimitOrder(action, abs(qty), float(req.limit_price), tif=req.tif)
            else:
                raise ValueError(f"Unsupported IBKR order_type: {order_type}")
            trade = ib.placeOrder(contract, order)
            broker_ids.append(str(trade.order.orderId))
        return broker_ids
