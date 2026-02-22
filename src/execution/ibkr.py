from __future__ import annotations

from src.execution.broker_base import AccountSnapshot, BrokerBase, OrderRequest, Position


class IBKRBroker(BrokerBase):
    def __init__(
        self,
        host: str,
        port: int,
        client_id: int,
        account: str | None = None,
        readonly: bool = True,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.client_id = int(client_id)
        self.account = account
        self.readonly = bool(readonly)
        self._ib = None
        self._Stock = None
        self._MarketOrder = None

    def connect(self) -> None:
        try:
            from ib_insync import IB, MarketOrder, Stock
        except Exception as exc:
            raise RuntimeError("ib_insync is required for IBKR execution.") from exc

        self._Stock = Stock
        self._MarketOrder = MarketOrder
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
        value_map = {(v.tag, v.currency): float(v.value) for v in account_values if v.value not in {"", None}}

        equity = value_map.get(("NetLiquidation", "USD"), 0.0)
        cash = value_map.get(("TotalCashValue", "USD"), 0.0)

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
        return prices

    def place_orders(self, orders: list[OrderRequest]) -> list[str]:
        self._ensure_connected()
        if self.readonly:
            raise RuntimeError("IBKR broker is configured as readonly; refusing to place orders.")
        if not orders:
            return []

        ib = self._ib
        Stock = self._Stock
        MarketOrder = self._MarketOrder

        broker_ids: list[str] = []
        for req in orders:
            qty = int(req.quantity)
            if qty == 0:
                continue

            action = "BUY" if qty > 0 else "SELL"
            contract = Stock(req.symbol, "SMART", "USD")
            ib.qualifyContracts(contract)
            order = MarketOrder(action, abs(qty), tif=req.tif)
            trade = ib.placeOrder(contract, order)
            broker_ids.append(str(trade.order.orderId))
        return broker_ids
