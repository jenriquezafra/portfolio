from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.execution.broker_base import OrderRequest
from src.execution.ibkr import IBKRBroker


@dataclass
class _FakeAccountValue:
    tag: str
    currency: str
    value: object


class _FakeIB:
    def __init__(self) -> None:
        self._connected = True

    def isConnected(self) -> bool:  # noqa: N802 - mimic ib_insync API
        return self._connected

    def accountSummary(self, account: str = ""):  # noqa: N802 - mimic ib_insync API
        return [
            _FakeAccountValue(tag="AccountType", currency="", value="INDIVIDUAL"),
            _FakeAccountValue(tag="NetLiquidation", currency="USD", value="12345.67"),
            _FakeAccountValue(tag="TotalCashValue", currency="USD", value="2345.67"),
        ]

    def positions(self, account: str = ""):  # noqa: N802 - mimic ib_insync API
        return []


class _FakeIBOrders(_FakeIB):
    def __init__(self) -> None:
        super().__init__()
        self.placed: list[tuple[object, object]] = []
        self._next_id = 1000

    def qualifyContracts(self, *contracts: object) -> None:  # noqa: N802 - mimic ib_insync API
        return None

    def placeOrder(self, contract: object, order: object):  # noqa: N802 - mimic ib_insync API
        self._next_id += 1
        setattr(order, "orderId", self._next_id)
        self.placed.append((contract, order))
        return SimpleNamespace(order=order)


class _FakeStock:
    def __init__(self, symbol: str, exchange: str, currency: str) -> None:
        self.symbol = symbol
        self.exchange = exchange
        self.currency = currency


class _FakeMarketOrder:
    def __init__(self, action: str, total_quantity: int, tif: str = "DAY") -> None:
        self.action = action
        self.totalQuantity = total_quantity
        self.tif = tif
        self.kind = "MKT"


class _FakeLimitOrder:
    def __init__(self, action: str, total_quantity: int, lmt_price: float, tif: str = "DAY") -> None:
        self.action = action
        self.totalQuantity = total_quantity
        self.lmtPrice = lmt_price
        self.tif = tif
        self.kind = "LMT"


def test_get_account_snapshot_ignores_non_numeric_account_summary_values() -> None:
    broker = IBKRBroker(
        host="127.0.0.1",
        port=4002,
        client_id=101,
        account="DU1234567",
        readonly=True,
        market_data_type="delayed",
    )
    broker._ib = _FakeIB()
    snapshot = broker.get_account_snapshot()

    assert snapshot.equity == 12345.67
    assert snapshot.cash == 2345.67
    assert snapshot.positions == {}


def test_place_orders_uses_limit_order_when_requested() -> None:
    broker = IBKRBroker(
        host="127.0.0.1",
        port=4002,
        client_id=101,
        account="DU1234567",
        readonly=False,
        market_data_type="delayed",
    )
    fake_ib = _FakeIBOrders()
    broker._ib = fake_ib
    broker._Stock = _FakeStock
    broker._MarketOrder = _FakeMarketOrder
    broker._LimitOrder = _FakeLimitOrder

    ids = broker.place_orders(
        [
            OrderRequest(
                symbol="AAPL",
                quantity=10,
                order_type="LMT",
                tif="GTC",
                limit_price=123.45,
            )
        ]
    )

    assert ids == ["1001"]
    assert len(fake_ib.placed) == 1
    contract, order = fake_ib.placed[0]
    assert contract.symbol == "AAPL"
    assert order.kind == "LMT"
    assert order.lmtPrice == 123.45
    assert order.tif == "GTC"


def test_place_orders_rejects_lmt_without_limit_price() -> None:
    broker = IBKRBroker(
        host="127.0.0.1",
        port=4002,
        client_id=101,
        account="DU1234567",
        readonly=False,
        market_data_type="delayed",
    )
    broker._ib = _FakeIBOrders()
    broker._Stock = _FakeStock
    broker._MarketOrder = _FakeMarketOrder
    broker._LimitOrder = _FakeLimitOrder

    with pytest.raises(ValueError):
        broker.place_orders([OrderRequest(symbol="AAPL", quantity=10, order_type="LMT", tif="GTC")])
