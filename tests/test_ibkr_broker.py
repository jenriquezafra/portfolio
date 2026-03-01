from __future__ import annotations

from dataclasses import dataclass

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
