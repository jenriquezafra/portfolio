from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float
    avg_cost: float | None = None


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    cash: float
    positions: dict[str, Position]


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    quantity: int
    order_type: str = "MKT"
    tif: str = "DAY"


class BrokerBase(ABC):
    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_account_snapshot(self) -> AccountSnapshot:
        raise NotImplementedError

    @abstractmethod
    def get_last_prices(self, symbols: list[str]) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def place_orders(self, orders: list[OrderRequest]) -> list[str]:
        raise NotImplementedError
