from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_rebalance_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "04_rebalance.py"
    spec = importlib.util.spec_from_file_location("rebalance_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_bool_env_like_values() -> None:
    module = _load_rebalance_module()
    assert module._parse_bool("true", default=False) is True
    assert module._parse_bool("TRUE", default=False) is True
    assert module._parse_bool("1", default=False) is True
    assert module._parse_bool("false", default=True) is False
    assert module._parse_bool("0", default=True) is False
    assert module._parse_bool(None, default=True) is True
    assert module._parse_bool(None, default=False) is False


def test_parse_bool_rejects_invalid_text() -> None:
    module = _load_rebalance_module()
    with pytest.raises(ValueError):
        module._parse_bool("not-a-bool", default=True)


def test_build_broker_parses_ibkr_readonly_from_env_string(tmp_path: Path) -> None:
    module = _load_rebalance_module()

    execution_cfg = {
        "execution": {"broker": "ibkr"},
        "ibkr": {
            "host": "127.0.0.1",
            "port": "7497",
            "client_id": "101",
            "account": "DU1234567",
            "readonly": "false",
        },
    }
    broker = module._build_broker(execution_cfg=execution_cfg, project_root=tmp_path, as_of_date=None)

    assert broker.readonly is False
    assert broker.market_data_type == 3


def test_build_broker_parses_ibkr_market_data_type_label(tmp_path: Path) -> None:
    module = _load_rebalance_module()

    execution_cfg = {
        "execution": {"broker": "ibkr"},
        "ibkr": {
            "host": "127.0.0.1",
            "port": "4002",
            "client_id": "101",
            "account": "DU1234567",
            "readonly": "true",
            "market_data_type": "delayed_frozen",
        },
    }
    broker = module._build_broker(execution_cfg=execution_cfg, project_root=tmp_path, as_of_date=None)

    assert broker.readonly is True
    assert broker.market_data_type == 4
