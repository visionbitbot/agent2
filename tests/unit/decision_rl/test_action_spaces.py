import pytest
from chimera.decision_rl.action_spaces import PrimaryAction, OrderTypeHint, TradeAction

def test_primary_action_enum_str():
    assert str(PrimaryAction.ENTER_LONG) == "ENTER_LONG"

def test_order_type_hint_enum_str():
    assert str(OrderTypeHint.MARKET) == "MARKET"

def test_trade_action_dataclass_creation():
    action = TradeAction(
        symbol="BTCUSD",
        primary_action=PrimaryAction.ENTER_LONG,
        target_exposure_pct=0.1,
        stop_loss_atr_multiplier=1.5,
        confidence_score=0.8
    )
    assert action.symbol == "BTCUSD"
    assert action.primary_action == PrimaryAction.ENTER_LONG
    assert action.target_exposure_pct == 0.1
    assert action.order_type_hint == OrderTypeHint.MARKET 

def test_trade_action_to_dict():
    action = TradeAction("ETHUSD", PrimaryAction.EXIT_SHORT, order_type_hint=OrderTypeHint.LIMIT, limit_price=1500.0)
    action_dict = action.to_dict()
    assert action_dict["symbol"] == "ETHUSD"
    assert action_dict["primary_action"] == "EXIT_SHORT" 
    assert action_dict["order_type_hint"] == "LIMIT"    
    assert action_dict["limit_price"] == 1500.0

def test_trade_action_from_dict():
    action_dict = {
        "symbol": "SOLUSD",
        "primary_action": "ENTER_SHORT", 
        "target_exposure_pct": 0.05,
        "order_type_hint": "MARKET",    
        "confidence_score": 0.77,
        "stop_loss_price": None, # Ensure all fields are present for robust from_dict
        "stop_loss_atr_multiplier": None,
        "take_profit_price": None,
        "take_profit_atr_multiplier": None,
        "limit_price": None,
        "source_signal_id": None,
        "action_metadata": {}
    }
    action = TradeAction.from_dict(action_dict)
    assert action.symbol == "SOLUSD"
    assert action.primary_action == PrimaryAction.ENTER_SHORT 
    assert action.target_exposure_pct == 0.05
    assert action.order_type_hint == OrderTypeHint.MARKET    
    assert action.confidence_score == 0.77
