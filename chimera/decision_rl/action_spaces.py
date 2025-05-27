from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

class PrimaryAction(Enum):
    DO_NOTHING = 0
    ENTER_LONG = 1
    ENTER_SHORT = 2
    EXIT_LONG = 3 
    EXIT_SHORT = 4 
    HOLD_ADJUST_RISK = 5 

    def __str__(self):
        return self.name

class OrderTypeHint(Enum):
    MARKET = 0
    LIMIT = 1
    ADAPTIVE_ALGO = 2 

    def __str__(self):
        return self.name

@dataclass
class TradeAction:
    symbol: str
    primary_action: PrimaryAction
    target_exposure_pct: Optional[float] = None 
    stop_loss_price: Optional[float] = None
    stop_loss_atr_multiplier: Optional[float] = None 
    take_profit_price: Optional[float] = None
    take_profit_atr_multiplier: Optional[float] = None 
    limit_price: Optional[float] = None 
    order_type_hint: OrderTypeHint = OrderTypeHint.MARKET
    source_signal_id: Optional[str] = None 
    confidence_score: Optional[float] = None 
    action_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['primary_action'] = str(self.primary_action)
        data['order_type_hint'] = str(self.order_type_hint)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeAction':
        data['primary_action'] = PrimaryAction[data['primary_action']]
        data['order_type_hint'] = OrderTypeHint[data['order_type_hint']]
        return cls(**data)
