import random
import pandas as pd # Used for Timestamp if needed, but not strictly required for this class
from typing import Optional, Dict, Any, List # Added List for trade_log type hint
from chimera.decision_rl.action_spaces import TradeAction, OrderTypeHint, PrimaryAction 

class SimulatedExchange:
    def __init__(self, initial_balance: float, config: dict | None = None):
        if not isinstance(initial_balance, (int, float)) or initial_balance < 0:
            raise ValueError("Initial balance must be a non-negative number.")
        
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.positions: Dict[str, Dict[str, float]] = {} 
        self.config = config if config is not None else {}
        self.trade_log: List[Dict[str, Any]] = []

        # Default config values if not provided
        self.config.setdefault("base_slippage_pct", 0.0001) # 0.01%
        self.config.setdefault("size_slippage_factor", 0.00001) # Slippage increases with size
        self.config.setdefault("volatility_slippage_factor", 0.001) # Slippage increases with volatility
        self.config.setdefault("market_order_slippage_multiplier", 1.5)
        self.config.setdefault("transaction_fee_pct", 0.001) # 0.1% fee
        self.config.setdefault("min_trade_size_units", 0.000001) # Minimum units to consider a trade/position
        self.config.setdefault("simulated_latency_sec", 0.0) # Placeholder for future latency modeling

    def reset(self, initial_balance: Optional[float] = None):
        new_initial_balance = initial_balance if initial_balance is not None else self.initial_balance
        if not isinstance(new_initial_balance, (int, float)) or new_initial_balance < 0:
            raise ValueError("Initial balance for reset must be a non-negative number.")

        self.balance = float(new_initial_balance)
        self.initial_balance = float(new_initial_balance) # Also update initial_balance if provided
        self.positions.clear()
        self.trade_log.clear()

    def get_current_position_size(self, symbol: str) -> float:
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("Symbol must be a non-empty string.")
        return self.positions.get(symbol, {}).get("size", 0.0)

    def get_current_average_entry_price(self, symbol: str) -> float:
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("Symbol must be a non-empty string.")
        return self.positions.get(symbol, {}).get("avg_entry_price", 0.0)

    def get_portfolio_value(self, current_market_prices: Dict[str, float]) -> float:
        if not isinstance(current_market_prices, dict) or            not all(isinstance(k, str) and isinstance(v, (int,float)) for k,v in current_market_prices.items()):
            raise TypeError("current_market_prices must be a dictionary of string keys and numeric values.")

        total_value = self.balance
        for symbol, pos_details in self.positions.items():
            market_price = current_market_prices.get(symbol, pos_details["avg_entry_price"])
            total_value += pos_details["size"] * market_price
        return total_value

    def get_unrealized_pnl(self, symbol: str, current_market_price: float) -> float:
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("Symbol must be a non-empty string.")
        if not isinstance(current_market_price, (int, float)):
            raise TypeError("Current market price must be a number.")
            
        pos = self.positions.get(symbol)
        if not pos or abs(pos["size"]) < self.config["min_trade_size_units"]: # Consider position closed if very small
            return 0.0
        return (current_market_price - pos["avg_entry_price"]) * pos["size"]

    def _calculate_slippage_pct(self, order_size_units: float, market_volatility_proxy: float, order_type: OrderTypeHint) -> float:
        slippage = self.config["base_slippage_pct"]
        slippage += abs(order_size_units) * self.config["size_slippage_factor"]
        slippage += market_volatility_proxy * self.config["volatility_slippage_factor"]
        if order_type == OrderTypeHint.MARKET:
            slippage *= self.config["market_order_slippage_multiplier"]
        return max(0, slippage) # Slippage cannot be negative

    def _calculate_fees(self, trade_value: float) -> float:
        return abs(trade_value) * self.config["transaction_fee_pct"]

    def execute_trade(self, action: TradeAction, market_data_tick: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(action, TradeAction):
            raise TypeError("Action must be a TradeAction object.")
        if not isinstance(market_data_tick, dict) or "price" not in market_data_tick or "symbol" not in market_data_tick:
            # print(f"SimEx: Invalid market_data_tick: {market_data_tick}") # Optional
            return None
        if action.symbol != market_data_tick["symbol"]:
            # print(f"SimEx: Action symbol {action.symbol} does not match market data tick symbol {market_data_tick['symbol']}") # Optional
            return None
            
        current_price = float(market_data_tick["price"])
        if current_price <= 0: return None # Cannot trade at zero or negative price

        current_pos_size = self.get_current_position_size(action.symbol)
        
        trade_size_units = 0.0
        
        if action.primary_action == PrimaryAction.ENTER_LONG:
            trade_size_units = action.target_exposure_pct if action.target_exposure_pct is not None else 1.0 
            if current_pos_size < 0: 
                trade_size_units += abs(current_pos_size) 
        elif action.primary_action == PrimaryAction.ENTER_SHORT:
            trade_size_units = -(action.target_exposure_pct if action.target_exposure_pct is not None else 1.0)
            if current_pos_size > 0: 
                trade_size_units -= current_pos_size
        elif action.primary_action == PrimaryAction.EXIT_LONG:
            if current_pos_size > 0:
                exit_amount = current_pos_size # Default full exit
                if action.target_exposure_pct is not None: # If specified, it's amount to exit
                    exit_amount = action.target_exposure_pct
                
                trade_size_units = -min(current_pos_size, abs(exit_amount)) 
                if action.target_exposure_pct == 0.0: trade_size_units = -current_pos_size 
        elif action.primary_action == PrimaryAction.EXIT_SHORT:
            if current_pos_size < 0:
                exit_amount = abs(current_pos_size) # Default full exit
                if action.target_exposure_pct is not None: # If specified, it's amount to exit
                     exit_amount = action.target_exposure_pct

                trade_size_units = min(abs(current_pos_size), abs(exit_amount))
                if action.target_exposure_pct == 0.0: trade_size_units = abs(current_pos_size) 
        elif action.primary_action == PrimaryAction.DO_NOTHING or action.primary_action == PrimaryAction.HOLD_ADJUST_RISK:
            return None 
        else: return None 

        if abs(trade_size_units) < self.config["min_trade_size_units"]:
            return None 

        market_volatility_proxy = float(market_data_tick.get("atr", current_price * 0.01)) / current_price 
        slippage_pct = self._calculate_slippage_pct(trade_size_units, market_volatility_proxy, action.order_type_hint)
        
        fill_price = current_price
        if trade_size_units > 0: 
            fill_price *= (1 + slippage_pct)
        elif trade_size_units < 0: 
            fill_price *= (1 - slippage_pct)
        
        if action.order_type_hint == OrderTypeHint.LIMIT and action.limit_price is not None:
            if trade_size_units > 0: 
                if fill_price > action.limit_price: return None 
                fill_price = min(fill_price, action.limit_price) 
            elif trade_size_units < 0: 
                if fill_price < action.limit_price: return None 
                fill_price = max(fill_price, action.limit_price) 
        
        trade_value_at_fill = trade_size_units * fill_price
        fees = self._calculate_fees(trade_value_at_fill)

        self.balance -= trade_value_at_fill 
        self.balance -= fees
        
        current_pos = self.positions.get(action.symbol, {"size": 0.0, "avg_entry_price": 0.0})
        old_size = current_pos["size"]
        new_total_size = old_size + trade_size_units

        if abs(new_total_size) < self.config["min_trade_size_units"]: 
            current_pos["avg_entry_price"] = 0.0
            current_pos["size"] = 0.0
            if action.symbol in self.positions: del self.positions[action.symbol]
        else:
            if (old_size == 0) or                (old_size > 0 and trade_size_units > 0) or                (old_size < 0 and trade_size_units < 0): 
                new_avg_entry_price = ((current_pos["avg_entry_price"] * old_size) + (fill_price * trade_size_units)) / new_total_size
                current_pos["avg_entry_price"] = new_avg_entry_price
            current_pos["size"] = new_total_size
            self.positions[action.symbol] = current_pos
        
        execution_details = {
            "timestamp_ms": market_data_tick.get("timestamp_ms", int(pd.Timestamp.now().timestamp()*1000)), 
            "symbol": action.symbol,
            "action_type": str(action.primary_action),
            "trade_size_units": trade_size_units,
            "requested_price_estimate": current_price, 
            "fill_price": fill_price,
            "fees": fees,
            "order_type": str(action.order_type_hint),
            "limit_price_requested": action.limit_price,
            "new_position_size": current_pos["size"],
            "new_avg_entry_price": current_pos["avg_entry_price"],
            "balance_after_trade": self.balance,
            "source_signal_id": action.source_signal_id
        }
        self.trade_log.append(execution_details)
        return execution_details

    def close(self): 
        pass
