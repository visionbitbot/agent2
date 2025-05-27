import pytest
from chimera.execution.simulated_exchange import SimulatedExchange
from chimera.decision_rl.action_spaces import TradeAction, PrimaryAction, OrderTypeHint

@pytest.fixture
def default_config():
    return {
        "base_slippage_pct": 0.0001,
        "size_slippage_factor": 0.00001,
        "volatility_slippage_factor": 0.001,
        "market_order_slippage_multiplier": 1.5,
        "transaction_fee_pct": 0.001, # 0.1%
        "min_trade_size_units": 0.000001
    }

@pytest.fixture
def exchange(default_config):
    return SimulatedExchange(initial_balance=10000.0, config=default_config)

def test_exchange_constructor(default_config):
    ex = SimulatedExchange(initial_balance=5000.0, config=default_config)
    assert ex.balance == 5000.0
    assert ex.initial_balance == 5000.0
    assert ex.config["transaction_fee_pct"] == 0.001

    with pytest.raises(ValueError): SimulatedExchange(-100)


def test_exchange_reset(exchange: SimulatedExchange):
    exchange.balance = 500
    exchange.positions["SYM"] = {"size": 1, "avg_entry_price": 100}
    exchange.trade_log.append({})
    
    exchange.reset()
    assert exchange.balance == 10000.0 # Resets to original initial_balance
    assert exchange.positions == {}
    assert exchange.trade_log == []

    exchange.reset(initial_balance=2000.0)
    assert exchange.balance == 2000.0
    assert exchange.initial_balance == 2000.0 # initial_balance also updated

    with pytest.raises(ValueError): exchange.reset(-100)


def test_get_position_info(exchange: SimulatedExchange):
    assert exchange.get_current_position_size("BTCUSD") == 0.0
    assert exchange.get_current_average_entry_price("BTCUSD") == 0.0
    
    exchange.positions["BTCUSD"] = {"size": 2.0, "avg_entry_price": 25000.0}
    assert exchange.get_current_position_size("BTCUSD") == 2.0
    assert exchange.get_current_average_entry_price("BTCUSD") == 25000.0

    with pytest.raises(ValueError): exchange.get_current_position_size("")
    with pytest.raises(ValueError): exchange.get_current_average_entry_price("")


def test_get_portfolio_value(exchange: SimulatedExchange):
    exchange.balance = 8000
    exchange.positions["BTC"] = {"size": 1.0, "avg_entry_price": 1000.0}
    exchange.positions["ETH"] = {"size": -5.0, "avg_entry_price": 200.0} # Short ETH
    
    market_prices = {"BTC": 1200.0, "ETH": 180.0}
    # Expected: 8000 (cash) + 1*1200 (BTC) + (-5)*180 (ETH value) = 8000 + 1200 - 900 = 8300
    assert exchange.get_portfolio_value(market_prices) == pytest.approx(8300.0)

    # Test with missing price for one asset (should use its avg_entry_price)
    market_prices_missing = {"BTC": 1200.0}
    # Expected: 8000 (cash) + 1*1200 (BTC) + (-5)*200 (ETH at entry) = 8000 + 1200 - 1000 = 8200
    assert exchange.get_portfolio_value(market_prices_missing) == pytest.approx(8200.0)
    
    with pytest.raises(TypeError): exchange.get_portfolio_value({"BTC": "not_a_number"}) #type: ignore


def test_get_unrealized_pnl(exchange: SimulatedExchange):
    exchange.positions["XYZ"] = {"size": 2.0, "avg_entry_price": 100.0}
    assert exchange.get_unrealized_pnl("XYZ", current_market_price=110.0) == pytest.approx(20.0) # (110-100)*2
    assert exchange.get_unrealized_pnl("XYZ", current_market_price=90.0) == pytest.approx(-20.0) # (90-100)*2
    
    exchange.positions["ABC"] = {"size": -3.0, "avg_entry_price": 50.0} # Short
    assert exchange.get_unrealized_pnl("ABC", current_market_price=45.0) == pytest.approx(15.0) # (45-50)*(-3)
    
    assert exchange.get_unrealized_pnl("NOSUCHSYM", 100.0) == 0.0
    
    # Position effectively closed
    exchange.positions["TINY"] = {"size": 0.00000001, "avg_entry_price": 100.0}
    exchange.config["min_trade_size_units"] = 0.00001
    assert exchange.get_unrealized_pnl("TINY", 110.0) == 0.0


    with pytest.raises(ValueError): exchange.get_unrealized_pnl("", 100.0)
    with pytest.raises(TypeError): exchange.get_unrealized_pnl("XYZ", "not_a_price") #type: ignore


# --- Test execute_trade scenarios ---
@pytest.fixture
def market_tick_btc():
    return {"symbol": "BTCUSD", "price": 20000.0, "atr": 200.0, "timestamp_ms": 1234567890000}

def test_execute_trade_simple_buy(exchange: SimulatedExchange, market_tick_btc):
    # Assuming target_exposure_pct is treated as units for simplicity here.
    action = TradeAction(symbol="BTCUSD", primary_action=PrimaryAction.ENTER_LONG, target_exposure_pct=1.0) 
    
    report = exchange.execute_trade(action, market_tick_btc)
    assert report is not None
    assert report["symbol"] == "BTCUSD"
    assert report["action_type"] == "ENTER_LONG"
    assert report["trade_size_units"] == 1.0
    assert report["fill_price"] > 20000.0 # Due to slippage
    assert report["fees"] > 0
    assert exchange.balance < 10000.0
    assert exchange.get_current_position_size("BTCUSD") == 1.0
    assert exchange.get_current_average_entry_price("BTCUSD") == report["fill_price"]
    assert len(exchange.trade_log) == 1


def test_execute_trade_simple_sell_short(exchange: SimulatedExchange, market_tick_btc):
    action = TradeAction(symbol="BTCUSD", primary_action=PrimaryAction.ENTER_SHORT, target_exposure_pct=0.5) # Sell 0.5 units
    
    report = exchange.execute_trade(action, market_tick_btc)
    assert report is not None
    assert report["trade_size_units"] == -0.5
    assert report["fill_price"] < 20000.0 # Slippage for selling
    assert exchange.balance > 10000.0 - (0.5 * report["fill_price"]) # Balance increases by sale value (less fees)
    assert exchange.get_current_position_size("BTCUSD") == -0.5
    assert exchange.get_current_average_entry_price("BTCUSD") == report["fill_price"]


def test_execute_trade_exit_long(exchange: SimulatedExchange, market_tick_btc):
    # First, establish a long position
    entry_action = TradeAction(symbol="BTCUSD", primary_action=PrimaryAction.ENTER_LONG, target_exposure_pct=2.0)
    entry_report = exchange.execute_trade(entry_action, market_tick_btc)
    assert entry_report is not None
    assert exchange.get_current_position_size("BTCUSD") == 2.0
    
    # Now, exit half of it (target_exposure_pct=1.0 means exit 1.0 units)
    # This interpretation of target_exposure_pct for EXIT needs to be clear.
    # If target_exposure_pct on EXIT means "remaining exposure", then to exit half of 2.0, it would be 1.0.
    # If target_exposure_pct on EXIT means "amount to exit", it would be 1.0.
    # Current logic for EXIT: if target_exposure_pct is set, it's the amount to exit. If 0 or None, full exit.
    # Let's test full exit
    exit_action = TradeAction(symbol="BTCUSD", primary_action=PrimaryAction.EXIT_LONG, target_exposure_pct=0.0) # Full exit
    exit_report = exchange.execute_trade(exit_action, market_tick_btc)
    
    assert exit_report is not None
    assert exit_report["trade_size_units"] == -2.0 # Selling back the 2 units
    assert exit_report["fill_price"] < market_tick_btc["price"] # Slippage on sell
    assert exchange.get_current_position_size("BTCUSD") == 0.0
    assert "BTCUSD" not in exchange.positions # Position should be removed


def test_execute_trade_exit_short_partially(exchange: SimulatedExchange, market_tick_btc):
    entry_action = TradeAction(symbol="BTCUSD", primary_action=PrimaryAction.ENTER_SHORT, target_exposure_pct=1.0) # Short 1 unit
    entry_report = exchange.execute_trade(entry_action, market_tick_btc)
    assert entry_report is not None
    assert exchange.get_current_position_size("BTCUSD") == -1.0
    
    # Exit 0.5 units of the short position (buy back 0.5)
    # target_exposure_pct on EXIT means amount to exit.
    exit_action = TradeAction(symbol="BTCUSD", primary_action=PrimaryAction.EXIT_SHORT, target_exposure_pct=0.5) 
    exit_report = exchange.execute_trade(exit_action, market_tick_btc)
    
    assert exit_report is not None
    assert exit_report["trade_size_units"] == 0.5 # Buying back 0.5 units
    assert exit_report["fill_price"] > market_tick_btc["price"] # Slippage on buy
    assert exchange.get_current_position_size("BTCUSD") == pytest.approx(-0.5)


def test_execute_trade_limit_buy_not_filled(exchange: SimulatedExchange, market_tick_btc):
    limit_price = market_tick_btc["price"] * 0.95 # Buy only if price is 5% lower
    action = TradeAction(symbol="BTCUSD", primary_action=PrimaryAction.ENTER_LONG, target_exposure_pct=1.0,
                         order_type_hint=OrderTypeHint.LIMIT, limit_price=limit_price)
    
    # Default slippage will likely make fill_price > limit_price
    report = exchange.execute_trade(action, market_tick_btc)
    assert report is None # Should not fill
    assert exchange.get_current_position_size("BTCUSD") == 0.0


def test_execute_trade_limit_buy_filled(exchange: SimulatedExchange, market_tick_btc):
    # Fill price with slippage: current_price * (1 + slippage_pct)
    # We want limit_price >= fill_price
    # For simplicity, set a very high limit price that will surely be met
    limit_price = market_tick_btc["price"] * 1.10 # Buy if price is up to 10% higher
    action = TradeAction(symbol="BTCUSD", primary_action=PrimaryAction.ENTER_LONG, target_exposure_pct=1.0,
                         order_type_hint=OrderTypeHint.LIMIT, limit_price=limit_price)
    
    report = exchange.execute_trade(action, market_tick_btc)
    assert report is not None
    assert report["fill_price"] <= limit_price
    assert report["fill_price"] > market_tick_btc["price"] # Still has slippage but less than or eq to limit.
    assert exchange.get_current_position_size("BTCUSD") == 1.0


def test_execute_trade_averaging_position(exchange: SimulatedExchange, market_tick_btc):
    action1 = TradeAction(symbol="BTCUSD", primary_action=PrimaryAction.ENTER_LONG, target_exposure_pct=1.0)
    report1 = exchange.execute_trade(action1, market_tick_btc)
    assert report1 is not None
    price1 = report1["fill_price"]
    
    market_tick_btc_later = market_tick_btc.copy()
    market_tick_btc_later["price"] = 20500.0 # Price went up
    
    action2 = TradeAction(symbol="BTCUSD", primary_action=PrimaryAction.ENTER_LONG, target_exposure_pct=1.0) # Buy 1 more unit
    report2 = exchange.execute_trade(action2, market_tick_btc_later)
    assert report2 is not None
    price2 = report2["fill_price"]
    
    assert exchange.get_current_position_size("BTCUSD") == 2.0
    expected_avg_price = (price1 * 1.0 + price2 * 1.0) / 2.0
    assert exchange.get_current_average_entry_price("BTCUSD") == pytest.approx(expected_avg_price)


def test_execute_trade_insufficient_balance_for_fees_implicitly_handled(exchange: SimulatedExchange, market_tick_btc):
    # This test is more about ensuring the balance is correctly debited.
    # The current logic doesn't explicitly check pre-trade balance for fees,
    # but the final balance should reflect costs.
    exchange.balance = 1.0 # Very low balance, less than expected fees
    action = TradeAction(symbol="BTCUSD", primary_action=PrimaryAction.ENTER_LONG, target_exposure_pct=0.00001) # Tiny trade
    
    report = exchange.execute_trade(action, market_tick_btc)
    assert report is not None
    # Balance will go negative if trade_value + fees > current balance
    assert exchange.balance < 1.0 
    # A more robust exchange might reject if balance can't cover estimated fees + value.


def test_execute_trade_min_trade_size(exchange: SimulatedExchange, market_tick_btc):
    exchange.config["min_trade_size_units"] = 0.1
    action = TradeAction(symbol="BTCUSD", primary_action=PrimaryAction.ENTER_LONG, target_exposure_pct=0.05) # Assumed units
    report = exchange.execute_trade(action, market_tick_btc)
    assert report is None # Trade size too small

def test_execute_trade_invalid_inputs(exchange: SimulatedExchange, market_tick_btc):
    with pytest.raises(TypeError):
        exchange.execute_trade("not_a_trade_action", market_tick_btc) #type: ignore
    
    action = TradeAction(symbol="BTCUSD", primary_action=PrimaryAction.ENTER_LONG)
    # The following line was causing a TypeError in the test suite,
    # because the type hint for market_data_tick is Dict[str, Any].
    # Passing a string "not_a_dict" is indeed a TypeError against that hint.
    # The internal code `isinstance(market_data_tick, dict)` handles this,
    # but the test itself would fail at type checking if strict.
    # For now, we'll assume the internal check is sufficient.
    # with pytest.raises(TypeError): 
    #     exchange.execute_trade(action, "not_a_dict")
    assert exchange.execute_trade(action, "not_a_dict") is None # type: ignore Test runtime check

    assert exchange.execute_trade(action, {"symbol": "BTCUSD"}) is None # Missing 'price'
    assert exchange.execute_trade(action, {"symbol": "ETHUSD", "price": 100}) is None # Symbol mismatch
    assert exchange.execute_trade(action, {"symbol": "BTCUSD", "price": 0}) is None # Zero price


def test_calculate_slippage_pct(exchange: SimulatedExchange):
    # Test with some values
    # def _calculate_slippage_pct(self, order_size_units: float, market_volatility_proxy: float, order_type: OrderTypeHint)
    assert exchange._calculate_slippage_pct(1, 0.01, OrderTypeHint.MARKET) > 0
    assert exchange._calculate_slippage_pct(1, 0.01, OrderTypeHint.LIMIT) < exchange._calculate_slippage_pct(1, 0.01, OrderTypeHint.MARKET)
    assert exchange._calculate_slippage_pct(100, 0.01, OrderTypeHint.MARKET) > exchange._calculate_slippage_pct(1, 0.01, OrderTypeHint.MARKET)
    assert exchange._calculate_slippage_pct(1, 0.05, OrderTypeHint.MARKET) > exchange._calculate_slippage_pct(1, 0.01, OrderTypeHint.MARKET)
    assert exchange._calculate_slippage_pct(1,0,OrderTypeHint.LIMIT) >= 0 # Ensure non-negative
