import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd # For Timestamp, ensure it's available or mock it

from chimera.core.message_queue.in_memory_mq import InMemoryMQClient
from chimera.execution.execution_service import ExecutionService
from chimera.execution.simulated_exchange import SimulatedExchange
from chimera.decision_rl.action_spaces import TradeAction, PrimaryAction, OrderTypeHint # Ensure this path is correct

@pytest.fixture
def mock_mq_client_es(): # Renamed
    client = InMemoryMQClient()
    client.publish = AsyncMock()
    # We will manually call the service's handlers, so direct subscribe mock isn't strictly needed for all tests
    # but it's good for start/stop tests.
    client.subscribe = AsyncMock()
    client.unsubscribe = AsyncMock()
    return client

@pytest.fixture
def mock_simulated_exchange():
    exchange = MagicMock(spec=SimulatedExchange)
    exchange.execute_trade = MagicMock() # This will be called by the service
    exchange.config = {"simulated_latency_sec": 0.0} # Default for tests
    return exchange

@pytest.fixture
def execution_service(mock_mq_client_es, mock_simulated_exchange):
    return ExecutionService(
        mq_client=mock_mq_client_es,
        simulated_exchange=mock_simulated_exchange
    )

def test_es_constructor_valid(mock_mq_client_es, mock_simulated_exchange):
    service = ExecutionService(
        mq_client=mock_mq_client_es,
        simulated_exchange=mock_simulated_exchange,
        input_topic_approved_actions="custom_approved",
        output_topic_confirmations="custom_confirm",
        market_data_topic="custom_market"
    )
    assert service.mq_client == mock_mq_client_es
    assert service.simulated_exchange == mock_simulated_exchange
    assert service.input_topic_approved_actions == "custom_approved"

def test_es_constructor_invalid_args(mock_mq_client_es, mock_simulated_exchange):
    with pytest.raises(TypeError): ExecutionService(None, mock_simulated_exchange) # type: ignore
    with pytest.raises(TypeError): ExecutionService(mock_mq_client_es, None) # type: ignore
    with pytest.raises(ValueError): ExecutionService(mock_mq_client_es, mock_simulated_exchange, input_topic_approved_actions="")
    with pytest.raises(ValueError): ExecutionService(mock_mq_client_es, mock_simulated_exchange, output_topic_confirmations="")
    with pytest.raises(ValueError): ExecutionService(mock_mq_client_es, mock_simulated_exchange, market_data_topic="")


@pytest.mark.asyncio
async def test_es_start_subscribes_to_topics(execution_service: ExecutionService, mock_mq_client_es: InMemoryMQClient):
    await execution_service.start()
    assert mock_mq_client_es.subscribe.call_count == 2
    mock_mq_client_es.subscribe.assert_any_call(execution_service.input_topic_approved_actions, execution_service._on_approved_action)
    mock_mq_client_es.subscribe.assert_any_call(execution_service.market_data_topic, execution_service._on_market_data)

@pytest.mark.asyncio
async def test_es_stop_unsubscribes_from_topics(execution_service: ExecutionService, mock_mq_client_es: InMemoryMQClient):
    await execution_service.start() # Call start to enable subscriptions
    await execution_service.stop()
    assert mock_mq_client_es.unsubscribe.call_count == 2
    mock_mq_client_es.unsubscribe.assert_any_call(execution_service.input_topic_approved_actions, execution_service._on_approved_action)
    mock_mq_client_es.unsubscribe.assert_any_call(execution_service.market_data_topic, execution_service._on_market_data)


@pytest.mark.asyncio
async def test_es_on_market_data_updates_latest_market_data(execution_service: ExecutionService):
    tick1 = {"symbol": "BTCUSD", "price": 20000}
    tick2 = {"symbol": "ETHUSD", "price": 1500}
    tick_btc_update = {"symbol": "BTCUSD", "price": 20050}

    await execution_service._on_market_data(tick1)
    assert execution_service.latest_market_data["BTCUSD"] == tick1
    
    await execution_service._on_market_data(tick2)
    assert execution_service.latest_market_data["ETHUSD"] == tick2
    
    await execution_service._on_market_data(tick_btc_update)
    assert execution_service.latest_market_data["BTCUSD"] == tick_btc_update

@pytest.mark.asyncio
async def test_es_on_market_data_invalid_tick(execution_service: ExecutionService):
    initial_state = execution_service.latest_market_data.copy()
    await execution_service._on_market_data({"price": 123}) # Missing symbol
    assert execution_service.latest_market_data == initial_state


@pytest.mark.asyncio
async def test_es_on_approved_action_successful_execution(execution_service: ExecutionService, mock_simulated_exchange: MagicMock, mock_mq_client_es: AsyncMock):
    # Setup market data context
    market_tick = {"symbol": "BTCUSD", "price": 20000, "atr": 200, "timestamp_ms": 12345}
    await execution_service._on_market_data(market_tick)

    action = TradeAction(symbol="BTCUSD", primary_action=PrimaryAction.ENTER_LONG, target_exposure_pct=1.0, source_signal_id="sig1")
    action_dict = action.to_dict() # This is what the service receives

    # Mock SimulatedExchange response
    mock_exec_report = {
        "timestamp_ms": 12345, "symbol": "BTCUSD", "action_type": "ENTER_LONG", 
        "trade_size_units": 1.0, "fill_price": 20010.0, "fees": 20.01, 
        "source_signal_id": "sig1"
    }
    mock_simulated_exchange.execute_trade.return_value = mock_exec_report
    
    await execution_service._on_approved_action(action_dict)

    mock_simulated_exchange.execute_trade.assert_called_once()
    # The first argument to execute_trade should be a TradeAction object
    called_action_arg = mock_simulated_exchange.execute_trade.call_args[0][0]
    assert isinstance(called_action_arg, TradeAction)
    assert called_action_arg.symbol == "BTCUSD"
    assert called_action_arg.primary_action == PrimaryAction.ENTER_LONG
    
    mock_mq_client_es.publish.assert_called_once()
    args, _ = mock_mq_client_es.publish.call_args
    topic, published_msg = args
    
    assert topic == execution_service.output_topic_confirmations
    assert published_msg["status"] == "FILLED"
    assert published_msg["symbol"] == "BTCUSD"
    assert published_msg["fill_price"] == 20010.0
    assert published_msg["source_signal_id"] == "sig1"


@pytest.mark.asyncio
async def test_es_on_approved_action_rejected_by_exchange(execution_service: ExecutionService, mock_simulated_exchange: MagicMock, mock_mq_client_es: AsyncMock):
    market_tick = {"symbol": "ETHUSD", "price": 1500, "timestamp_ms": 67890}
    await execution_service._on_market_data(market_tick)

    action = TradeAction(symbol="ETHUSD", primary_action=PrimaryAction.ENTER_SHORT, target_exposure_pct=0.5, source_signal_id="sig2")
    action_dict = action.to_dict()

    mock_simulated_exchange.execute_trade.return_value = None # Simulate rejection
    
    await execution_service._on_approved_action(action_dict)

    mock_simulated_exchange.execute_trade.assert_called_once()
    mock_mq_client_es.publish.assert_called_once()
    args, _ = mock_mq_client_es.publish.call_args
    _, published_msg = args
    
    assert published_msg["status"] == "REJECTED_BY_EXCHANGE"
    assert published_msg["symbol"] == "ETHUSD"
    assert published_msg["source_signal_id"] == "sig2"


@pytest.mark.asyncio
async def test_es_on_approved_action_no_market_data(execution_service: ExecutionService, mock_simulated_exchange: MagicMock, mock_mq_client_es: AsyncMock):
    # No market data for "XYZ"
    execution_service.latest_market_data.clear()

    action = TradeAction(symbol="XYZ", primary_action=PrimaryAction.ENTER_LONG, target_exposure_pct=1.0, source_signal_id="sig3")
    action_dict = action.to_dict()
    
    await execution_service._on_approved_action(action_dict)

    mock_simulated_exchange.execute_trade.assert_not_called() # Should not attempt to execute
    mock_mq_client_es.publish.assert_called_once()
    args, _ = mock_mq_client_es.publish.call_args
    _, published_msg = args
    
    assert published_msg["status"] == "REJECTED_NO_MARKET_DATA"
    assert published_msg["symbol"] == "XYZ"
    assert published_msg["source_signal_id"] == "sig3"

@pytest.mark.asyncio
async def test_es_on_approved_action_invalid_action_dict(execution_service: ExecutionService, mock_simulated_exchange: MagicMock, mock_mq_client_es: AsyncMock):
    await execution_service._on_approved_action({"invalid": "dict"}) # Does not conform to TradeAction
    
    mock_simulated_exchange.execute_trade.assert_not_called()
    # Depending on how strict the error handling for malformed TradeAction is,
    # it might publish a different kind of error or nothing.
    # Current code for _on_approved_action has a try-except for TradeAction.from_dict that would print and return.
    # So, no publish call is expected if reconstruction fails.
    mock_mq_client_es.publish.assert_not_called()


@pytest.mark.asyncio
@patch('asyncio.sleep', new_callable=AsyncMock) # Patch asyncio.sleep
async def test_es_on_approved_action_with_simulated_latency(mock_sleep: AsyncMock, execution_service: ExecutionService, mock_simulated_exchange: MagicMock, mock_mq_client_es: AsyncMock):
    mock_simulated_exchange.config["simulated_latency_sec"] = 0.1 # Set latency
    
    market_tick = {"symbol": "BTCUSD", "price": 20000, "timestamp_ms": 12345}
    await execution_service._on_market_data(market_tick)

    action = TradeAction(symbol="BTCUSD", primary_action=PrimaryAction.ENTER_LONG)
    action_dict = action.to_dict()
    
    mock_simulated_exchange.execute_trade.return_value = {"status": "FILLED_placeholder"} # Dummy report

    await execution_service._on_approved_action(action_dict)
    
    mock_sleep.assert_called_once_with(0.1) # Check if sleep was called with the configured latency
    mock_simulated_exchange.execute_trade.assert_called_once()
    mock_mq_client_es.publish.assert_called_once()
