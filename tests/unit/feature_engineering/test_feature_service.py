import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from chimera.core.message_queue.in_memory_mq import InMemoryMQClient
from chimera.feature_engineering.feature_service import FeatureEngineeringService
# Mocking the technical_indicators module directly if its functions are complex
# or to ensure isolation for the service logic test.
# For this test, we'll allow it to call the actual functions as they are simple.

@pytest.fixture
def mock_mq_client_fes(): # Renamed to avoid conflict
    client = InMemoryMQClient()
    client.publish = AsyncMock()
    client.subscribe = AsyncMock() # Will be called by service.start()
    client.unsubscribe = AsyncMock() # Will be called by service.stop()
    return client

@pytest.fixture
def feature_service(mock_mq_client_fes):
    service = FeatureEngineeringService(
        mq_client=mock_mq_client_fes,
        input_topics={"market_data": "raw_market_data"},
        output_topic="engineered_features",
        sma_periods=[3], # Use a small period for easier testing
        rsi_periods=[3]  # Use a small period for easier testing
    )
    # Override max_buffer_size for predictability in tests
    # Max of (sma_period=3, rsi_period=3+1=4) is 4. Default is 50.
    # For this test, setting it explicitly based on test needs.
    service.max_buffer_size = 5 
    return service

@pytest.mark.asyncio
async def test_fes_constructor_valid(mock_mq_client_fes):
    service = FeatureEngineeringService(
        mq_client=mock_mq_client_fes,
        input_topics={"market_data": "test_in"},
        output_topic="test_out",
        sma_periods=[5, 10],
        rsi_periods=[7, 14]
    )
    assert service.mq_client == mock_mq_client_fes
    assert service.input_topics == {"market_data": "test_in"}
    assert service.output_topic == "test_out"
    assert service.sma_periods == [5, 10]
    assert service.rsi_periods == [7, 14]
    # Check max_buffer_size logic: max(sma_10, rsi_14+1, default_50) = max(10, 15, 50) = 50
    assert service.max_buffer_size == 50

    service_no_periods = FeatureEngineeringService(
        mq_client=mock_mq_client_fes,
        input_topics={"market_data": "test_in"},
        output_topic="test_out"
    )
    assert service_no_periods.sma_periods == [10, 20] # Defaults
    assert service_no_periods.rsi_periods == [14]    # Defaults
    # Check max_buffer_size logic: max(sma_20, rsi_14+1, default_50) = max(20, 15, 50) = 50
    assert service_no_periods.max_buffer_size == 50

    service_small_periods = FeatureEngineeringService(
        mq_client=mock_mq_client_fes,
        input_topics={"market_data": "test_in"},
        output_topic="test_out",
        sma_periods=[2], rsi_periods=[2]
    )
    # Check max_buffer_size logic: max(sma_2, rsi_2+1, default_50) = max(2, 3, 50) = 50
    assert service_small_periods.max_buffer_size == 50


@pytest.mark.asyncio
async def test_fes_constructor_invalid_args(mock_mq_client_fes):
    with pytest.raises(TypeError): FeatureEngineeringService(None, {"mkd": "t"}, "out") #type: ignore
    with pytest.raises(ValueError): FeatureEngineeringService(mock_mq_client_fes, {}, "out")
    with pytest.raises(ValueError): FeatureEngineeringService(mock_mq_client_fes, {"mkd": "t"}, "")


@pytest.mark.asyncio
async def test_fes_start_subscribes_to_topics(feature_service: FeatureEngineeringService, mock_mq_client_fes: InMemoryMQClient):
    await feature_service.start()
    # Check that subscribe was called on the mq_client mock
    mock_mq_client_fes.subscribe.assert_called_once_with(
        "raw_market_data", feature_service._process_market_data
    )

@pytest.mark.asyncio
async def test_fes_stop_unsubscribes_from_topics(feature_service: FeatureEngineeringService, mock_mq_client_fes: InMemoryMQClient):
    # Start first to allow stop to actually run unsubscribe
    await feature_service.start() 
    await feature_service.stop()
    mock_mq_client_fes.unsubscribe.assert_called_once_with(
        "raw_market_data", feature_service._process_market_data
    )


@pytest.mark.asyncio
async def test_fes_process_market_data_calculates_and_publishes_features(feature_service: FeatureEngineeringService, mock_mq_client_fes: InMemoryMQClient):
    # feature_service has sma_periods=[3], rsi_periods=[3], max_buffer_size = 5
    market_data_sequence = [
        {"timestamp_ms": 1000, "symbol": "BTCUSD", "price": 10.0, "id": "md1"},
        {"timestamp_ms": 2000, "symbol": "BTCUSD", "price": 11.0, "id": "md2"},
        {"timestamp_ms": 3000, "symbol": "BTCUSD", "price": 12.0, "id": "md3"}, # SMA(3) on [10,11,12] = 11.0. RSI(3) needs 4 points.
        {"timestamp_ms": 4000, "symbol": "BTCUSD", "price": 13.0, "id": "md4"}, # SMA(3) on [11,12,13] = 12.0. RSI(3) on [10,11,12,13] = 100.0
        {"timestamp_ms": 5000, "symbol": "BTCUSD", "price": 12.0, "id": "md5"}, # Buffer: [10,11,12,13,12]. SMA(3) on [12,13,12]=12.33. RSI(3) on [11,12,13,12]
    ]

    for msg in market_data_sequence:
        await feature_service._process_market_data(msg)
    
    # Expected publish calls:
    # 1. After md3: SMA_3 available.
    # 2. After md4: SMA_3 and RSI_3 available.
    # 3. After md5: SMA_3 and RSI_3 available.
    assert mock_mq_client_fes.publish.call_count == 3

    # First call (after md3)
    args1, _ = mock_mq_client_fes.publish.call_args_list[0]
    topic1, features1 = args1
    assert topic1 == "engineered_features"
    assert features1["symbol"] == "BTCUSD"
    assert features1["timestamp_ms"] == 3000
    assert features1["price_at_feature_generation"] == 12.0
    assert features1["sma_3"] == pytest.approx(11.0)
    assert "rsi_3" not in features1 

    # Second call (after md4)
    args2, _ = mock_mq_client_fes.publish.call_args_list[1]
    topic2, features2 = args2
    assert topic2 == "engineered_features"
    assert features2["symbol"] == "BTCUSD"
    assert features2["timestamp_ms"] == 4000
    assert features2["price_at_feature_generation"] == 13.0
    assert features2["sma_3"] == pytest.approx(12.0) # SMA of [11,12,13]
    assert features2["rsi_3"] == pytest.approx(100.0) # RSI of [10,11,12,13] is 100

    # Third call (after md5)
    # Buffer: [10,11,12,13,12] (max_buffer_size is 5, so all are kept)
    # SMA_3 on [12,13,12] = 12.333...
    # RSI_3 on prices [11,12,13,12] (last 4 from buffer of 5 for RSI period 3)
    # Delta for RSI: [1, 1, -1] (from [11,12,13,12])
    # Gains: 1, 1, 0. Losses: 0, 0, 1. Window 3.
    # Avg Gain over last 3 deltas: (1+1+0)/3 = 2/3
    # Avg Loss over last 3 deltas: (0+0+1)/3 = 1/3
    # RS = (2/3) / (1/3) = 2
    # RSI = 100 - (100 / (1+2)) = 100 - 100/3 = 66.666...
    args3, _ = mock_mq_client_fes.publish.call_args_list[2]
    topic3, features3 = args3
    assert topic3 == "engineered_features"
    assert features3["symbol"] == "BTCUSD"
    assert features3["timestamp_ms"] == 5000
    assert features3["price_at_feature_generation"] == 12.0
    assert features3["sma_3"] == pytest.approx(12.333333, abs=1e-5)
    assert features3["rsi_3"] == pytest.approx(66.666666, abs=1e-5)


@pytest.mark.asyncio
async def test_fes_process_market_data_buffer_management(feature_service: FeatureEngineeringService):
    # feature_service.max_buffer_size is 5
    for i in range(10):
        await feature_service._process_market_data(
            {"timestamp_ms": 1000 * (i+1), "symbol": "XYZ", "price": float(100 + i), "id": f"m{i}"}
        )
    
    assert len(feature_service.market_data_buffer["XYZ"]["price"]) == feature_service.max_buffer_size
    # Check that the latest prices are in the buffer
    assert feature_service.market_data_buffer["XYZ"]["price"] == [105.0, 106.0, 107.0, 108.0, 109.0]

@pytest.mark.asyncio
async def test_fes_process_market_data_invalid_message(feature_service: FeatureEngineeringService, mock_mq_client_fes: InMemoryMQClient):
    # Message missing 'price'
    await feature_service._process_market_data({"timestamp_ms": 1000, "symbol": "ABC", "id":"inv1"}) 
    mock_mq_client_fes.publish.assert_not_called() 

    # Message missing 'symbol'
    await feature_service._process_market_data({"timestamp_ms": 1000, "price": 1.0, "id":"inv2"}) 
    mock_mq_client_fes.publish.assert_not_called() 

    # Message missing 'timestamp_ms' 
    await feature_service._process_market_data({"symbol": "ABC", "price": 1.0, "id":"inv3"}) 
    mock_mq_client_fes.publish.assert_not_called()

    # Empty message
    await feature_service._process_market_data({})
    mock_mq_client_fes.publish.assert_not_called()
    
    # Ensure no calls were made in any of these invalid cases
    assert mock_mq_client_fes.publish.call_count == 0
