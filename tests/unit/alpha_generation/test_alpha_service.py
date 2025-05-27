import pytest
import asyncio
import pandas as pd
from unittest.mock import AsyncMock, MagicMock

from chimera.core.message_queue.in_memory_mq import InMemoryMQClient
from chimera.alpha_generation.service import AlphaGenerationService
from chimera.alpha_generation.base_alpha_model import BaseAlphaModel
from chimera.alpha_generation.moe_transformer_model import MoETransformerModel # Placeholder

@pytest.fixture
def mock_mq_client_ags(): # Renamed
    client = InMemoryMQClient()
    client.publish = AsyncMock()
    client.subscribe = AsyncMock()
    client.unsubscribe = AsyncMock()
    return client

@pytest.fixture
def placeholder_model_fixture(): # Renamed
    return MoETransformerModel(config={"random_seed": 42}) # Consistent seed for tests

@pytest.fixture
def alpha_service(mock_mq_client_ags, placeholder_model_fixture):
    return AlphaGenerationService(
        mq_client=mock_mq_client_ags,
        model_registry={"default_model": placeholder_model_fixture},
        input_topic="engineered_features",
        output_topic="alpha_signals"
    )

def test_ags_constructor_valid(mock_mq_client_ags, placeholder_model_fixture):
    service = AlphaGenerationService(
        mq_client=mock_mq_client_ags,
        model_registry={"model1": placeholder_model_fixture},
        default_model_key="model1"
    )
    assert service.mq_client == mock_mq_client_ags
    assert "model1" in service.models
    assert service.default_model_key == "model1"

def test_ags_constructor_default_model(mock_mq_client_ags):
    service = AlphaGenerationService(mq_client=mock_mq_client_ags) # No model registry
    assert "default_model" in service.models
    assert isinstance(service.models["default_model"], MoETransformerModel)

def test_ags_constructor_invalid_args(mock_mq_client_ags, placeholder_model_fixture):
    with pytest.raises(TypeError): AlphaGenerationService(None, {"m": placeholder_model_fixture}) #type: ignore
    with pytest.raises(ValueError): AlphaGenerationService(mock_mq_client_ags, {"m": placeholder_model_fixture}, input_topic="")
    with pytest.raises(ValueError): AlphaGenerationService(mock_mq_client_ags, {"m": placeholder_model_fixture}, output_topic="")
    with pytest.raises(TypeError): AlphaGenerationService(mock_mq_client_ags, {"m": "not_a_model"}) #type: ignore
    with pytest.raises(TypeError): AlphaGenerationService(mock_mq_client_ags, {123: placeholder_model_fixture}) #type: ignore


@pytest.mark.asyncio
async def test_ags_start_subscribes(alpha_service: AlphaGenerationService, mock_mq_client_ags: InMemoryMQClient):
    await alpha_service.start()
    mock_mq_client_ags.subscribe.assert_called_once_with(
        "engineered_features", alpha_service._on_features_received
    )

@pytest.mark.asyncio
async def test_ags_stop_unsubscribes(alpha_service: AlphaGenerationService, mock_mq_client_ags: InMemoryMQClient):
    await alpha_service.start() # Must start before stop
    await alpha_service.stop()
    mock_mq_client_ags.unsubscribe.assert_called_once_with(
        "engineered_features", alpha_service._on_features_received
    )

@pytest.mark.asyncio
async def test_ags_on_features_received_publishes_alpha(alpha_service: AlphaGenerationService, mock_mq_client_ags: InMemoryMQClient, placeholder_model_fixture: MoETransformerModel):
    # Ensure the model used by the service is the one we can mock or predict behavior for
    alpha_service.models[alpha_service.default_model_key] = placeholder_model_fixture
    
    # Mock the model's methods to control its output
    mock_signals = pd.DataFrame([{'signal_score': 0.75, 'predicted_movement_pct': 0.01}])
    mock_uncertainty = pd.DataFrame([{'prediction_variance': 0.05, 'confidence': 0.95}])
    
    # Patch the specific model instance used by the service for this test
    placeholder_model_fixture.preprocess_features = MagicMock(return_value=pd.DataFrame([{"processed_feature":1}])) # Ensure it's not empty
    placeholder_model_fixture.predict_alpha = MagicMock(return_value=(mock_signals, mock_uncertainty))
    
    feature_message = {
        "timestamp_ms": 1234567890,
        "symbol": "BTCUSD",
        "sma_20": 50000.0,
        "rsi_14": 60.0,
        "price_at_feature_generation": 50050.0, # Added this based on model's preprocess
        "source_event_id": "feat_event_123"
    }
    await alpha_service._on_features_received(feature_message)

    placeholder_model_fixture.preprocess_features.assert_called_once()
    placeholder_model_fixture.predict_alpha.assert_called_once()
    
    mock_mq_client_ags.publish.assert_called_once()
    args, _ = mock_mq_client_ags.publish.call_args
    topic, published_message = args
    
    assert topic == "alpha_signals"
    assert published_message["symbol"] == "BTCUSD"
    assert published_message["model_name"] == placeholder_model_fixture.model_name
    assert published_message["signals"] == mock_signals.to_dict(orient='records')[0]
    assert published_message["uncertainty"] == mock_uncertainty.to_dict(orient='records')[0]
    assert published_message["source_event_id"] == "feat_event_123"

@pytest.mark.asyncio
async def test_ags_on_features_received_no_model_found(alpha_service: AlphaGenerationService, mock_mq_client_ags: InMemoryMQClient):
    alpha_service.service_config["regime_model_mapping"]["normal"] = "non_existent_model"
    feature_message = {"timestamp_ms": 123, "symbol": "XYZ", "price_at_feature_generation": 1.0}
    await alpha_service._on_features_received(feature_message)
    mock_mq_client_ags.publish.assert_not_called()

@pytest.mark.asyncio
async def test_ags_on_features_received_model_predict_empty(alpha_service: AlphaGenerationService, mock_mq_client_ags: InMemoryMQClient, placeholder_model_fixture: MoETransformerModel):
    alpha_service.models[alpha_service.default_model_key] = placeholder_model_fixture
    placeholder_model_fixture.predict_alpha = MagicMock(return_value=(pd.DataFrame(), pd.DataFrame())) # Empty DFs
    placeholder_model_fixture.preprocess_features = MagicMock(return_value=pd.DataFrame([{"processed":1}]))


    feature_message = {"timestamp_ms": 123, "symbol": "ABC", "price_at_feature_generation": 1.0, "source_event_id": "empty_sig"}
    await alpha_service._on_features_received(feature_message)
    mock_mq_client_ags.publish.assert_not_called()

@pytest.mark.asyncio
async def test_ags_on_features_received_preprocess_empty(alpha_service: AlphaGenerationService, mock_mq_client_ags: InMemoryMQClient, placeholder_model_fixture: MoETransformerModel):
    alpha_service.models[alpha_service.default_model_key] = placeholder_model_fixture
    placeholder_model_fixture.preprocess_features = MagicMock(return_value=pd.DataFrame()) # Empty DF from preprocess

    feature_message = {"timestamp_ms": 123, "symbol": "DEF", "price_at_feature_generation": 1.0, "source_event_id": "empty_prep"}
    await alpha_service._on_features_received(feature_message)
    mock_mq_client_ags.publish.assert_not_called()

def test_ags_update_market_regime(alpha_service: AlphaGenerationService): # Added this test as it was in the prompt
    new_mapping = {"volatile": "alt_model_key"}
    alpha_service.update_market_regime("volatile", new_mapping)
    assert alpha_service.current_market_regime == "volatile"
    assert alpha_service.service_config["regime_model_mapping"] == new_mapping

    alpha_service.update_market_regime("stable_no_map_change") # No mapping change
    assert alpha_service.current_market_regime == "stable_no_map_change"
    assert alpha_service.service_config["regime_model_mapping"] == new_mapping # Should persist

    alpha_service.update_market_regime(123) # Invalid type
    assert alpha_service.current_market_regime == "stable_no_map_change" # Should not change
