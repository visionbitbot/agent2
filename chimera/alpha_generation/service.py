import asyncio
import pandas as pd

from chimera.core.message_queue.in_memory_mq import InMemoryMQClient
from chimera.alpha_generation.base_alpha_model import BaseAlphaModel 
# Assuming MoETransformerModel is the placeholder we'll use by default
from chimera.alpha_generation.moe_transformer_model import MoETransformerModel


class AlphaGenerationService:
    def __init__(self, mq_client: InMemoryMQClient, 
                 model_registry: dict[str, BaseAlphaModel] | None = None, 
                 input_topic: str = "engineered_features", 
                 output_topic: str = "alpha_signals",
                 default_model_key: str = "default_model"):
        
        if not isinstance(mq_client, InMemoryMQClient):
            raise TypeError("mq_client must be an InMemoryMQClient instance.")
        if not isinstance(input_topic, str) or not input_topic:
            raise ValueError("input_topic must be a non-empty string.")
        if not isinstance(output_topic, str) or not output_topic:
            raise ValueError("output_topic must be a non-empty string.")
        if not isinstance(default_model_key, str) or not default_model_key:
            raise ValueError("default_model_key must be a non-empty string.")

        self.mq_client = mq_client
        
        if model_registry is None:
            # Provide a default placeholder model if none given
            self.models: dict[str, BaseAlphaModel] = {default_model_key: MoETransformerModel()}
        else:
            if not isinstance(model_registry, dict) or \
               not all(isinstance(k, str) and isinstance(v, BaseAlphaModel) for k, v in model_registry.items()):
                raise TypeError("model_registry must be a dictionary of string keys and BaseAlphaModel instances.")
            self.models = model_registry
            
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.default_model_key = default_model_key
        
        self.current_market_regime = "normal" # Placeholder, would be updated by Orchestrator
        self.service_config = {"regime_model_mapping": {"normal": default_model_key}} # Default mapping

        self._running = False

    async def _on_features_received(self, features_message: dict):
        if not isinstance(features_message, dict) or "symbol" not in features_message:
            # print(f"AGS: Invalid features message: {features_message}") # Optional logging
            return

        features_df = pd.DataFrame([features_message]) # Models expect DataFrame

        selected_model_key = self.service_config.get("regime_model_mapping", {}).get(
            self.current_market_regime, self.default_model_key
        )
        model = self.models.get(selected_model_key)
        
        if not model:
            # print(f"AGS: No model found for key '{selected_model_key}' or regime '{self.current_market_regime}'.") # Optional
            return

        try:
            processed_data = model.preprocess_features(features_df)
            if processed_data.empty:
                # print(f"AGS: Preprocessing returned empty DataFrame for model {model.model_name}. Skipping prediction.") # Optional
                return
                
            alpha_signals, uncertainty = model.predict_alpha(processed_data)

            if alpha_signals.empty or uncertainty.empty:
                # print(f"AGS: Model {model.model_name} produced empty signals or uncertainty. Skipping publish.") # Optional
                return

            output_message = {
                "timestamp_ms": features_message["timestamp_ms"],
                "symbol": features_message["symbol"],
                "model_name": model.model_name,
                "model_version": model.version,
                "signals": alpha_signals.to_dict(orient='records')[0], # Assuming single row DF
                "uncertainty": uncertainty.to_dict(orient='records')[0], # Assuming single row DF
                "source_event_id": features_message.get("source_event_id", "N/A")
            }
            await self.mq_client.publish(self.output_topic, output_message)
        except Exception as e:
            # print(f"AGS: Error during alpha generation with model {model.model_name}: {e}") # Optional logging
            pass # Or log more robustly


    async def start(self):
        if self._running:
            return
        self._running = True
        await self.mq_client.subscribe(self.input_topic, self._on_features_received)
        # print(f"AlphaGenerationService started, listening on {self.input_topic}.") # Optional

    async def stop(self):
        if not self._running:
            return
        self._running = False
        # Add try-except as unsubscribe might fail if not subscribed or callback mismatch
        try:
            await self.mq_client.unsubscribe(self.input_topic, self._on_features_received)
        except Exception: # Broad exception for safety, could be more specific if MQ client raises specific errors
            pass
        # print("AlphaGenerationService stopped.") # Optional

    def update_market_regime(self, new_regime: str, regime_model_mapping: dict | None = None):
        if not isinstance(new_regime, str): return
        # print(f"AGS: Market regime updated to {new_regime}") # Optional
        self.current_market_regime = new_regime
        if regime_model_mapping and isinstance(regime_model_mapping, dict):
            self.service_config["regime_model_mapping"] = regime_model_mapping
