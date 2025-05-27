import asyncio
import pandas as pd
from collections import defaultdict

from chimera.core.message_queue.in_memory_mq import InMemoryMQClient
from chimera.feature_engineering.alpha_factors import technical_indicators

class FeatureEngineeringService:
    def __init__(self, mq_client: InMemoryMQClient, 
                 input_topics: dict, output_topic: str = "engineered_features",
                 sma_periods: list[int] | None = None, 
                 rsi_periods: list[int] | None = None):
        
        if not isinstance(mq_client, InMemoryMQClient):
            raise TypeError("mq_client must be an InMemoryMQClient instance.")
        if not isinstance(input_topics, dict) or not input_topics:
            raise ValueError("input_topics must be a non-empty dictionary.")
        if not isinstance(output_topic, str) or not output_topic:
            raise ValueError("output_topic must be a non-empty string.")

        self.mq_client = mq_client
        self.input_topics = input_topics 
        self.output_topic = output_topic
        
        self.sma_periods = sma_periods if sma_periods else [10, 20] # Default SMA periods
        self.rsi_periods = rsi_periods if rsi_periods else [14]    # Default RSI periods

        self._running = False
        self._processing_tasks: list[asyncio.Task] = []
        
        # Buffer to store recent market data for windowed calculations
        # {symbol: {price: [list of prices], volume: [list of volumes]}}
        self.market_data_buffer = defaultdict(lambda: defaultdict(list))
        
        # Corrected max_buffer_size calculation
        max_sma = max(self.sma_periods, default=0)
        max_rsi = max(self.rsi_periods, default=0)
        self.max_buffer_size = max(max_sma, max_rsi + 1 if max_rsi > 0 else 0, 50)


    async def _process_market_data(self, message: dict):
        # Corrected check for message content
        if not isinstance(message, dict) or \
           "symbol" not in message or \
           "price" not in message or \
           "timestamp_ms" not in message:
            # print(f"FES: Invalid market data message received: {message}") # Optional logging
            return

        symbol = message["symbol"]
        price = message.get("price") # Already checked "price" in message above
        
        # Update buffer
        self.market_data_buffer[symbol]["price"].append(price)
        # Keep buffer size manageable
        if len(self.market_data_buffer[symbol]["price"]) > self.max_buffer_size:
            self.market_data_buffer[symbol]["price"].pop(0) # Remove oldest price

        # Calculate features
        features = {
            "timestamp_ms": message["timestamp_ms"], 
            "symbol": symbol, 
            "source_event_id": message.get("id", "N/A"),
            "price_at_feature_generation": price # Include current price for context
        }
        
        current_prices = self.market_data_buffer[symbol]["price"]

        for period in self.sma_periods:
            sma_value = technical_indicators.calculate_sma(current_prices, period)
            if sma_value is not None:
                features[f"sma_{period}"] = sma_value
        
        for period in self.rsi_periods:
            rsi_value = technical_indicators.calculate_rsi(current_prices, period)
            if rsi_value is not None:
                features[f"rsi_{period}"] = rsi_value
        
        # Only publish if some actual features (beyond the basic 4 info fields) were generated
        if len(features) > 4: 
            await self.mq_client.publish(self.output_topic, features)

    async def _message_handler_wrapper(self, topic: str):
        # print(f"FES: Message handler started for topic {topic}") # Optional
        # This method is a placeholder if a more complex handler routing logic was needed.
        # In the current design, mq_client.subscribe directly uses _process_market_data (or similar)
        # as the callback, so this wrapper isn't strictly necessary for a single message type.
        # It's kept for potential future expansion where one service might handle multiple message types
        # from the same topic or route messages from different topics to different processors.
        while self._running:
            try:
                # This loop would typically involve an await on a queue specific to this handler,
                # if the service managed its own internal queues per topic/handler.
                # However, with InMemoryMQClient's direct callback system, this method
                # would be part of the callback chain or the callback itself.
                # For now, let it be a conceptual placeholder.
                await asyncio.sleep(0.1) # Keep alive, but actual work is event-driven by MQ callbacks
            except asyncio.CancelledError:
                # print(f"FES: Handler for {topic} cancelled.") # Optional
                break
            except Exception as e:
                # print(f"FES: Error in handler for {topic}: {e}") # Optional
                await asyncio.sleep(1) # Avoid tight loop on error

    async def start(self):
        if self._running:
            return
        self._running = True
        
        # Subscribe to relevant input topics
        market_data_topic = self.input_topics.get("market_data")
        if market_data_topic:
            # The callback for the message queue is _process_market_data
            await self.mq_client.subscribe(market_data_topic, self._process_market_data)
            # print(f"FES: Subscribed to {market_data_topic}") # Optional

        # Example for starting conceptual handler tasks if they were managing internal queues:
        # for topic_type, topic_name in self.input_topics.items():
        #     if topic_type == "market_data":
        #         task = asyncio.create_task(self._message_handler_wrapper(topic_name))
        #         self._processing_tasks.append(task)
        
        # print("FeatureEngineeringService started.") # Optional

    async def stop(self):
        if not self._running:
            return
        self._running = False
        
        # Unsubscribe from topics
        market_data_topic = self.input_topics.get("market_data")
        if market_data_topic:
            try: # Add try-except as unsubscribe might fail if not subscribed
                await self.mq_client.unsubscribe(market_data_topic, self._process_market_data)
                # print(f"FES: Unsubscribed from {market_data_topic}") # Optional
            except Exception as e: # Catching generic Exception, could be more specific
                pass # print(f"FES: Error unsubscribing from {market_data_topic}: {e}") # Optional
            
        # Cancel any running processing tasks (if they were used)
        for task in self._processing_tasks:
            if not task.done():
                task.cancel()
        try:
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass # print("FES: Processing tasks cancelled during stop.") # Optional
        self._processing_tasks.clear()
        
        # print("FeatureEngineeringService stopped.") # Optional
