import asyncio
from typing import Dict, Any
import pandas as pd # For Timestamp fallback

from chimera.core.message_queue.in_memory_mq import InMemoryMQClient
from chimera.decision_rl.action_spaces import TradeAction # For type hinting and reconstruction
from chimera.execution.simulated_exchange import SimulatedExchange 

class ExecutionService:
    def __init__(self, mq_client: InMemoryMQClient, simulated_exchange: SimulatedExchange,
                 input_topic_approved_actions: str = "approved_trade_actions",
                 output_topic_confirmations: str = "execution_confirmations",
                 market_data_topic: str = "raw_market_data"): # Needs market data for execution context
        
        if not isinstance(mq_client, InMemoryMQClient):
            raise TypeError("mq_client must be an InMemoryMQClient instance.")
        if not isinstance(simulated_exchange, SimulatedExchange):
            raise TypeError("simulated_exchange must be a SimulatedExchange instance.")
        if not isinstance(input_topic_approved_actions, str) or not input_topic_approved_actions:
            raise ValueError("input_topic_approved_actions must be a non-empty string.")
        if not isinstance(output_topic_confirmations, str) or not output_topic_confirmations:
            raise ValueError("output_topic_confirmations must be a non-empty string.")
        if not isinstance(market_data_topic, str) or not market_data_topic:
            raise ValueError("market_data_topic must be a non-empty string.")

        self.mq_client = mq_client
        self.simulated_exchange = simulated_exchange
        self.input_topic_approved_actions = input_topic_approved_actions
        self.output_topic_confirmations = output_topic_confirmations
        self.market_data_topic = market_data_topic
        
        self.latest_market_data: Dict[str, Dict[str, Any]] = {} # Store latest tick per symbol: symbol -> tick_data
        self._running = False
        self._processing_task_approved_actions: asyncio.Task | None = None
        self._processing_task_market_data: asyncio.Task | None = None


    async def _on_market_data(self, market_data_tick: Dict[str, Any]):
        if not isinstance(market_data_tick, dict) or "symbol" not in market_data_tick:
            # print(f"ES: Invalid market data tick received: {market_data_tick}") # Optional
            return
        symbol = market_data_tick["symbol"]
        self.latest_market_data[symbol] = market_data_tick

    async def _on_approved_action(self, approved_action_dict: Dict[str, Any]):
        if not isinstance(approved_action_dict, dict):
            # print(f"ES: Invalid approved action format: {approved_action_dict}") # Optional
            return
        
        try:
            # Reconstruct TradeAction. from_dict was added to TradeAction in previous step's code.
            # If not, it would be: trade_action = TradeAction(**approved_action_dict)
            # Need to ensure enums are handled if they were stringified by .to_dict()
            trade_action = TradeAction.from_dict(approved_action_dict)
        except Exception as e:
            # print(f"ES: Failed to reconstruct TradeAction from dict: {approved_action_dict}. Error: {e}") # Optional
            # Potentially publish a failed "malformed_action_event"
            return
            
        symbol = trade_action.symbol
        market_context = self.latest_market_data.get(symbol)

        # Use UTC timestamp for consistency
        current_time_ms = int(pd.Timestamp.now(tz='UTC').timestamp() * 1000)


        if not market_context:
            # print(f"ES: No market data context for {symbol} to execute {trade_action.primary_action}. Skipping.") # Optional
            failed_report = {
                "timestamp_ms": current_time_ms, 
                "symbol": symbol,
                "action_type": str(trade_action.primary_action), # Use str representation
                "status": "REJECTED_NO_MARKET_DATA",
                "reason": f"No market data context available for symbol {symbol}",
                "requested_action": approved_action_dict,
                "source_signal_id": trade_action.source_signal_id
            }
            await self.mq_client.publish(self.output_topic_confirmations, failed_report)
            return
        
        # --- Latency Simulation ---
        simulated_delay = self.simulated_exchange.config.get("simulated_latency_sec", 0.0)
        if simulated_delay > 0:
            await asyncio.sleep(simulated_delay) 

        execution_report = self.simulated_exchange.execute_trade(trade_action, market_context)

        if execution_report: 
            execution_report["status"] = "FILLED" 
            if "timestamp_ms" not in execution_report: # Ensure timestamp from exchange or market context
                 execution_report["timestamp_ms"] = market_context.get("timestamp_ms", current_time_ms)
            await self.mq_client.publish(self.output_topic_confirmations, execution_report)
            # print(f"ES: Published FILLED confirmation for {symbol} {trade_action.primary_action}.") # Optional
        else:
            # print(f"ES: Trade for {symbol} ({trade_action.primary_action}) was not executed by sim exchange.") # Optional
            failed_report = {
                "timestamp_ms": market_context.get("timestamp_ms", current_time_ms),
                "symbol": symbol,
                "action_type": str(trade_action.primary_action),
                "status": "REJECTED_BY_EXCHANGE", 
                "reason": "Simulated exchange did not fill the order (e.g., limit price not met, insufficient liquidity simulation).",
                "requested_action": approved_action_dict,
                "fill_price": None, 
                "trade_size_units": 0, 
                "fees": 0, 
                "source_signal_id": trade_action.source_signal_id
            }
            await self.mq_client.publish(self.output_topic_confirmations, failed_report)


    async def start(self):
        if self._running: return
        self._running = True
        await self.mq_client.subscribe(self.input_topic_approved_actions, self._on_approved_action)
        await self.mq_client.subscribe(self.market_data_topic, self._on_market_data)
        # print(f"ExecutionService started. Listening on {self.input_topic_approved_actions} and {self.market_data_topic}.") # Optional

    async def stop(self):
        if not self._running: return
        self._running = False
        # Add try-except for robustness, though InMemoryMQClient's unsubscribe is currently simple
        try:
            await self.mq_client.unsubscribe(self.input_topic_approved_actions, self._on_approved_action)
        except Exception: 
            pass 
        try:
            await self.mq_client.unsubscribe(self.market_data_topic, self._on_market_data)
        except Exception:
            pass
        # print("ExecutionService stopped.") # Optional
