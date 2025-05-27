import asyncio
import pandas as pd 
from typing import Optional, Dict, Any

from chimera.core.message_queue.in_memory_mq import InMemoryMQClient
from chimera.decision_rl.action_spaces import TradeAction, PrimaryAction, OrderTypeHint


class RLDecisionService:
    def __init__(self, mq_client: InMemoryMQClient, 
                 agent_config: dict | None = None, 
                 input_topic_signals: str = "alpha_signals",
                 input_topic_portfolio: str = "portfolio_state", 
                 output_topic_actions: str = "trade_actions",
                 decision_thresholds: dict | None = None): 
        
        if not isinstance(mq_client, InMemoryMQClient):
            raise TypeError("mq_client must be an InMemoryMQClient instance.")
        if not isinstance(input_topic_signals, str) or not input_topic_signals:
            raise ValueError("input_topic_signals must be a non-empty string.")
        if not isinstance(input_topic_portfolio, str) or not input_topic_portfolio:
            raise ValueError("input_topic_portfolio must be a non-empty string.")
        if not isinstance(output_topic_actions, str) or not output_topic_actions:
            raise ValueError("output_topic_actions must be a non-empty string.")

        self.mq_client = mq_client
        self.agent_config = agent_config if agent_config is not None else {}
        
        self.input_topic_signals = input_topic_signals
        self.input_topic_portfolio = input_topic_portfolio
        self.output_topic_actions = output_topic_actions
        
        self.decision_thresholds = decision_thresholds if decision_thresholds is not None else {
            "enter_long_score": 0.7, "enter_long_confidence": 0.6,
            "enter_short_score": -0.7, "enter_short_confidence": 0.6,
            "exit_long_score": -0.3, "exit_long_confidence": 0.5,
            "exit_short_score": 0.3, "exit_short_confidence": 0.5,
            "default_exposure_pct": 0.05, 
            "default_sl_atr_multiplier": 1.5,
            "default_tp_atr_multiplier": 3.0,
        }
        
        self.current_portfolio_state: Dict[str, Dict[str, Any]] = {} 
        self._running = False

    async def _on_alpha_signal(self, signal_message: dict):
        if not isinstance(signal_message, dict) or "symbol" not in signal_message            or "signals" not in signal_message or "uncertainty" not in signal_message:
            return

        symbol = signal_message["symbol"]
        signal_data = signal_message.get("signals", {}) 
        uncertainty_data = signal_message.get("uncertainty", {})

        trade_action = self._placeholder_decision_logic(symbol, signal_data, uncertainty_data, signal_message.get("source_event_id"))
        
        if trade_action and trade_action.primary_action != PrimaryAction.DO_NOTHING:
            await self.mq_client.publish(self.output_topic_actions, trade_action.to_dict())

    def _placeholder_decision_logic(self, symbol: str, signal: Dict[str, Any], 
                                   uncertainty: Dict[str, Any], source_id: Optional[str]) -> Optional[TradeAction]:
        
        signal_score = signal.get("signal_score", 0.0)
        confidence = uncertainty.get("confidence", 0.0)
        
        current_position_info = self.current_portfolio_state.get(symbol, {"size": 0.0}) 
        current_pos_size = current_position_info.get("size", 0.0)

        primary_action = PrimaryAction.DO_NOTHING
        target_exposure = self.decision_thresholds["default_exposure_pct"]
        
        if signal_score > self.decision_thresholds["enter_long_score"] and            confidence > self.decision_thresholds["enter_long_confidence"]:
            if current_pos_size <= 0: 
                primary_action = PrimaryAction.ENTER_LONG
        elif signal_score < self.decision_thresholds["enter_short_score"] and              confidence > self.decision_thresholds["enter_short_confidence"]:
            if current_pos_size >= 0: 
                primary_action = PrimaryAction.ENTER_SHORT
        elif current_pos_size > 0 and              signal_score < self.decision_thresholds["exit_long_score"] and              confidence > self.decision_thresholds["exit_long_confidence"]:
            primary_action = PrimaryAction.EXIT_LONG
            target_exposure = 0.0 
        elif current_pos_size < 0 and              signal_score > self.decision_thresholds["exit_short_score"] and              confidence > self.decision_thresholds["exit_short_confidence"]:
            primary_action = PrimaryAction.EXIT_SHORT
            target_exposure = 0.0 

        if primary_action == PrimaryAction.DO_NOTHING:
            return None

        return TradeAction(
            symbol=symbol,
            primary_action=primary_action,
            target_exposure_pct=target_exposure,
            stop_loss_atr_multiplier=self.decision_thresholds["default_sl_atr_multiplier"],
            take_profit_atr_multiplier=self.decision_thresholds["default_tp_atr_multiplier"],
            source_signal_id=source_id,
            confidence_score=confidence, 
            action_metadata={"agent_type": "placeholder_v1_threshold"}
        )

    async def _on_portfolio_update(self, portfolio_message: dict):
        if not isinstance(portfolio_message, dict) or "positions" not in portfolio_message:
            return
            
        self.current_portfolio_state = portfolio_message.get("positions", {})

    async def start(self):
        if self._running: return
        self._running = True
        await self.mq_client.subscribe(self.input_topic_signals, self._on_alpha_signal)
        await self.mq_client.subscribe(self.input_topic_portfolio, self._on_portfolio_update)

    async def stop(self):
        if not self._running: return
        self._running = False
        await self.mq_client.unsubscribe(self.input_topic_signals, self._on_alpha_signal)
        await self.mq_client.unsubscribe(self.input_topic_portfolio, self._on_portfolio_update)
