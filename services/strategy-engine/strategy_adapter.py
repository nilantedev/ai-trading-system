#!/usr/bin/env python3
"""
Strategy Adapter - Adds evaluate() method to all strategies for backtesting compatibility
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "../../../shared/python-common"))

import numpy as np
from typing import Dict, Any
from datetime import datetime
from trading_common import get_logger

logger = get_logger(__name__)


class StrategyAdapter:
    """
    Wraps strategies that don't have evaluate() method and provides a standardized interface
    """
    
    def __init__(self, strategy, strategy_name: str):
        self.strategy = strategy
        self.strategy_name = strategy_name
        self.name = strategy_name
        
    async def evaluate(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardized evaluate method for backtesting
        
        Converts various strategy interfaces to common format:
        - generate_signals() -> evaluate()
        - generate_signal() -> evaluate()
        - generate_quotes() -> evaluate()
        """
        
        try:
            # Check if strategy has native evaluate method
            if hasattr(self.strategy, 'evaluate'):
                return await self.strategy.evaluate(symbol, data)
            
            # Try generate_signal (singular)
            if hasattr(self.strategy, 'generate_signal'):
                result = await self.strategy.generate_signal(symbol, data)
                return self._normalize_signal(result, symbol)
            
            # Try generate_signals (plural)
            if hasattr(self.strategy, 'generate_signals'):
                result = await self.strategy.generate_signals([symbol], data)
                if isinstance(result, dict) and symbol in result:
                    return self._normalize_signal(result[symbol], symbol)
                return self._normalize_signal(result, symbol)
            
            # Try generate_quotes (market making)
            if hasattr(self.strategy, 'generate_quotes'):
                quotes = await self.strategy.generate_quotes(symbol, data)
                # Market making generates bid/ask, convert to directional signal
                if quotes and 'bid' in quotes and 'ask' in quotes:
                    mid = (quotes['bid'] + quotes['ask']) / 2
                    current_price = data.get('close', [0])[-1] if isinstance(data.get('close'), (list, np.ndarray)) else data.get('price', mid)
                    
                    # Simple directional bias based on spread
                    spread = quotes['ask'] - quotes['bid']
                    if current_price < quotes['bid']:
                        action = "BUY"
                        confidence = 0.6
                    elif current_price > quotes['ask']:
                        action = "SELL"
                        confidence = 0.6
                    else:
                        action = "HOLD"
                        confidence = 0.3
                    
                    return {
                        "strategy": self.strategy_name,
                        "strategy_name": self.strategy_name,
                        "symbol": symbol,
                        "signal_type": action,
                        "recommended_action": action,
                        "confidence": confidence,
                        "position_size": 0.05 if action != "HOLD" else 0.0,
                        "reasoning": f"Market making: bid={quotes['bid']:.2f}, ask={quotes['ask']:.2f}, spread={spread:.4f}",
                        "indicators": quotes,
                        "timestamp": datetime.utcnow().isoformat()
                    }
            
            # No compatible method found - return HOLD
            logger.warning(f"Strategy {self.strategy_name} has no compatible signal generation method")
            return self._no_signal(symbol, "Strategy interface not compatible")
            
        except Exception as e:
            logger.error(f"Strategy adapter error for {self.strategy_name}: {e}", exc_info=True)
            return self._no_signal(symbol, f"Error: {str(e)}")
    
    def _normalize_signal(self, signal: Any, symbol: str) -> Dict[str, Any]:
        """Normalize various signal formats to standard format"""
        
        if isinstance(signal, dict):
            # Already in correct format
            if 'signal_type' in signal or 'recommended_action' in signal:
                # Ensure all required fields exist
                signal.setdefault('strategy', self.strategy_name)
                signal.setdefault('strategy_name', self.strategy_name)
                signal.setdefault('symbol', symbol)
                signal.setdefault('confidence', 0.5)
                signal.setdefault('position_size', 0.05)
                signal.setdefault('timestamp', datetime.utcnow().isoformat())
                
                # Normalize signal_type
                if 'signal_type' not in signal and 'recommended_action' in signal:
                    signal['signal_type'] = signal['recommended_action']
                elif 'recommended_action' not in signal and 'signal_type' in signal:
                    signal['recommended_action'] = signal['signal_type']
                
                return signal
            
            # Convert numeric signal (-1, 0, 1) to action
            if 'signal' in signal:
                sig_val = signal['signal']
                if sig_val > 0.5:
                    action = "BUY"
                elif sig_val < -0.5:
                    action = "SELL"
                else:
                    action = "HOLD"
                
                return {
                    "strategy": self.strategy_name,
                    "strategy_name": self.strategy_name,
                    "symbol": symbol,
                    "signal_type": action,
                    "recommended_action": action,
                    "confidence": abs(sig_val),
                    "position_size": abs(sig_val) * 0.1,
                    "reasoning": signal.get('reason', 'Signal value: ' + str(sig_val)),
                    "indicators": signal,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        # Assume numeric signal (-1, 0, 1)
        if isinstance(signal, (int, float)):
            if signal > 0.5:
                action = "BUY"
            elif signal < -0.5:
                action = "SELL"
            else:
                action = "HOLD"
            
            return {
                "strategy": self.strategy_name,
                "strategy_name": self.strategy_name,
                "symbol": symbol,
                "signal_type": action,
                "recommended_action": action,
                "confidence": abs(signal),
                "position_size": abs(signal) * 0.1,
                "reasoning": f"Signal value: {signal}",
                "indicators": {},
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Unknown format - return HOLD
        return self._no_signal(symbol, "Unknown signal format")
    
    def _no_signal(self, symbol: str, reason: str) -> Dict[str, Any]:
        """Return no-signal response"""
        return {
            "strategy": self.strategy_name,
            "strategy_name": self.strategy_name,
            "symbol": symbol,
            "signal_type": "HOLD",
            "recommended_action": "HOLD",
            "confidence": 0.0,
            "position_size": 0.0,
            "reasoning": reason,
            "indicators": {},
            "timestamp": datetime.utcnow().isoformat()
        }
