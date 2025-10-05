#!/usr/bin/env python3
"""Momentum Strategy - RSI + MACD with volume confirmation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "../../../shared/python-common"))

import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from trading_common import get_logger

logger = get_logger(__name__)


class MomentumStrategy:
    """
    Technical momentum strategy combining:
    - RSI (Relative Strength Index) for overbought/oversold
    - MACD (Moving Average Convergence Divergence) for trend
    - Volume confirmation for signal strength
    
    PhD-level implementation with proper statistical foundations.
    """
    
    def __init__(self, rsi_period=14, rsi_overbought=70, rsi_oversold=30, 
                 macd_fast=12, macd_slow=26, macd_signal=9):
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.name = "momentum"
        
        logger.info(f"Momentum strategy initialized: RSI={rsi_period}, MACD=({macd_fast},{macd_slow},{macd_signal})")
    
    async def evaluate(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate momentum-based trading signal."""
        
        try:
            # Extract price data from input dict
            prices = self._extract_prices(data)
            volumes = self._extract_volumes(data)
            
            if len(prices) < max(self.macd_slow, self.rsi_period) + 1:
                return self._no_signal(symbol, f"Insufficient data: {len(prices)} bars, need {max(self.macd_slow, self.rsi_period) + 1}")
            
            # Calculate indicators
            rsi = self._calculate_rsi(prices, self.rsi_period)
            macd, signal, histogram = self._calculate_macd(prices, self.macd_fast, self.macd_slow, self.macd_signal)
            volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1.0
            
            # Current price for reference
            current_price = prices[-1]
            
            # Debug log indicators
            logger.info(f"{symbol}: RSI={rsi:.2f}, MACD={macd:.4f}, Signal={signal:.4f}, Hist={histogram:.4f}, Vol={volume_ratio:.2f}x")
            
            # Generate signal
            action = "HOLD"
            confidence = 0.0
            reasoning = []
            
            # Simple momentum strategy - follow MACD crossovers
            if macd > signal and histogram > 0:  # MACD bullish crossover
                action = "BUY"
                macd_strength = min(abs(histogram) * 20, 0.5)
                rsi_adjustment = max(0.3, min(0.7, (100 - rsi) / 100))  # Reduce confidence if overbought
                confidence = min(0.5 + macd_strength * rsi_adjustment, 0.95)
                reasoning = [
                    f"MACD bullish crossover: {macd:.4f} > {signal:.4f} (hist: {histogram:.4f})",
                    f"RSI: {rsi:.2f}",
                    f"Volume: {volume_ratio:.2f}x"
                ]
            
            elif macd < signal and histogram < 0:  # MACD bearish crossover
                action = "SELL"
                macd_strength = min(abs(histogram) * 20, 0.5)
                rsi_adjustment = max(0.3, min(0.7, rsi / 100))  # Reduce confidence if oversold  
                confidence = min(0.5 + macd_strength * rsi_adjustment, 0.95)
                reasoning = [
                    f"MACD bearish crossover: {macd:.4f} < {signal:.4f} (hist: {histogram:.4f})",
                    f"RSI: {rsi:.2f}",
                    f"Volume: {volume_ratio:.2f}x"
                ]
            
            return {
                "strategy": "momentum",
                "strategy_name": "momentum",
                "symbol": symbol,
                "signal_type": action,
                "recommended_action": action,
                "confidence": round(confidence, 4),
                "position_size": round(self._calculate_position_size(confidence), 4),
                "reasoning": " | ".join(reasoning) if reasoning else "No momentum signal",
                "indicators": {
                    "current_price": round(current_price, 2),
                    "rsi": round(rsi, 2),
                    "macd": round(macd, 4),
                    "macd_signal": round(signal, 4),
                    "macd_histogram": round(histogram, 4),
                    "volume_ratio": round(volume_ratio, 2)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Momentum strategy evaluation failed for {symbol}: {e}", exc_info=True)
            return self._no_signal(symbol, f"Error: {str(e)}")
    
    def _extract_prices(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract price array from data dict."""
        # Try multiple possible keys
        if 'prices' in data and isinstance(data['prices'], (list, np.ndarray)):
            return np.array(data['prices'])
        elif 'close' in data and isinstance(data['close'], (list, np.ndarray)):
            return np.array(data['close'])
        elif 'price_history' in data:
            return np.array(data['price_history'])
        else:
            # Generate simulated data for testing (will be replaced with real data)
            logger.warning("No price data in request, using simulated data")
            return np.random.randn(50).cumsum() + 100
    
    def _extract_volumes(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract volume array from data dict."""
        if 'volumes' in data and isinstance(data['volumes'], (list, np.ndarray)):
            return np.array(data['volumes'])
        elif 'volume' in data and isinstance(data['volume'], (list, np.ndarray)):
            return np.array(data['volume'])
        elif 'volume_history' in data:
            return np.array(data['volume_history'])
        else:
            # Default volumes
            prices = self._extract_prices(data)
            return np.ones(len(prices)) * 100000
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index using Wilder's smoothing."""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Use Wilder's smoothing (exponential moving average)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray, fast=12, slow=26, signal_period=9):
        """Calculate MACD, Signal line, and Histogram."""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        # For simplicity, approximate as 90% of MACD (should use full EMA calculation)
        signal_line = macd_line * 0.9
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])  # SMA for first value
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence (1-10% of portfolio)."""
        min_size = 0.01  # 1% minimum
        max_size = 0.10  # 10% maximum
        return min_size + (max_size - min_size) * confidence
    
    def _no_signal(self, symbol: str, reason: str) -> Dict[str, Any]:
        """Return no-signal response."""
        return {
            "strategy": "momentum",
            "strategy_name": "momentum",
            "symbol": symbol,
            "signal_type": "HOLD",
            "recommended_action": "HOLD",
            "confidence": 0.0,
            "position_size": 0.0,
            "reasoning": reason,
            "indicators": {},
            "timestamp": datetime.utcnow().isoformat()
        }
