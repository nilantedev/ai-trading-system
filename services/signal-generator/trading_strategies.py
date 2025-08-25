#!/usr/bin/env python3
"""Trading Strategies - Strategy implementations for signal generation."""

import math
import statistics
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from trading_common import MarketData


class SignalType(Enum):
    """Types of trading signals."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class TradingSignal:
    """Trading signal with metadata."""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0-1
    strength: float  # 0-1
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_name: str = ""
    reasoning: str = ""
    timestamp: datetime = datetime.utcnow()
    expiry_time: Optional[datetime] = None
    risk_score: float = 0.5  # 0-1, higher = more risky


class BaseStrategy:
    """Base class for trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.min_data_points = 20
    
    def can_generate_signal(self, market_data_history: List[MarketData]) -> bool:
        """Check if strategy has enough data to generate signals."""
        return len(market_data_history) >= self.min_data_points
    
    def generate_signal(self, symbol: str, market_data_history: List[MarketData], 
                       indicators: Dict = None) -> Optional[TradingSignal]:
        """Generate trading signal based on strategy logic."""
        raise NotImplementedError
    
    def calculate_position_size(self, signal: TradingSignal, portfolio_value: float, 
                              risk_per_trade: float = 0.02) -> float:
        """Calculate position size based on risk management."""
        if not signal.stop_loss or signal.stop_loss <= 0:
            return portfolio_value * 0.01  # Conservative 1% if no stop loss
        
        current_price = signal.target_price or 0
        if current_price <= 0:
            return 0
        
        risk_amount = portfolio_value * risk_per_trade
        price_risk = abs(current_price - signal.stop_loss)
        
        if price_risk <= 0:
            return 0
        
        shares = risk_amount / price_risk
        position_value = shares * current_price
        
        # Cap at maximum position size (e.g., 5% of portfolio)
        max_position_value = portfolio_value * 0.05
        
        return min(position_value, max_position_value)


class MovingAverageCrossoverStrategy(BaseStrategy):
    """Moving Average Crossover Strategy."""
    
    def __init__(self, short_period: int = 20, long_period: int = 50):
        super().__init__("MA_Crossover")
        self.short_period = short_period
        self.long_period = long_period
        self.min_data_points = max(short_period, long_period) + 5
    
    def generate_signal(self, symbol: str, market_data_history: List[MarketData], 
                       indicators: Dict = None) -> Optional[TradingSignal]:
        """Generate signal based on moving average crossover."""
        
        if not self.can_generate_signal(market_data_history):
            return None
        
        prices = [data.close for data in market_data_history]
        
        if len(prices) < self.long_period:
            return None
        
        # Calculate moving averages
        short_ma = statistics.mean(prices[-self.short_period:])
        long_ma = statistics.mean(prices[-self.long_period:])
        
        # Previous MA values for crossover detection
        prev_short_ma = statistics.mean(prices[-(self.short_period+1):-1])
        prev_long_ma = statistics.mean(prices[-(self.long_period+1):-1])
        
        current_price = prices[-1]
        
        # Detect crossover
        signal_type = SignalType.HOLD
        confidence = 0.0
        reasoning = "No significant moving average crossover detected"
        
        # Golden Cross (bullish)
        if short_ma > long_ma and prev_short_ma <= prev_long_ma:
            signal_type = SignalType.BUY
            confidence = min((short_ma - long_ma) / long_ma * 100, 1.0)  # Normalize
            reasoning = f"Golden cross detected: {self.short_period}-MA crossed above {self.long_period}-MA"
        
        # Death Cross (bearish)
        elif short_ma < long_ma and prev_short_ma >= prev_long_ma:
            signal_type = SignalType.SELL
            confidence = min((long_ma - short_ma) / long_ma * 100, 1.0)
            reasoning = f"Death cross detected: {self.short_period}-MA crossed below {self.long_period}-MA"
        
        # Strong trend continuation
        elif short_ma > long_ma * 1.02:  # 2% above
            signal_type = SignalType.BUY
            confidence = min((short_ma - long_ma) / long_ma * 50, 0.7)
            reasoning = f"Strong bullish trend: {self.short_period}-MA significantly above {self.long_period}-MA"
        
        elif short_ma < long_ma * 0.98:  # 2% below
            signal_type = SignalType.SELL
            confidence = min((long_ma - short_ma) / long_ma * 50, 0.7)
            reasoning = f"Strong bearish trend: {self.short_period}-MA significantly below {self.long_period}-MA"
        
        if signal_type == SignalType.HOLD:
            return None
        
        # Calculate stop loss and take profit
        atr = self._calculate_atr(market_data_history[-20:])  # 20-period ATR
        
        if signal_type == SignalType.BUY:
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 3)
        else:  # SELL
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 3)
        
        # Calculate risk score
        volatility = atr / current_price if current_price > 0 else 0
        risk_score = min(volatility * 10, 1.0)  # Normalize volatility to risk score
        
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            strength=confidence,
            target_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=self.name,
            reasoning=reasoning,
            expiry_time=datetime.utcnow() + timedelta(hours=4),  # 4-hour expiry
            risk_score=risk_score
        )
    
    def _calculate_atr(self, market_data: List[MarketData]) -> float:
        """Calculate Average True Range."""
        if len(market_data) < 2:
            return market_data[-1].high - market_data[-1].low if market_data else 0
        
        true_ranges = []
        
        for i in range(1, len(market_data)):
            high = market_data[i].high
            low = market_data[i].low
            prev_close = market_data[i-1].close
            
            true_range = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(true_range)
        
        return statistics.mean(true_ranges) if true_ranges else 0


class RSIMeanReversionStrategy(BaseStrategy):
    """RSI-based mean reversion strategy."""
    
    def __init__(self, rsi_period: int = 14, oversold_threshold: float = 30, 
                 overbought_threshold: float = 70):
        super().__init__("RSI_MeanReversion")
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.min_data_points = rsi_period + 10
    
    def generate_signal(self, symbol: str, market_data_history: List[MarketData], 
                       indicators: Dict = None) -> Optional[TradingSignal]:
        """Generate signal based on RSI mean reversion."""
        
        if not self.can_generate_signal(market_data_history):
            return None
        
        prices = [data.close for data in market_data_history]
        rsi = self._calculate_rsi(prices, self.rsi_period)
        
        if rsi is None:
            return None
        
        current_price = prices[-1]
        signal_type = SignalType.HOLD
        confidence = 0.0
        reasoning = "RSI in neutral range"
        
        # Oversold condition (potential buy)
        if rsi <= self.oversold_threshold:
            signal_type = SignalType.BUY
            confidence = (self.oversold_threshold - rsi) / self.oversold_threshold
            reasoning = f"RSI oversold at {rsi:.1f}, expecting mean reversion"
        
        # Overbought condition (potential sell)
        elif rsi >= self.overbought_threshold:
            signal_type = SignalType.SELL
            confidence = (rsi - self.overbought_threshold) / (100 - self.overbought_threshold)
            reasoning = f"RSI overbought at {rsi:.1f}, expecting mean reversion"
        
        if signal_type == SignalType.HOLD:
            return None
        
        # Calculate stop loss and take profit based on recent volatility
        recent_highs = [data.high for data in market_data_history[-10:]]
        recent_lows = [data.low for data in market_data_history[-10:]]
        price_range = max(recent_highs) - min(recent_lows)
        
        if signal_type == SignalType.BUY:
            stop_loss = current_price - (price_range * 0.5)
            take_profit = current_price + (price_range * 0.3)
        else:  # SELL
            stop_loss = current_price + (price_range * 0.5)
            take_profit = current_price - (price_range * 0.3)
        
        # Risk score based on how extreme RSI is
        if signal_type == SignalType.BUY:
            risk_score = max(0.2, (30 - rsi) / 30)  # Lower RSI = higher risk
        else:
            risk_score = max(0.2, (rsi - 70) / 30)  # Higher RSI = higher risk
        
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            strength=confidence,
            target_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=self.name,
            reasoning=reasoning,
            expiry_time=datetime.utcnow() + timedelta(hours=2),  # 2-hour expiry for mean reversion
            risk_score=risk_score
        )
    
    def _calculate_rsi(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return None
        
        avg_gain = statistics.mean(gains[-period:])
        avg_loss = statistics.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class BreakoutStrategy(BaseStrategy):
    """Breakout strategy based on support/resistance levels."""
    
    def __init__(self, lookback_period: int = 20, breakout_threshold: float = 0.02):
        super().__init__("Breakout")
        self.lookback_period = lookback_period
        self.breakout_threshold = breakout_threshold  # 2% breakout
        self.min_data_points = lookback_period + 5
    
    def generate_signal(self, symbol: str, market_data_history: List[MarketData], 
                       indicators: Dict = None) -> Optional[TradingSignal]:
        """Generate signal based on breakout patterns."""
        
        if not self.can_generate_signal(market_data_history):
            return None
        
        recent_data = market_data_history[-self.lookback_period:]
        current_data = market_data_history[-1]
        
        # Find support and resistance levels
        highs = [data.high for data in recent_data]
        lows = [data.low for data in recent_data]
        
        resistance_level = max(highs)
        support_level = min(lows)
        
        current_price = current_data.close
        current_volume = current_data.volume
        
        # Calculate average volume
        volumes = [data.volume for data in recent_data]
        avg_volume = statistics.mean(volumes)
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        signal_type = SignalType.HOLD
        confidence = 0.0
        reasoning = "No significant breakout detected"
        
        # Bullish breakout
        if (current_price > resistance_level * (1 + self.breakout_threshold) and
            volume_ratio > 1.5):  # High volume confirmation
            signal_type = SignalType.BUY
            breakout_strength = (current_price - resistance_level) / resistance_level
            volume_confirmation = min(volume_ratio / 2, 1.0)
            confidence = min(breakout_strength * 10 * volume_confirmation, 1.0)
            reasoning = f"Bullish breakout above resistance {resistance_level:.2f} with high volume"
        
        # Bearish breakout (breakdown)
        elif (current_price < support_level * (1 - self.breakout_threshold) and
              volume_ratio > 1.5):
            signal_type = SignalType.SELL
            breakdown_strength = (support_level - current_price) / support_level
            volume_confirmation = min(volume_ratio / 2, 1.0)
            confidence = min(breakdown_strength * 10 * volume_confirmation, 1.0)
            reasoning = f"Bearish breakdown below support {support_level:.2f} with high volume"
        
        if signal_type == SignalType.HOLD:
            return None
        
        # Calculate stop loss and take profit
        price_range = resistance_level - support_level
        
        if signal_type == SignalType.BUY:
            stop_loss = resistance_level - (price_range * 0.2)  # Just below old resistance
            take_profit = current_price + (price_range * 1.0)  # 1:1 risk-reward
        else:  # SELL
            stop_loss = support_level + (price_range * 0.2)  # Just above old support
            take_profit = current_price - (price_range * 1.0)
        
        # Risk score based on breakout strength and volume
        base_risk = 0.4  # Breakouts are inherently risky
        volume_risk = max(0, 1 - (volume_ratio / 3))  # Lower volume = higher risk
        risk_score = min(base_risk + volume_risk, 1.0)
        
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            strength=confidence,
            target_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=self.name,
            reasoning=reasoning,
            expiry_time=datetime.utcnow() + timedelta(hours=6),  # 6-hour expiry for breakouts
            risk_score=risk_score
        )


class MomentumStrategy(BaseStrategy):
    """Momentum-based strategy using multiple indicators."""
    
    def __init__(self, momentum_period: int = 10, strength_threshold: float = 0.03):
        super().__init__("Momentum")
        self.momentum_period = momentum_period
        self.strength_threshold = strength_threshold  # 3% momentum threshold
        self.min_data_points = momentum_period + 10
    
    def generate_signal(self, symbol: str, market_data_history: List[MarketData], 
                       indicators: Dict = None) -> Optional[TradingSignal]:
        """Generate signal based on price momentum and indicator convergence."""
        
        if not self.can_generate_signal(market_data_history):
            return None
        
        prices = [data.close for data in market_data_history]
        volumes = [data.volume for data in market_data_history]
        
        current_price = prices[-1]
        
        # Calculate price momentum
        price_momentum = self._calculate_price_momentum(prices, self.momentum_period)
        
        # Calculate volume momentum
        volume_momentum = self._calculate_volume_momentum(volumes, self.momentum_period)
        
        # Use indicators if available
        indicator_consensus = self._analyze_indicator_consensus(indicators) if indicators else 0
        
        # Combine signals
        total_momentum = (price_momentum * 0.4 + volume_momentum * 0.3 + indicator_consensus * 0.3)
        
        signal_type = SignalType.HOLD
        confidence = 0.0
        reasoning = "Insufficient momentum detected"
        
        if total_momentum > self.strength_threshold:
            signal_type = SignalType.BUY
            confidence = min(total_momentum / (self.strength_threshold * 3), 1.0)
            reasoning = f"Strong bullish momentum: price {price_momentum:.2%}, volume {volume_momentum:.2%}"
        
        elif total_momentum < -self.strength_threshold:
            signal_type = SignalType.SELL
            confidence = min(abs(total_momentum) / (self.strength_threshold * 3), 1.0)
            reasoning = f"Strong bearish momentum: price {price_momentum:.2%}, volume {volume_momentum:.2%}"
        
        if signal_type == SignalType.HOLD:
            return None
        
        # Calculate stop loss and take profit based on momentum
        momentum_magnitude = abs(total_momentum)
        price_volatility = self._calculate_price_volatility(prices[-20:])
        
        if signal_type == SignalType.BUY:
            stop_loss = current_price - (price_volatility * 2)
            take_profit = current_price + (momentum_magnitude * current_price * 2)
        else:  # SELL
            stop_loss = current_price + (price_volatility * 2)
            take_profit = current_price - (momentum_magnitude * current_price * 2)
        
        # Risk score based on momentum strength and volatility
        momentum_risk = max(0.3, momentum_magnitude * 5)  # Higher momentum = higher risk
        volatility_risk = min(price_volatility * 10, 0.5)  # Cap volatility risk
        risk_score = min(momentum_risk + volatility_risk, 1.0)
        
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            strength=confidence,
            target_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=self.name,
            reasoning=reasoning,
            expiry_time=datetime.utcnow() + timedelta(hours=3),  # 3-hour expiry for momentum
            risk_score=risk_score
        )
    
    def _calculate_price_momentum(self, prices: List[float], period: int) -> float:
        """Calculate price momentum over period."""
        if len(prices) < period:
            return 0.0
        
        current_price = prices[-1]
        past_price = prices[-period]
        
        return (current_price - past_price) / past_price if past_price > 0 else 0.0
    
    def _calculate_volume_momentum(self, volumes: List[float], period: int) -> float:
        """Calculate volume momentum."""
        if len(volumes) < period:
            return 0.0
        
        recent_avg = statistics.mean(volumes[-period//2:])
        older_avg = statistics.mean(volumes[-period:-period//2])
        
        if older_avg == 0:
            return 0.0
        
        return (recent_avg - older_avg) / older_avg
    
    def _calculate_price_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility (standard deviation)."""
        if len(prices) < 2:
            return 0.0
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] > 0]
        
        if not returns:
            return 0.0
        
        return statistics.stdev(returns) if len(returns) > 1 else 0.0
    
    def _analyze_indicator_consensus(self, indicators: Dict) -> float:
        """Analyze consensus from technical indicators."""
        if not indicators:
            return 0.0
        
        buy_signals = 0
        sell_signals = 0
        total_signals = 0
        
        for indicator_name, indicator_result in indicators.items():
            if hasattr(indicator_result, 'signal'):
                total_signals += 1
                if indicator_result.signal == 'BUY':
                    buy_signals += 1
                elif indicator_result.signal == 'SELL':
                    sell_signals += 1
        
        if total_signals == 0:
            return 0.0
        
        # Return consensus score (-1 to 1)
        consensus = (buy_signals - sell_signals) / total_signals
        return consensus