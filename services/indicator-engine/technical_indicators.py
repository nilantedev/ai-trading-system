#!/usr/bin/env python3
"""Technical Indicators - Comprehensive trading indicator calculations."""

import math
import statistics
from typing import List, Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import numpy as np

from trading_common import MarketData


@dataclass
class IndicatorResult:
    """Result of an indicator calculation."""
    name: str
    value: float
    signal: Optional[str] = None  # 'BUY', 'SELL', 'HOLD'
    strength: float = 0.0  # Signal strength 0-1
    parameters: Dict = None


class TechnicalIndicators:
    """Comprehensive technical indicators calculation engine."""
    
    def __init__(self):
        # Data storage for indicator calculations
        self.price_data: Dict[str, deque] = {}
        self.volume_data: Dict[str, deque] = {}
        self.high_data: Dict[str, deque] = {}
        self.low_data: Dict[str, deque] = {}
        
    def update_data(self, symbol: str, market_data: MarketData, max_periods: int = 200):
        """Update price data for a symbol."""
        if symbol not in self.price_data:
            self.price_data[symbol] = deque(maxlen=max_periods)
            self.volume_data[symbol] = deque(maxlen=max_periods)
            self.high_data[symbol] = deque(maxlen=max_periods)
            self.low_data[symbol] = deque(maxlen=max_periods)
        
        self.price_data[symbol].append(market_data.close)
        self.volume_data[symbol].append(market_data.volume)
        self.high_data[symbol].append(market_data.high)
        self.low_data[symbol].append(market_data.low)
    
    def calculate_all_indicators(self, symbol: str) -> Dict[str, IndicatorResult]:
        """Calculate all available indicators for a symbol."""
        if symbol not in self.price_data or len(self.price_data[symbol]) < 2:
            return {}
        
        indicators = {}
        
        # Moving Averages
        indicators.update(self._calculate_moving_averages(symbol))
        
        # Momentum Indicators
        indicators.update(self._calculate_momentum_indicators(symbol))
        
        # Trend Indicators
        indicators.update(self._calculate_trend_indicators(symbol))
        
        # Volume Indicators
        indicators.update(self._calculate_volume_indicators(symbol))
        
        # Volatility Indicators
        indicators.update(self._calculate_volatility_indicators(symbol))
        
        return indicators
    
    def _calculate_moving_averages(self, symbol: str) -> Dict[str, IndicatorResult]:
        """Calculate moving average indicators."""
        indicators = {}
        prices = list(self.price_data[symbol])
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            if len(prices) >= period:
                sma = statistics.mean(prices[-period:])
                current_price = prices[-1]
                
                # Generate signal
                signal = "HOLD"
                strength = 0.0
                
                if len(prices) >= period + 1:
                    prev_sma = statistics.mean(prices[-(period+1):-1])
                    if current_price > sma and prices[-2] <= prev_sma:
                        signal = "BUY"
                        strength = min((current_price - sma) / sma, 0.1) * 10
                    elif current_price < sma and prices[-2] >= prev_sma:
                        signal = "SELL"
                        strength = min((sma - current_price) / sma, 0.1) * 10
                
                indicators[f'sma_{period}'] = IndicatorResult(
                    name=f'SMA_{period}',
                    value=sma,
                    signal=signal,
                    strength=strength,
                    parameters={'period': period}
                )
        
        # Exponential Moving Averages
        for period in [12, 26, 50]:
            if len(prices) >= period:
                ema = self._calculate_ema(prices, period)
                current_price = prices[-1]
                
                signal = "BUY" if current_price > ema else "SELL" if current_price < ema else "HOLD"
                strength = abs(current_price - ema) / ema * 10 if ema > 0 else 0
                
                indicators[f'ema_{period}'] = IndicatorResult(
                    name=f'EMA_{period}',
                    value=ema,
                    signal=signal,
                    strength=min(strength, 1.0),
                    parameters={'period': period}
                )
        
        # Golden Cross / Death Cross
        if 'sma_50' in indicators and 'sma_200' in indicators:
            sma_50 = indicators['sma_50'].value
            sma_200 = indicators['sma_200'].value
            
            if sma_50 > sma_200:
                cross_signal = "BUY"
                cross_strength = min((sma_50 - sma_200) / sma_200, 0.05) * 20
            else:
                cross_signal = "SELL"
                cross_strength = min((sma_200 - sma_50) / sma_200, 0.05) * 20
            
            indicators['golden_cross'] = IndicatorResult(
                name='Golden_Cross',
                value=sma_50 / sma_200 if sma_200 > 0 else 1.0,
                signal=cross_signal,
                strength=cross_strength,
                parameters={'short_period': 50, 'long_period': 200}
            )
        
        return indicators
    
    def _calculate_momentum_indicators(self, symbol: str) -> Dict[str, IndicatorResult]:
        """Calculate momentum-based indicators."""
        indicators = {}
        prices = list(self.price_data[symbol])
        
        # RSI (Relative Strength Index)
        if len(prices) >= 15:  # Need at least 15 periods for reliable RSI
            rsi = self._calculate_rsi(prices, 14)
            
            signal = "HOLD"
            strength = 0.0
            
            if rsi <= 30:
                signal = "BUY"  # Oversold
                strength = (30 - rsi) / 30
            elif rsi >= 70:
                signal = "SELL"  # Overbought
                strength = (rsi - 70) / 30
            
            indicators['rsi'] = IndicatorResult(
                name='RSI',
                value=rsi,
                signal=signal,
                strength=strength,
                parameters={'period': 14}
            )
        
        # MACD (Moving Average Convergence Divergence)
        if len(prices) >= 26:
            macd_line, signal_line, histogram = self._calculate_macd(prices)
            
            signal = "HOLD"
            strength = 0.0
            
            if macd_line > signal_line and histogram > 0:
                signal = "BUY"
                strength = min(abs(histogram) / abs(macd_line), 1.0) if macd_line != 0 else 0
            elif macd_line < signal_line and histogram < 0:
                signal = "SELL"
                strength = min(abs(histogram) / abs(macd_line), 1.0) if macd_line != 0 else 0
            
            indicators['macd'] = IndicatorResult(
                name='MACD',
                value=macd_line,
                signal=signal,
                strength=strength,
                parameters={'fast': 12, 'slow': 26, 'signal': 9}
            )
            
            indicators['macd_signal'] = IndicatorResult(
                name='MACD_Signal',
                value=signal_line,
                parameters={'period': 9}
            )
            
            indicators['macd_histogram'] = IndicatorResult(
                name='MACD_Histogram',
                value=histogram,
                parameters={}
            )
        
        # Stochastic Oscillator
        if len(prices) >= 14:
            highs = list(self.high_data[symbol])
            lows = list(self.low_data[symbol])
            
            if len(highs) >= 14 and len(lows) >= 14:
                k_percent, d_percent = self._calculate_stochastic(prices, highs, lows, 14)
                
                signal = "HOLD"
                strength = 0.0
                
                if k_percent <= 20 and d_percent <= 20:
                    signal = "BUY"  # Oversold
                    strength = (20 - min(k_percent, d_percent)) / 20
                elif k_percent >= 80 and d_percent >= 80:
                    signal = "SELL"  # Overbought
                    strength = (min(k_percent, d_percent) - 80) / 20
                
                indicators['stoch_k'] = IndicatorResult(
                    name='Stochastic_K',
                    value=k_percent,
                    signal=signal,
                    strength=strength,
                    parameters={'period': 14}
                )
                
                indicators['stoch_d'] = IndicatorResult(
                    name='Stochastic_D',
                    value=d_percent,
                    parameters={'period': 3}
                )
        
        return indicators
    
    def _calculate_trend_indicators(self, symbol: str) -> Dict[str, IndicatorResult]:
        """Calculate trend-following indicators."""
        indicators = {}
        prices = list(self.price_data[symbol])
        highs = list(self.high_data[symbol])
        lows = list(self.low_data[symbol])
        
        # Bollinger Bands
        if len(prices) >= 20:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices, 20, 2)
            current_price = prices[-1]
            
            signal = "HOLD"
            strength = 0.0
            
            if current_price <= bb_lower:
                signal = "BUY"  # Price at lower band
                strength = (bb_lower - current_price) / (bb_middle - bb_lower) if bb_middle != bb_lower else 0
            elif current_price >= bb_upper:
                signal = "SELL"  # Price at upper band
                strength = (current_price - bb_upper) / (bb_upper - bb_middle) if bb_upper != bb_middle else 0
            
            indicators['bb_upper'] = IndicatorResult(
                name='Bollinger_Upper',
                value=bb_upper,
                signal=signal if signal == "SELL" else "HOLD",
                strength=strength if signal == "SELL" else 0,
                parameters={'period': 20, 'std_dev': 2}
            )
            
            indicators['bb_middle'] = IndicatorResult(
                name='Bollinger_Middle',
                value=bb_middle,
                parameters={'period': 20}
            )
            
            indicators['bb_lower'] = IndicatorResult(
                name='Bollinger_Lower',
                value=bb_lower,
                signal=signal if signal == "BUY" else "HOLD",
                strength=strength if signal == "BUY" else 0,
                parameters={'period': 20, 'std_dev': 2}
            )
        
        # ADX (Average Directional Index) - Simplified
        if len(prices) >= 14 and len(highs) >= 14 and len(lows) >= 14:
            adx = self._calculate_adx(highs, lows, prices, 14)
            
            signal = "HOLD"
            strength = 0.0
            
            if adx > 25:
                # Strong trend, determine direction
                if len(prices) >= 2 and prices[-1] > prices[-2]:
                    signal = "BUY"
                else:
                    signal = "SELL"
                strength = min((adx - 25) / 50, 1.0)  # Normalize to 0-1
            
            indicators['adx'] = IndicatorResult(
                name='ADX',
                value=adx,
                signal=signal,
                strength=strength,
                parameters={'period': 14}
            )
        
        return indicators
    
    def _calculate_volume_indicators(self, symbol: str) -> Dict[str, IndicatorResult]:
        """Calculate volume-based indicators."""
        indicators = {}
        prices = list(self.price_data[symbol])
        volumes = list(self.volume_data[symbol])
        
        if len(volumes) < 2:
            return indicators
        
        # On-Balance Volume (OBV)
        if len(prices) >= 2:
            obv = self._calculate_obv(prices, volumes)
            
            # OBV trend signal
            signal = "HOLD"
            strength = 0.0
            
            if len(prices) >= 10:
                recent_obv_trend = self._calculate_trend(obv[-10:])
                if recent_obv_trend > 0:
                    signal = "BUY"
                    strength = min(recent_obv_trend / 1000000, 1.0)  # Normalize
                elif recent_obv_trend < 0:
                    signal = "SELL"
                    strength = min(abs(recent_obv_trend) / 1000000, 1.0)
            
            indicators['obv'] = IndicatorResult(
                name='OBV',
                value=obv[-1] if obv else 0,
                signal=signal,
                strength=strength,
                parameters={}
            )
        
        # Volume Rate of Change
        if len(volumes) >= 10:
            current_vol = volumes[-1]
            avg_vol = statistics.mean(volumes[-10:])
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            signal = "HOLD"
            strength = 0.0
            
            if vol_ratio > 2.0:  # High volume
                signal = "BUY" if len(prices) >= 2 and prices[-1] > prices[-2] else "SELL"
                strength = min((vol_ratio - 1) / 3, 1.0)  # Normalize
            
            indicators['volume_ratio'] = IndicatorResult(
                name='Volume_Ratio',
                value=vol_ratio,
                signal=signal,
                strength=strength,
                parameters={'period': 10}
            )
        
        return indicators
    
    def _calculate_volatility_indicators(self, symbol: str) -> Dict[str, IndicatorResult]:
        """Calculate volatility indicators."""
        indicators = {}
        prices = list(self.price_data[symbol])
        highs = list(self.high_data[symbol])
        lows = list(self.low_data[symbol])
        
        # Average True Range (ATR)
        if len(prices) >= 14 and len(highs) >= 14 and len(lows) >= 14:
            atr = self._calculate_atr(highs, lows, prices, 14)
            current_price = prices[-1]
            
            # ATR as percentage of price
            atr_percent = (atr / current_price * 100) if current_price > 0 else 0
            
            signal = "HOLD"
            strength = 0.0
            
            # High ATR might indicate breakout opportunity
            if atr_percent > 3.0:  # High volatility
                signal = "HOLD"  # Wait for direction
                strength = min((atr_percent - 2) / 5, 1.0)
            
            indicators['atr'] = IndicatorResult(
                name='ATR',
                value=atr,
                signal=signal,
                strength=strength,
                parameters={'period': 14}
            )
            
            indicators['atr_percent'] = IndicatorResult(
                name='ATR_Percent',
                value=atr_percent,
                parameters={'period': 14}
            )
        
        return indicators
    
    # Helper calculation methods
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = statistics.mean(prices[:period])  # Start with SMA
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
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
            return 50.0
        
        avg_gain = statistics.mean(gains[-period:])
        avg_loss = statistics.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: List[float]) -> Tuple[float, float, float]:
        """Calculate MACD line, signal line, and histogram."""
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        macd_line = ema_12 - ema_26
        
        # Calculate signal line (9-period EMA of MACD line)
        # For simplicity, using approximation
        signal_line = macd_line * 0.1 + (macd_line * 0.9)  # Simplified
        
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_stochastic(self, prices: List[float], highs: List[float], 
                            lows: List[float], period: int) -> Tuple[float, float]:
        """Calculate Stochastic %K and %D."""
        if len(prices) < period:
            return 50.0, 50.0
        
        recent_highs = highs[-period:]
        recent_lows = lows[-period:]
        current_price = prices[-1]
        
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = ((current_price - lowest_low) / (highest_high - lowest_low)) * 100
        
        # %D is typically a 3-period moving average of %K
        # For simplicity, using current %K
        d_percent = k_percent
        
        return k_percent, d_percent
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int, 
                                 std_dev: float) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            current_price = prices[-1] if prices else 0
            return current_price, current_price, current_price
        
        recent_prices = prices[-period:]
        middle = statistics.mean(recent_prices)
        std = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def _calculate_adx(self, highs: List[float], lows: List[float], 
                      closes: List[float], period: int) -> float:
        """Calculate Average Directional Index (simplified)."""
        if len(highs) < period or len(lows) < period or len(closes) < period:
            return 0.0
        
        # Simplified ADX calculation
        true_ranges = []
        plus_dms = []
        minus_dms = []
        
        for i in range(1, min(len(highs), len(lows), len(closes))):
            # True Range
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)
            
            # Directional Movement
            plus_dm = max(highs[i] - highs[i-1], 0) if highs[i] - highs[i-1] > lows[i-1] - lows[i] else 0
            minus_dm = max(lows[i-1] - lows[i], 0) if lows[i-1] - lows[i] > highs[i] - highs[i-1] else 0
            
            plus_dms.append(plus_dm)
            minus_dms.append(minus_dm)
        
        if len(true_ranges) < period:
            return 0.0
        
        # Average True Range
        atr = statistics.mean(true_ranges[-period:])
        
        # Average Directional Index (simplified)
        avg_plus_dm = statistics.mean(plus_dms[-period:])
        avg_minus_dm = statistics.mean(minus_dms[-period:])
        
        if atr == 0:
            return 0.0
        
        plus_di = (avg_plus_dm / atr) * 100
        minus_di = (avg_minus_dm / atr) * 100
        
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if plus_di + minus_di > 0 else 0
        
        return dx  # Simplified ADX (would normally be smoothed)
    
    def _calculate_obv(self, prices: List[float], volumes: List[float]) -> List[float]:
        """Calculate On-Balance Volume."""
        if len(prices) < 2 or len(volumes) != len(prices):
            return [0]
        
        obv = [0]
        current_obv = 0
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                current_obv += volumes[i]
            elif prices[i] < prices[i-1]:
                current_obv -= volumes[i]
            # If prices[i] == prices[i-1], OBV stays the same
            
            obv.append(current_obv)
        
        return obv
    
    def _calculate_atr(self, highs: List[float], lows: List[float], 
                      closes: List[float], period: int) -> float:
        """Calculate Average True Range."""
        if len(highs) < period or len(lows) < period or len(closes) < period:
            return 0.0
        
        true_ranges = []
        
        for i in range(1, min(len(highs), len(lows), len(closes))):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)
        
        if len(true_ranges) < period:
            return 0.0
        
        return statistics.mean(true_ranges[-period:])
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate simple trend (slope) of values."""
        if len(values) < 2:
            return 0.0
        
        x = list(range(len(values)))
        y = values
        
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope