#!/usr/bin/env python3
"""
Market Data Validator - Validates all incoming market data for quality and sanity
Critical component to prevent bad data from corrupting trading decisions
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Data quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    quality: DataQuality
    reason: str
    anomalies: List[str]
    metrics: Dict[str, float]


class MarketDataValidator:
    """
    Validates all incoming market data to ensure quality and prevent
    bad data from affecting trading decisions.
    """
    
    def __init__(self):
        # Sanity check thresholds
        self.SANITY_CHECKS = {
            "min_price": 0.01,           # Minimum valid price
            "max_price": 100000,         # Maximum valid price  
            "max_price_change_pct": 0.50, # 50% max change in one tick
            "max_spread_pct": 0.10,      # 10% max bid-ask spread
            "stale_data_seconds": 60,    # Data older than 60s is stale
            "min_volume": 0,              # Minimum valid volume
            "max_volume": 1e12,           # Maximum valid volume (1 trillion)
        }
        
        # Historical data for anomaly detection
        self.price_history = {}  # symbol -> list of recent prices
        self.volume_history = {}  # symbol -> list of recent volumes
        self.history_size = 100   # Keep last 100 data points
        
    def validate_tick(self, tick: Dict[str, Any]) -> ValidationResult:
        """
        Validate a single market data tick
        """
        anomalies = []
        metrics = {}
        
        # Required fields check
        required_fields = ['symbol', 'price', 'timestamp']
        for field in required_fields:
            if field not in tick or tick[field] is None:
                return ValidationResult(
                    is_valid=False,
                    quality=DataQuality.INVALID,
                    reason=f"Missing required field: {field}",
                    anomalies=[f"missing_{field}"],
                    metrics={}
                )
        
        symbol = tick['symbol']
        price = tick['price']
        timestamp = tick['timestamp']
        
        # Convert timestamp if needed
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # Price sanity checks
        if price <= self.SANITY_CHECKS['min_price']:
            anomalies.append("price_too_low")
            return ValidationResult(
                is_valid=False,
                quality=DataQuality.INVALID,
                reason=f"Price too low: ${price}",
                anomalies=anomalies,
                metrics={"price": price}
            )
        
        if price > self.SANITY_CHECKS['max_price']:
            anomalies.append("price_too_high")
            return ValidationResult(
                is_valid=False,
                quality=DataQuality.INVALID,
                reason=f"Price too high: ${price}",
                anomalies=anomalies,
                metrics={"price": price}
            )
        
        metrics["price"] = price
        
        # Timestamp freshness check
        age = datetime.now() - timestamp
        age_seconds = age.total_seconds()
        metrics["data_age_seconds"] = age_seconds
        
        if age_seconds > self.SANITY_CHECKS['stale_data_seconds']:
            anomalies.append("stale_data")
            if age_seconds > 300:  # More than 5 minutes old
                return ValidationResult(
                    is_valid=False,
                    quality=DataQuality.INVALID,
                    reason=f"Data too stale: {age_seconds:.1f}s old",
                    anomalies=anomalies,
                    metrics=metrics
                )
        
        # Bid-Ask spread validation
        if 'bid' in tick and 'ask' in tick:
            bid = tick['bid']
            ask = tick['ask']
            
            # Basic sanity
            if bid <= 0 or ask <= 0:
                anomalies.append("invalid_bid_ask")
                return ValidationResult(
                    is_valid=False,
                    quality=DataQuality.INVALID,
                    reason="Invalid bid or ask price",
                    anomalies=anomalies,
                    metrics=metrics
                )
            
            # Spread check
            if bid > ask:
                anomalies.append("inverted_spread")
                return ValidationResult(
                    is_valid=False,
                    quality=DataQuality.INVALID,
                    reason=f"Inverted spread: bid ${bid} > ask ${ask}",
                    anomalies=anomalies,
                    metrics=metrics
                )
            
            spread = ask - bid
            spread_pct = spread / bid if bid > 0 else 0
            metrics["spread_pct"] = spread_pct * 100
            
            if spread_pct > self.SANITY_CHECKS['max_spread_pct']:
                anomalies.append("wide_spread")
                # Wide spread is concerning but not always invalid
                if spread_pct > 0.20:  # 20% spread is definitely invalid
                    return ValidationResult(
                        is_valid=False,
                        quality=DataQuality.INVALID,
                        reason=f"Spread too wide: {spread_pct*100:.2f}%",
                        anomalies=anomalies,
                        metrics=metrics
                    )
        
        # Volume validation
        if 'volume' in tick:
            volume = tick['volume']
            
            if volume < self.SANITY_CHECKS['min_volume']:
                anomalies.append("negative_volume")
                return ValidationResult(
                    is_valid=False,
                    quality=DataQuality.INVALID,
                    reason=f"Invalid volume: {volume}",
                    anomalies=anomalies,
                    metrics=metrics
                )
            
            if volume > self.SANITY_CHECKS['max_volume']:
                anomalies.append("excessive_volume")
                return ValidationResult(
                    is_valid=False,
                    quality=DataQuality.INVALID,
                    reason=f"Excessive volume: {volume}",
                    anomalies=anomalies,
                    metrics=metrics
                )
            
            metrics["volume"] = volume
            
            # Check for volume spike
            if symbol in self.volume_history and len(self.volume_history[symbol]) > 10:
                avg_volume = statistics.mean(self.volume_history[symbol][-20:])
                if avg_volume > 0 and volume > avg_volume * 10:
                    anomalies.append("volume_spike")
        
        # Price change validation (if we have history)
        if symbol in self.price_history and len(self.price_history[symbol]) > 0:
            last_price = self.price_history[symbol][-1]
            price_change = abs(price - last_price) / last_price if last_price > 0 else 0
            metrics["price_change_pct"] = price_change * 100
            
            if price_change > self.SANITY_CHECKS['max_price_change_pct']:
                anomalies.append("large_price_jump")
                # Large jumps need investigation
                if price_change > 0.75:  # 75% change is definitely invalid
                    return ValidationResult(
                        is_valid=False,
                        quality=DataQuality.INVALID,
                        reason=f"Price jump too large: {price_change*100:.2f}%",
                        anomalies=anomalies,
                        metrics=metrics
                    )
        
        # Update history
        self._update_history(symbol, price, tick.get('volume', 0))
        
        # Determine overall quality
        quality = self._assess_quality(anomalies, metrics)
        
        return ValidationResult(
            is_valid=True,
            quality=quality,
            reason="Validation passed" if not anomalies else f"Minor issues: {', '.join(anomalies)}",
            anomalies=anomalies,
            metrics=metrics
        )
    
    def validate_candle(self, candle: Dict[str, Any]) -> ValidationResult:
        """
        Validate OHLCV candle data
        """
        anomalies = []
        metrics = {}
        
        # Required fields
        required = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        for field in required:
            if field not in candle or candle[field] is None:
                return ValidationResult(
                    is_valid=False,
                    quality=DataQuality.INVALID,
                    reason=f"Missing required field: {field}",
                    anomalies=[f"missing_{field}"],
                    metrics={}
                )
        
        o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
        volume = candle['volume']
        
        # OHLC relationship checks
        if not (l <= o <= h and l <= c <= h):
            anomalies.append("invalid_ohlc_relationship")
            return ValidationResult(
                is_valid=False,
                quality=DataQuality.INVALID,
                reason=f"Invalid OHLC: O={o}, H={h}, L={l}, C={c}",
                anomalies=anomalies,
                metrics={}
            )
        
        # Price sanity
        for price_type, price in [('open', o), ('high', h), ('low', l), ('close', c)]:
            if price <= 0 or price > self.SANITY_CHECKS['max_price']:
                return ValidationResult(
                    is_valid=False,
                    quality=DataQuality.INVALID,
                    reason=f"Invalid {price_type} price: {price}",
                    anomalies=[f"invalid_{price_type}"],
                    metrics={}
                )
        
        # Calculate metrics
        metrics["range_pct"] = ((h - l) / l * 100) if l > 0 else 0
        metrics["change_pct"] = ((c - o) / o * 100) if o > 0 else 0
        metrics["volume"] = volume
        
        # Check for anomalies
        if metrics["range_pct"] > 50:  # 50% range in one candle
            anomalies.append("excessive_range")
        
        if metrics["range_pct"] < 0.001:  # Essentially no movement
            anomalies.append("flatlined")
        
        # Determine quality
        quality = self._assess_quality(anomalies, metrics)
        
        return ValidationResult(
            is_valid=True,
            quality=quality,
            reason="Candle validated",
            anomalies=anomalies,
            metrics=metrics
        )
    
    def detect_anomalies(self, prices: List[float]) -> Dict[str, Any]:
        """
        Detect various anomalies in price series
        """
        if len(prices) < 2:
            return {"insufficient_data": True}
        
        anomalies = {
            "has_gaps": self._detect_gaps(prices),
            "has_spikes": self._detect_spikes(prices),
            "is_flatlined": self._detect_flatline(prices),
            "has_negatives": any(p <= 0 for p in prices),
            "has_outliers": self._detect_outliers(prices)
        }
        
        return anomalies
    
    def _detect_gaps(self, prices: List[float]) -> bool:
        """Detect price gaps (>5% jump between consecutive prices)"""
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                change = abs(prices[i] - prices[i-1]) / prices[i-1]
                if change > 0.05:  # 5% gap
                    return True
        return False
    
    def _detect_spikes(self, prices: List[float]) -> bool:
        """Detect price spikes using z-score"""
        if len(prices) < 10:
            return False
        
        mean = statistics.mean(prices)
        stdev = statistics.stdev(prices)
        
        if stdev == 0:
            return False
        
        for price in prices:
            z_score = abs((price - mean) / stdev)
            if z_score > 3:  # 3 standard deviations
                return True
        
        return False
    
    def _detect_flatline(self, prices: List[float]) -> bool:
        """Detect if prices are flatlined (no movement)"""
        if len(prices) < 5:
            return False
        
        # Check if all prices are the same
        return len(set(prices)) == 1
    
    def _detect_outliers(self, prices: List[float]) -> bool:
        """Detect outliers using IQR method"""
        if len(prices) < 10:
            return False
        
        sorted_prices = sorted(prices)
        q1_idx = len(sorted_prices) // 4
        q3_idx = 3 * len(sorted_prices) // 4
        
        q1 = sorted_prices[q1_idx]
        q3 = sorted_prices[q3_idx]
        iqr = q3 - q1
        
        if iqr == 0:
            return False
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        for price in prices:
            if price < lower_bound or price > upper_bound:
                return True
        
        return False
    
    def _update_history(self, symbol: str, price: float, volume: float):
        """Update price and volume history for a symbol"""
        # Initialize if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.volume_history[symbol] = []
        
        # Add new data
        self.price_history[symbol].append(price)
        self.volume_history[symbol].append(volume)
        
        # Trim to max size
        if len(self.price_history[symbol]) > self.history_size:
            self.price_history[symbol] = self.price_history[symbol][-self.history_size:]
            self.volume_history[symbol] = self.volume_history[symbol][-self.history_size:]
    
    def _assess_quality(self, anomalies: List[str], metrics: Dict[str, float]) -> DataQuality:
        """Assess overall data quality based on anomalies and metrics"""
        if not anomalies:
            return DataQuality.EXCELLENT
        
        # Critical anomalies
        critical = ['invalid_bid_ask', 'inverted_spread', 'negative_volume', 'invalid_ohlc_relationship']
        if any(a in critical for a in anomalies):
            return DataQuality.POOR
        
        # Count anomalies
        if len(anomalies) >= 3:
            return DataQuality.POOR
        elif len(anomalies) == 2:
            return DataQuality.ACCEPTABLE
        elif len(anomalies) == 1:
            # Check severity
            if 'stale_data' in anomalies or 'large_price_jump' in anomalies:
                return DataQuality.ACCEPTABLE
            return DataQuality.GOOD
        
        return DataQuality.GOOD
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            "symbols_tracked": len(self.price_history),
            "sanity_checks": self.SANITY_CHECKS,
            "history_size": self.history_size
        }