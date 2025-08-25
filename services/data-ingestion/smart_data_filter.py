#!/usr/bin/env python3
"""
Smart Data Filtering System - Only process high-value market data
Filters out noise and focuses on signals with profit potential.
"""

import numpy as np
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque, defaultdict
import statistics

from trading_common import MarketData, get_logger
from trading_common.cache import get_trading_cache

logger = get_logger(__name__)


@dataclass
class DataQualityScore:
    """Comprehensive data quality and value scoring."""
    overall_score: float  # 0-1, higher = more valuable
    volume_score: float   # Volume surge indicator
    volatility_score: float  # Price movement significance
    momentum_score: float    # Trend strength
    timing_score: float     # Market timing factors
    noise_ratio: float     # Signal to noise ratio
    should_process: bool   # Final decision
    reasoning: str         # Why this score was assigned


class SmartDataFilter:
    """Intelligent filtering system that only processes high-value market data."""
    
    def __init__(self, 
                 min_quality_score: float = 0.6,
                 volume_surge_threshold: float = 2.0,
                 volatility_threshold: float = 0.02):
        """
        Initialize smart data filter.
        
        Args:
            min_quality_score: Minimum score to process data (0.6 = only top 40% of data)
            volume_surge_threshold: Multiple of average volume to consider significant
            volatility_threshold: Minimum price change % to consider significant
        """
        self.min_quality_score = min_quality_score
        self.volume_surge_threshold = volume_surge_threshold  
        self.volatility_threshold = volatility_threshold
        
        # Historical data for comparison
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.volatility_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        
        # Performance tracking
        self.total_data_points = 0
        self.filtered_out = 0
        self.processed = 0
        self.cache = None
        
        # Market session awareness
        self.market_open_time = 9.5  # 9:30 AM
        self.market_close_time = 16.0  # 4:00 PM
        self.premarket_start = 4.0   # 4:00 AM
        self.afterhours_end = 20.0   # 8:00 PM
    
    async def initialize(self):
        """Initialize the data filter."""
        self.cache = get_trading_cache()
        logger.info("Smart Data Filter initialized")
    
    async def should_process_data(self, market_data: MarketData) -> DataQualityScore:
        """
        Determine if market data is worth processing based on quality and value.
        
        Returns:
            DataQualityScore with decision and reasoning
        """
        self.total_data_points += 1
        
        symbol = market_data.symbol
        current_price = market_data.close
        current_volume = market_data.volume
        timestamp = market_data.timestamp
        
        # Update historical data
        self.price_history[symbol].append(current_price)
        self.volume_history[symbol].append(current_volume)
        
        # Calculate individual scores
        volume_score = self._calculate_volume_score(symbol, current_volume)
        volatility_score = self._calculate_volatility_score(symbol, market_data)
        momentum_score = self._calculate_momentum_score(symbol)
        timing_score = self._calculate_timing_score(timestamp)
        noise_ratio = self._calculate_noise_ratio(symbol, market_data)
        
        # Calculate overall quality score
        # Weighted combination favoring high-impact signals
        overall_score = (
            0.30 * volume_score +      # 30% - Volume surge is critical
            0.25 * volatility_score +  # 25% - Price movement significance  
            0.20 * momentum_score +    # 20% - Trend strength
            0.15 * timing_score +      # 15% - Market timing
            0.10 * (1.0 - noise_ratio) # 10% - Lower noise is better
        )
        
        # Decision logic
        should_process = overall_score >= self.min_quality_score
        
        # Special cases for forced processing
        if not should_process:
            # Always process if extreme volume spike (5x+ average)
            if volume_score > 0.9:
                should_process = True
                overall_score = max(overall_score, 0.85)
            # Always process if major price movement (5%+)
            elif volatility_score > 0.9:
                should_process = True
                overall_score = max(overall_score, 0.80)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            overall_score, volume_score, volatility_score, 
            momentum_score, timing_score, noise_ratio
        )
        
        # Update statistics
        if should_process:
            self.processed += 1
        else:
            self.filtered_out += 1
        
        # Log filtering decisions for high-value data
        if overall_score > 0.8 or not should_process:
            logger.debug(f"Filter decision for {symbol}: {should_process} (score: {overall_score:.2f}) - {reasoning}")
        
        return DataQualityScore(
            overall_score=overall_score,
            volume_score=volume_score,
            volatility_score=volatility_score,
            momentum_score=momentum_score,
            timing_score=timing_score,
            noise_ratio=noise_ratio,
            should_process=should_process,
            reasoning=reasoning
        )
    
    def _calculate_volume_score(self, symbol: str, current_volume: int) -> float:
        """Score based on volume significance (0-1)."""
        if len(self.volume_history[symbol]) < 10:
            return 0.5  # Neutral until we have enough data
        
        volume_history = list(self.volume_history[symbol])
        avg_volume = statistics.mean(volume_history[-20:])  # 20-period average
        
        if avg_volume == 0:
            return 0.5
        
        volume_ratio = current_volume / avg_volume
        
        # Score based on volume surge magnitude
        if volume_ratio >= 5.0:      # 5x+ volume surge
            return 1.0
        elif volume_ratio >= 3.0:    # 3x volume surge  
            return 0.9
        elif volume_ratio >= 2.0:    # 2x volume surge
            return 0.7
        elif volume_ratio >= 1.5:    # 1.5x above average
            return 0.6
        elif volume_ratio >= 1.0:    # Average volume
            return 0.4
        else:                        # Below average volume
            return 0.2
    
    def _calculate_volatility_score(self, symbol: str, market_data: MarketData) -> float:
        """Score based on price movement significance (0-1)."""
        if len(self.price_history[symbol]) < 2:
            return 0.5
        
        current_price = market_data.close
        previous_price = list(self.price_history[symbol])[-2]
        
        # Calculate price change percentage
        price_change_pct = abs(current_price - previous_price) / previous_price
        
        # Calculate intraday range
        intraday_range_pct = (market_data.high - market_data.low) / market_data.open if market_data.open > 0 else 0
        
        # Calculate volatility score
        movement_score = min(price_change_pct / 0.05, 1.0)  # Normalize to 5% max
        range_score = min(intraday_range_pct / 0.08, 1.0)   # Normalize to 8% max
        
        # Weighted combination
        volatility_score = 0.6 * movement_score + 0.4 * range_score
        
        return min(volatility_score, 1.0)
    
    def _calculate_momentum_score(self, symbol: str) -> float:
        """Score based on momentum and trend strength (0-1)."""
        if len(self.price_history[symbol]) < 10:
            return 0.5
        
        prices = list(self.price_history[symbol])
        recent_prices = prices[-10:]  # Last 10 data points
        
        # Calculate trend strength using linear regression slope
        x = np.arange(len(recent_prices))
        y = np.array(recent_prices)
        
        try:
            # Calculate correlation coefficient as trend strength
            correlation = np.corrcoef(x, y)[0, 1]
            
            # Convert correlation to momentum score
            momentum_strength = abs(correlation) if not np.isnan(correlation) else 0
            
            # Bonus for consistent direction
            if len(recent_prices) >= 5:
                price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
                same_direction = sum(1 for change in price_changes if change > 0) / len(price_changes)
                
                if same_direction > 0.7 or same_direction < 0.3:  # Strong directional bias
                    momentum_strength *= 1.2
            
            return min(momentum_strength, 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_timing_score(self, timestamp: datetime) -> float:
        """Score based on market timing factors (0-1)."""
        hour = timestamp.hour + timestamp.minute / 60.0
        
        # Market session scoring
        if self.market_open_time <= hour <= self.market_close_time:
            # Regular trading hours
            if 9.5 <= hour <= 10.5 or 15.0 <= hour <= 16.0:
                # Opening hour or closing hour - high activity
                return 1.0
            elif 10.5 <= hour <= 15.0:
                # Mid-day - moderate activity
                return 0.7
            else:
                return 0.8
        elif self.premarket_start <= hour <= self.market_open_time:
            # Pre-market - significant for earnings/news
            return 0.8
        elif self.market_close_time <= hour <= self.afterhours_end:
            # After-hours - moderate significance
            return 0.6
        else:
            # Overnight - low significance
            return 0.3
    
    def _calculate_noise_ratio(self, symbol: str, market_data: MarketData) -> float:
        """Calculate signal-to-noise ratio (0-1, lower is better)."""
        # Check for data inconsistencies
        noise_factors = []
        
        # Price consistency check
        if market_data.high < market_data.close or market_data.low > market_data.close:
            noise_factors.append(0.5)  # Inconsistent OHLC
        
        # Volume consistency check  
        if market_data.volume < 0:
            noise_factors.append(0.8)  # Invalid volume
        
        # Timestamp freshness
        if market_data.timestamp:
            staleness_hours = (datetime.utcnow() - market_data.timestamp).total_seconds() / 3600
            if staleness_hours > 1.0:
                noise_factors.append(min(staleness_hours / 24, 0.7))  # Stale data
        
        # Price gap analysis
        if len(self.price_history[symbol]) >= 2:
            prev_price = list(self.price_history[symbol])[-2]
            price_gap = abs(market_data.open - prev_price) / prev_price
            if price_gap > 0.1:  # 10%+ gap might be erroneous
                noise_factors.append(min(price_gap, 0.6))
        
        # Return average noise ratio
        return statistics.mean(noise_factors) if noise_factors else 0.1
    
    def _generate_reasoning(self, overall_score: float, volume_score: float, 
                          volatility_score: float, momentum_score: float,
                          timing_score: float, noise_ratio: float) -> str:
        """Generate human-readable reasoning for the filtering decision."""
        reasons = []
        
        if volume_score > 0.8:
            reasons.append("high volume surge")
        elif volume_score < 0.3:
            reasons.append("low volume")
        
        if volatility_score > 0.7:
            reasons.append("significant price movement")
        elif volatility_score < 0.3:
            reasons.append("minimal price change")
        
        if momentum_score > 0.7:
            reasons.append("strong momentum")
        elif momentum_score < 0.3:
            reasons.append("weak momentum")
        
        if timing_score > 0.8:
            reasons.append("optimal market timing")
        elif timing_score < 0.4:
            reasons.append("off-hours timing")
        
        if noise_ratio > 0.5:
            reasons.append("high noise")
        
        if overall_score >= 0.8:
            return f"High-value signal: {', '.join(reasons)}"
        elif overall_score >= 0.6:
            return f"Moderate signal: {', '.join(reasons)}"
        else:
            return f"Low-value signal: {', '.join(reasons)}"
    
    async def get_filter_statistics(self) -> Dict[str, float]:
        """Get filtering performance statistics."""
        if self.total_data_points == 0:
            return {"status": "no_data"}
        
        filter_rate = self.filtered_out / self.total_data_points
        process_rate = self.processed / self.total_data_points
        
        stats = {
            "total_data_points": self.total_data_points,
            "filtered_out": self.filtered_out,
            "processed": self.processed,
            "filter_rate": filter_rate,
            "process_rate": process_rate,
            "noise_reduction": f"{filter_rate * 100:.1f}%",
            "efficiency_gain": f"{1/process_rate:.1f}x" if process_rate > 0 else "∞"
        }
        
        # Cache statistics
        if self.cache:
            await self.cache.set_json("smart_filter_stats", stats, ttl=60)
        
        return stats
    
    async def reset_statistics(self):
        """Reset filtering statistics."""
        self.total_data_points = 0
        self.filtered_out = 0
        self.processed = 0
        logger.info("Smart data filter statistics reset")


# Global filter instance
_smart_filter: Optional[SmartDataFilter] = None


async def get_smart_data_filter() -> SmartDataFilter:
    """Get or create global smart data filter instance."""
    global _smart_filter
    if _smart_filter is None:
        _smart_filter = SmartDataFilter(
            min_quality_score=0.6,  # Process top 40% of data
            volume_surge_threshold=2.0,
            volatility_threshold=0.02
        )
        await _smart_filter.initialize()
    return _smart_filter


async def filter_market_data(market_data: MarketData) -> Optional[MarketData]:
    """
    Filter market data through smart filtering system.
    
    Returns:
        MarketData if should be processed, None if should be filtered out
    """
    filter_system = await get_smart_data_filter()
    quality_score = await filter_system.should_process_data(market_data)
    
    if quality_score.should_process:
        # Add quality metadata to market data for downstream use
        market_data.metadata = {
            "quality_score": quality_score.overall_score,
            "volume_score": quality_score.volume_score, 
            "volatility_score": quality_score.volatility_score,
            "filter_reasoning": quality_score.reasoning
        }
        return market_data
    else:
        return None


async def get_filtering_performance() -> Dict[str, float]:
    """Get current filtering performance statistics."""
    filter_system = await get_smart_data_filter()
    return await filter_system.get_filter_statistics()