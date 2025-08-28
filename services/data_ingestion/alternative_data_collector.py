#!/usr/bin/env python3
"""
Alternative Data Collector - High-alpha data sources for trading edge
Collects options flow, insider trading, earnings whispers, and sentiment data.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import os

from trading_common import get_logger, get_settings
from trading_common.cache import get_trading_cache

logger = get_logger(__name__)
settings = get_settings()


class OptionsFlowType(Enum):
    """Types of options flow signals."""
    LARGE_CALL_SWEEP = "large_call_sweep"
    LARGE_PUT_SWEEP = "large_put_sweep"  
    UNUSUAL_CALL_VOLUME = "unusual_call_volume"
    UNUSUAL_PUT_VOLUME = "unusual_put_volume"
    DARK_POOL_CALL = "dark_pool_call"
    DARK_POOL_PUT = "dark_pool_put"


@dataclass
class OptionsFlow:
    """Options flow data point."""
    symbol: str
    flow_type: OptionsFlowType
    strike: float
    expiration: datetime
    option_type: str  # 'call' or 'put'
    volume: int
    open_interest: int
    premium: float
    implied_volatility: float
    delta: float
    timestamp: datetime
    confidence: float  # 0-1, how significant this flow is
    market_impact_score: float  # 0-1, expected market impact


@dataclass  
class InsiderActivity:
    """Insider trading activity."""
    symbol: str
    insider_name: str
    insider_title: str
    transaction_type: str  # 'buy', 'sell'
    shares: int
    price: float
    transaction_value: float
    filing_date: datetime
    transaction_date: datetime
    insider_ownership_change: float  # % change in ownership
    significance_score: float  # 0-1, how significant this transaction is


@dataclass
class EarningsWhisper:
    """Earnings whisper numbers and estimates."""
    symbol: str
    earnings_date: datetime
    official_estimate: float
    whisper_number: float
    whisper_confidence: float  # 0-1
    beat_probability: float   # 0-1, probability of beating estimate
    surprise_magnitude: float  # Expected surprise as %
    analyst_revision_trend: str  # 'up', 'down', 'stable'
    institutional_positioning: str  # 'bullish', 'bearish', 'neutral'


@dataclass
class SocialSentiment:
    """Social media sentiment aggregation."""
    symbol: str
    platform: str  # 'reddit', 'twitter', 'stocktwits'
    sentiment_score: float  # -1 to 1
    volume_score: float     # 0-1, volume of mentions
    momentum_score: float   # 0-1, rate of change in sentiment
    key_topics: List[str]   # Main discussion topics
    influential_mentions: int  # Mentions by high-follower accounts
    timestamp: datetime
    confidence: float       # 0-1, reliability of sentiment


@dataclass
class DarkPoolActivity:
    """Dark pool trading activity."""
    symbol: str
    volume: int
    average_price: float
    transaction_count: int
    institutional_percentage: float  # % likely institutional
    accumulation_score: float  # -1 to 1, accumulation vs distribution
    timestamp: datetime
    significance: str  # 'low', 'medium', 'high'


class AlternativeDataCollector:
    """Collector for high-value alternative data sources."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache = None
        
        # API configurations (add your API keys to environment)
        self.unusual_whales_key = os.getenv('UNUSUAL_WHALES_API_KEY')
        self.quiver_quant_key = os.getenv('QUIVER_QUANT_API_KEY')
        self.sentiment_api_key = os.getenv('SENTIMENT_API_KEY')
        self.insider_trading_key = os.getenv('INSIDER_TRADING_API_KEY')
        
        # Performance tracking
        self.data_points_collected = 0
        self.high_value_signals = 0
        
    async def initialize(self):
        """Initialize alternative data collector."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'AI-Trading-System/1.0'}
        )
        self.cache = get_trading_cache()
        logger.info("Alternative Data Collector initialized")
    
    async def close(self):
        """Close the alternative data collector."""
        if self.session:
            await self.session.close()
    
    async def get_options_flow(self, symbol: str, lookback_hours: int = 4) -> List[OptionsFlow]:
        """
        Get unusual options activity for a symbol.
        
        This is where institutional money moves first - often predicting stock movements.
        """
        options_flow = []
        
        if not self.unusual_whales_key:
            # Mock data for development
            return self._mock_options_flow(symbol)
        
        try:
            # Unusual Whales API or similar options flow service
            url = f"https://api.unusualwhales.com/api/stock/{symbol}/options-flow"
            headers = {"Authorization": f"Bearer {self.unusual_whales_key}"}
            params = {
                "date": datetime.utcnow().strftime("%Y-%m-%d"),
                "min_premium": 50000,  # Minimum $50k premium for significance
                "min_volume": 100      # Minimum 100 contracts
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for flow in data.get('flows', []):
                        # Calculate significance scores
                        confidence = self._calculate_options_confidence(flow)
                        market_impact = self._calculate_options_impact(flow)
                        
                        if confidence > 0.6:  # Only high-confidence flows
                            flow_type = self._classify_options_flow(flow)
                            
                            options_flow.append(OptionsFlow(
                                symbol=symbol,
                                flow_type=flow_type,
                                strike=flow['strike'],
                                expiration=datetime.fromisoformat(flow['expiration']),
                                option_type=flow['type'],
                                volume=flow['volume'],
                                open_interest=flow['open_interest'],
                                premium=flow['premium'],
                                implied_volatility=flow['iv'],
                                delta=flow['delta'],
                                timestamp=datetime.fromisoformat(flow['timestamp']),
                                confidence=confidence,
                                market_impact_score=market_impact
                            ))
                
        except Exception as e:
            logger.warning(f"Failed to get options flow for {symbol}: {e}")
            return self._mock_options_flow(symbol)
        
        # Cache the results
        if options_flow and self.cache:
            await self._cache_options_flow(symbol, options_flow)
        
        self.data_points_collected += len(options_flow)
        self.high_value_signals += len([f for f in options_flow if f.confidence > 0.8])
        
        return options_flow
    
    async def get_insider_trading(self, symbol: str, lookback_days: int = 30) -> List[InsiderActivity]:
        """
        Get recent insider trading activity.
        
        Insiders know their company best - follow the smart money.
        """
        insider_activities = []
        
        if not self.insider_trading_key:
            return self._mock_insider_activity(symbol)
        
        try:
            # SEC EDGAR or Quiver Quant for insider trading
            url = f"https://api.quiverquant.com/beta/live/insidertrading/{symbol}"
            headers = {"Authorization": f"Bearer {self.insider_trading_key}"}
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for activity in data:
                        significance = self._calculate_insider_significance(activity)
                        
                        if significance > 0.5:  # Only significant trades
                            insider_activities.append(InsiderActivity(
                                symbol=symbol,
                                insider_name=activity['name'],
                                insider_title=activity['title'],
                                transaction_type=activity['transaction_type'],
                                shares=activity['shares'],
                                price=activity['price'],
                                transaction_value=activity['value'],
                                filing_date=datetime.fromisoformat(activity['filing_date']),
                                transaction_date=datetime.fromisoformat(activity['transaction_date']),
                                insider_ownership_change=activity['ownership_change'],
                                significance_score=significance
                            ))
                
        except Exception as e:
            logger.warning(f"Failed to get insider trading for {symbol}: {e}")
            return self._mock_insider_activity(symbol)
        
        return insider_activities
    
    async def get_earnings_whispers(self, symbol: str) -> Optional[EarningsWhisper]:
        """
        Get earnings whisper numbers and beat probability.
        
        Wall Street whispers often more accurate than official estimates.
        """
        if not self.sentiment_api_key:
            return self._mock_earnings_whisper(symbol)
        
        try:
            # Earnings whispers API
            url = f"https://api.earningswhispers.com/api/stock/{symbol}/estimate"
            headers = {"X-API-Key": self.sentiment_api_key}
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return EarningsWhisper(
                        symbol=symbol,
                        earnings_date=datetime.fromisoformat(data['earnings_date']),
                        official_estimate=data['official_estimate'],
                        whisper_number=data['whisper_estimate'],
                        whisper_confidence=data['whisper_confidence'],
                        beat_probability=data['beat_probability'],
                        surprise_magnitude=data['expected_surprise'],
                        analyst_revision_trend=data['revision_trend'],
                        institutional_positioning=data['institutional_sentiment']
                    )
                
        except Exception as e:
            logger.warning(f"Failed to get earnings whispers for {symbol}: {e}")
            return self._mock_earnings_whisper(symbol)
    
    async def get_social_sentiment(self, symbol: str) -> List[SocialSentiment]:
        """
        Get aggregated social media sentiment.
        
        Reddit, Twitter, StockTwits sentiment can predict short-term moves.
        """
        sentiment_data = []
        
        # Get sentiment from multiple platforms
        platforms = ['reddit', 'twitter', 'stocktwits']
        
        for platform in platforms:
            sentiment = await self._get_platform_sentiment(symbol, platform)
            if sentiment:
                sentiment_data.append(sentiment)
        
        return sentiment_data
    
    async def get_dark_pool_activity(self, symbol: str) -> Optional[DarkPoolActivity]:
        """
        Get dark pool trading activity.
        
        Large institutional orders often routed through dark pools.
        """
        if not self.quiver_quant_key:
            return self._mock_dark_pool_activity(symbol)
        
        try:
            url = f"https://api.quiverquant.com/beta/live/darkpools/{symbol}"
            headers = {"Authorization": f"Bearer {self.quiver_quant_key}"}
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return DarkPoolActivity(
                        symbol=symbol,
                        volume=data['volume'],
                        average_price=data['average_price'],
                        transaction_count=data['transaction_count'],
                        institutional_percentage=data['institutional_percentage'],
                        accumulation_score=data['accumulation_score'],
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        significance=data['significance']
                    )
                
        except Exception as e:
            logger.warning(f"Failed to get dark pool activity for {symbol}: {e}")
            return self._mock_dark_pool_activity(symbol)
    
    async def get_comprehensive_alternative_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get all alternative data sources for a symbol in one call.
        
        Returns comprehensive alternative data for trading decisions.
        """
        # Run all data collection concurrently
        tasks = [
            self.get_options_flow(symbol),
            self.get_insider_trading(symbol),
            self.get_earnings_whispers(symbol),
            self.get_social_sentiment(symbol),
            self.get_dark_pool_activity(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "symbol": symbol,
            "timestamp": datetime.utcnow(),
            "options_flow": results[0] if not isinstance(results[0], Exception) else [],
            "insider_trading": results[1] if not isinstance(results[1], Exception) else [],
            "earnings_whispers": results[2] if not isinstance(results[2], Exception) else None,
            "social_sentiment": results[3] if not isinstance(results[3], Exception) else [],
            "dark_pool_activity": results[4] if not isinstance(results[4], Exception) else None,
            "overall_sentiment_score": self._calculate_overall_sentiment(results),
            "high_conviction_signals": self._identify_high_conviction_signals(results)
        }
    
    # Private helper methods for calculations and mock data
    def _calculate_options_confidence(self, flow: Dict) -> float:
        """Calculate confidence score for options flow."""
        # Volume vs open interest ratio
        volume_ratio = flow['volume'] / max(flow['open_interest'], 1)
        
        # Premium size (larger = more significant)
        premium_score = min(flow['premium'] / 100000, 1.0)  # Normalize to $100k
        
        # Time to expiration (closer = more significant)
        days_to_exp = (datetime.fromisoformat(flow['expiration']) - datetime.utcnow()).days
        time_score = max(0.2, 1.0 - (days_to_exp / 30))  # Decay over 30 days
        
        return (volume_ratio * 0.4 + premium_score * 0.4 + time_score * 0.2)
    
    def _calculate_options_impact(self, flow: Dict) -> float:
        """Calculate market impact score for options flow."""
        # Delta-adjusted exposure
        delta_exposure = abs(flow['delta']) * flow['volume'] * 100  # 100 shares per contract
        
        # Normalize to stock price
        stock_price = flow.get('underlying_price', 100)  # Default if not provided
        impact_score = min(delta_exposure / (stock_price * 10000), 1.0)
        
        return impact_score
    
    def _classify_options_flow(self, flow: Dict) -> OptionsFlowType:
        """Classify the type of options flow."""
        if flow['premium'] > 100000 and flow['volume'] > flow['open_interest']:
            if flow['type'] == 'call':
                return OptionsFlowType.LARGE_CALL_SWEEP
            else:
                return OptionsFlowType.LARGE_PUT_SWEEP
        elif flow['volume'] > flow['open_interest'] * 3:
            if flow['type'] == 'call':
                return OptionsFlowType.UNUSUAL_CALL_VOLUME
            else:
                return OptionsFlowType.UNUSUAL_PUT_VOLUME
        else:
            if flow['type'] == 'call':
                return OptionsFlowType.DARK_POOL_CALL
            else:
                return OptionsFlowType.DARK_POOL_PUT
    
    def _calculate_insider_significance(self, activity: Dict) -> float:
        """Calculate significance of insider trading activity."""
        # Transaction size relative to typical insider trades
        value_score = min(activity['value'] / 1000000, 1.0)  # Normalize to $1M
        
        # Ownership change percentage
        ownership_score = min(abs(activity['ownership_change']) / 10, 1.0)  # Normalize to 10%
        
        # Insider title importance
        title_scores = {
            'CEO': 1.0, 'CFO': 0.9, 'President': 0.8, 'COO': 0.7,
            'Director': 0.6, 'Officer': 0.5, 'Other': 0.3
        }
        title_score = title_scores.get(activity['title'], 0.3)
        
        return (value_score * 0.4 + ownership_score * 0.3 + title_score * 0.3)
    
    def _calculate_overall_sentiment(self, results: List) -> float:
        """Calculate overall sentiment from all alternative data sources."""
        sentiment_factors = []
        
        # Options flow sentiment
        if results[0] and not isinstance(results[0], Exception):
            call_flows = [f for f in results[0] if 'call' in f.flow_type.value]
            put_flows = [f for f in results[0] if 'put' in f.flow_type.value]
            
            if call_flows or put_flows:
                options_sentiment = (len(call_flows) - len(put_flows)) / max(len(call_flows) + len(put_flows), 1)
                sentiment_factors.append(options_sentiment)
        
        # Insider trading sentiment
        if results[1] and not isinstance(results[1], Exception):
            buys = [a for a in results[1] if a.transaction_type == 'buy']
            sells = [a for a in results[1] if a.transaction_type == 'sell']
            
            if buys or sells:
                insider_sentiment = (len(buys) - len(sells)) / max(len(buys) + len(sells), 1)
                sentiment_factors.append(insider_sentiment)
        
        # Social sentiment
        if results[3] and not isinstance(results[3], Exception):
            social_scores = [s.sentiment_score for s in results[3]]
            if social_scores:
                sentiment_factors.append(sum(social_scores) / len(social_scores))
        
        # Dark pool sentiment
        if results[4] and not isinstance(results[4], Exception):
            sentiment_factors.append(results[4].accumulation_score)
        
        return sum(sentiment_factors) / len(sentiment_factors) if sentiment_factors else 0.0
    
    def _identify_high_conviction_signals(self, results: List) -> List[str]:
        """Identify high-conviction signals from alternative data."""
        signals = []
        
        # High-value options flows
        if results[0] and not isinstance(results[0], Exception):
            high_conviction_flows = [f for f in results[0] if f.confidence > 0.8 and f.market_impact_score > 0.7]
            for flow in high_conviction_flows:
                signals.append(f"High-conviction {flow.flow_type.value}: ${flow.premium:,.0f} premium")
        
        # Significant insider activity
        if results[1] and not isinstance(results[1], Exception):
            significant_insider = [a for a in results[1] if a.significance_score > 0.8]
            for activity in significant_insider:
                signals.append(f"Insider {activity.transaction_type}: {activity.insider_title} ${activity.transaction_value:,.0f}")
        
        # Earnings beat probability
        if results[2] and not isinstance(results[2], Exception):
            whisper = results[2]
            if whisper.beat_probability > 0.75:
                signals.append(f"High earnings beat probability: {whisper.beat_probability:.0%}")
        
        return signals
    
    # Mock data methods for development/testing
    def _mock_options_flow(self, symbol: str) -> List[OptionsFlow]:
        """Generate mock options flow data for development."""
        return [
            OptionsFlow(
                symbol=symbol,
                flow_type=OptionsFlowType.LARGE_CALL_SWEEP,
                strike=150.0,
                expiration=datetime.utcnow() + timedelta(days=14),
                option_type='call',
                volume=500,
                open_interest=200,
                premium=75000.0,
                implied_volatility=0.35,
                delta=0.65,
                timestamp=datetime.utcnow(),
                confidence=0.85,
                market_impact_score=0.78
            )
        ]
    
    def _mock_insider_activity(self, symbol: str) -> List[InsiderActivity]:
        """Generate mock insider activity for development."""
        return [
            InsiderActivity(
                symbol=symbol,
                insider_name="John CEO",
                insider_title="CEO",
                transaction_type="buy",
                shares=10000,
                price=145.50,
                transaction_value=1455000.0,
                filing_date=datetime.utcnow() - timedelta(days=2),
                transaction_date=datetime.utcnow() - timedelta(days=5),
                insider_ownership_change=2.5,
                significance_score=0.92
            )
        ]
    
    def _mock_earnings_whisper(self, symbol: str) -> EarningsWhisper:
        """Generate mock earnings whisper for development."""
        return EarningsWhisper(
            symbol=symbol,
            earnings_date=datetime.utcnow() + timedelta(days=7),
            official_estimate=2.45,
            whisper_number=2.52,
            whisper_confidence=0.78,
            beat_probability=0.68,
            surprise_magnitude=2.8,
            analyst_revision_trend="up",
            institutional_positioning="bullish"
        )
    
    def _mock_dark_pool_activity(self, symbol: str) -> DarkPoolActivity:
        """Generate mock dark pool activity."""
        return DarkPoolActivity(
            symbol=symbol,
            volume=125000,
            average_price=148.75,
            transaction_count=15,
            institutional_percentage=0.85,
            accumulation_score=0.72,
            timestamp=datetime.utcnow() - timedelta(hours=1),
            significance="high"
        )
    
    async def _get_platform_sentiment(self, symbol: str, platform: str) -> Optional[SocialSentiment]:
        """Get sentiment from a specific platform."""
        # Mock implementation
        sentiment_scores = {'reddit': 0.65, 'twitter': 0.45, 'stocktwits': 0.72}
        
        return SocialSentiment(
            symbol=symbol,
            platform=platform,
            sentiment_score=sentiment_scores.get(platform, 0.5),
            volume_score=0.68,
            momentum_score=0.55,
            key_topics=[f"{symbol} bullish", "earnings beat", "technical breakout"],
            influential_mentions=12,
            timestamp=datetime.utcnow(),
            confidence=0.74
        )
    
    async def _cache_options_flow(self, symbol: str, flows: List[OptionsFlow]):
        """Cache options flow data."""
        if self.cache:
            cache_key = f"options_flow:{symbol}:latest"
            cache_data = {
                "symbol": symbol,
                "flows": [
                    {
                        "flow_type": flow.flow_type.value,
                        "strike": flow.strike,
                        "premium": flow.premium,
                        "confidence": flow.confidence,
                        "market_impact_score": flow.market_impact_score,
                        "timestamp": flow.timestamp.isoformat()
                    }
                    for flow in flows
                ],
                "cached_at": datetime.utcnow().isoformat()
            }
            await self.cache.set_json(cache_key, cache_data, ttl=3600)  # 1 hour
    
    async def get_collector_statistics(self) -> Dict[str, Any]:
        """Get alternative data collector statistics."""
        return {
            "data_points_collected": self.data_points_collected,
            "high_value_signals": self.high_value_signals,
            "api_keys_configured": {
                "unusual_whales": bool(self.unusual_whales_key),
                "quiver_quant": bool(self.quiver_quant_key),
                "sentiment_api": bool(self.sentiment_api_key),
                "insider_trading": bool(self.insider_trading_key)
            },
            "high_value_signal_ratio": self.high_value_signals / max(self.data_points_collected, 1)
        }


# Global collector instance
_alternative_data_collector: Optional[AlternativeDataCollector] = None


async def get_alternative_data_collector() -> AlternativeDataCollector:
    """Get or create global alternative data collector."""
    global _alternative_data_collector
    if _alternative_data_collector is None:
        _alternative_data_collector = AlternativeDataCollector()
        await _alternative_data_collector.initialize()
    return _alternative_data_collector


async def get_alternative_signals(symbol: str) -> Dict[str, Any]:
    """
    Get comprehensive alternative data signals for a symbol.
    
    This is the main entry point for alternative data collection.
    """
    collector = await get_alternative_data_collector()
    return await collector.get_comprehensive_alternative_data(symbol)