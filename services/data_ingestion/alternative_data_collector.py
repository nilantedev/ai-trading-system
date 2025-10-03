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
from contextlib import asynccontextmanager

# Optional Prometheus metrics (best-effort; continue if not available)
try:  # noqa: SIM105
    from prometheus_client import Counter  # type: ignore
except Exception:  # noqa: BLE001
    Counter = None  # type: ignore

# Optional QuestDB sender
try:
    from questdb.ingress import Sender, TimestampNanos  # type: ignore
except Exception:  # noqa: BLE001
    Sender = None  # type: ignore
    TimestampNanos = None  # type: ignore

from trading_common import get_logger, get_settings
from trading_common.messaging import get_pulsar_client
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
    # value_score will be computed dynamically (not part of incoming model creation) and added before persistence
    # We don't store it as a dataclass field to avoid breaking existing instantiations; handled via setattr.


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
        # Messaging / streaming (mirrors news service lightweight pattern)
        self.pulsar_client = None
        self.producer = None
        self._pulsar_topic = 'persistent://trading/production/social-sentiment'
        # Internal lightweight resilience primitives for producer path
        self._pulsar_error_tokens = 15
        self._pulsar_error_last_refill = datetime.utcnow()
        try:
            # Reuse existing global counter name for consistency if already registered elsewhere
            from prometheus_client import Counter as _PC  # type: ignore
            try:
                self._pulsar_error_counter = _PC('pulsar_persistence_errors_total', 'Total Pulsar persistence or producer send errors')
            except Exception:  # noqa: BLE001
                self._pulsar_error_counter = None
        except Exception:  # noqa: BLE001
            self._pulsar_error_counter = None
        # Simple adaptive token bucket (looser than news; social sentiment volume moderate)
        self._bucket_rate = 40.0
        self._bucket_capacity = 120.0
        self._bucket_tokens = self._bucket_capacity
        self._bucket_last = datetime.utcnow()
        
        # API configurations (add your API keys to environment)
        self.unusual_whales_key = os.getenv('UNUSUAL_WHALES_API_KEY')
        self.quiver_quant_key = os.getenv('QUIVER_QUANT_API_KEY')
        self.sentiment_api_key = os.getenv('SENTIMENT_API_KEY')
        self.insider_trading_key = os.getenv('INSIDER_TRADING_API_KEY')
        
        # Performance tracking
        self.data_points_collected = 0
        self.high_value_signals = 0
        # Feature flags for multi-sink persistence
        self.enable_questdb_persist = os.getenv('ENABLE_QUESTDB_SOCIAL_PERSIST', os.getenv('ENABLE_QUESTDB_HIST_PERSIST','false')).lower() in ('1','true','yes')
        self.enable_postgres_persist = os.getenv('ENABLE_POSTGRES_SOCIAL_PERSIST','false').lower() in ('1','true','yes')
        self.enable_hist_dry_run = os.getenv('ENABLE_HIST_DRY_RUN','false').lower() in ('1','true','yes')
        self.enable_weaviate_persist = os.getenv('ENABLE_WEAVIATE_PERSIST','false').lower() in ('1','true','yes')
        self.ml_service_url = os.getenv('ML_SERVICE_URL', 'http://trading-ml:8001')
        # QuestDB configuration (align with market_data_service pattern)
        self.questdb_conf: Optional[str] = None
        if self.enable_questdb_persist and Sender:
            try:
                host = os.getenv('QUESTDB_HOST', 'trading-questdb')
                proto = os.getenv('QUESTDB_INGEST_PROTOCOL','tcp').lower().strip()
                if proto not in ('tcp','http'):
                    proto = 'tcp'
                if proto == 'tcp':
                    port = int(os.getenv('QUESTDB_LINE_TCP_PORT','9009'))
                    self.questdb_conf = f"tcp::addr={host}:{port};"
                else:
                    http_port = int(os.getenv('QUESTDB_HTTP_PORT','9000'))
                    self.questdb_conf = f"http::addr={host}:{http_port};"
            except Exception:
                self.questdb_conf = None
        # Metrics registry (local dict to avoid duplicate registration)
        self.prom_metrics: Dict[str, Any] = {}
        self._register_metrics()
        # Ingestion-time filtering threshold separate from retention pruning floor.
        try:
            self._social_ingest_value_min = float(os.getenv('SOCIAL_INGEST_VALUE_MIN','0.0'))
        except Exception:
            self._social_ingest_value_min = 0.0
        self._social_ingest_filter_metrics_ready = False
        
    async def initialize(self):
        """Initialize alternative data collector."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'AI-Trading-System/1.0'}
        )
        self.cache = get_trading_cache()
        logger.info("Alternative Data Collector initialized")
        # Initialize Pulsar producer (best-effort, never raise)
        try:
            self.pulsar_client = get_pulsar_client()
            self.producer = self.pulsar_client.create_producer(
                topic=self._pulsar_topic,
                producer_name='alternative-data-collector'
            )
            logger.info("AlternativeDataCollector Pulsar producer connected topic=%s", self._pulsar_topic)
        except Exception as e:  # noqa: BLE001
            logger.warning("AlternativeDataCollector Pulsar init failed: %s", e)
    
    async def close(self):
        """Close the alternative data collector."""
        if self.session:
            await self.session.close()
        try:
            if self.producer:
                self.producer.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.pulsar_client:
                self.pulsar_client.close()
        except Exception:  # noqa: BLE001
            pass
    
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
        sentiment_data: List[SocialSentiment] = []

        platforms = ['reddit', 'twitter', 'stocktwits']
        for platform in platforms:
            try:
                sentiment = await self._get_platform_sentiment(symbol, platform)
                if sentiment:
                    sentiment_data.append(sentiment)
            except Exception as e:  # noqa: BLE001
                logger.debug("social.sentiment.platform_failed symbol=%s platform=%s err=%s", symbol, platform, e)

        # Multi-sink persistence (best-effort, never raise)
        if sentiment_data:
            # Compute value_score for each row (heuristic) before persistence
            for r in sentiment_data:
                try:
                    setattr(r, 'value_score', self._compute_social_value_score(r))
                except Exception:
                    setattr(r, 'value_score', 0.0)
            # Lazy metrics registration for ingest filtering
            if not self._social_ingest_filter_metrics_ready:
                try:
                    from prometheus_client import Counter as _PC  # type: ignore
                    try:
                        self.prom_metrics['social_ingest_filtered_total'] = _PC('social_ingest_filtered_total','Social sentiment rows dropped at ingestion',['reason'])
                    except Exception:
                        self.prom_metrics['social_ingest_filtered_total'] = None
                    try:
                        self.prom_metrics['social_ingest_kept_total'] = _PC('social_ingest_kept_total','Social sentiment rows kept after ingestion filters')
                    except Exception:
                        self.prom_metrics['social_ingest_kept_total'] = None
                except Exception:
                    self.prom_metrics['social_ingest_filtered_total'] = None
                    self.prom_metrics['social_ingest_kept_total'] = None
                self._social_ingest_filter_metrics_ready = True
            # Apply ingestion-time filter
            filtered_rows: List[Any] = []
            filtered_ct = 0
            for r in sentiment_data:
                try:
                    if float(getattr(r,'value_score',0.0)) < self._social_ingest_value_min:
                        filtered_ct += 1
                        ctr = self.prom_metrics.get('social_ingest_filtered_total')
                        if ctr:
                            try: ctr.labels(reason='value_score').inc()
                            except Exception: pass
                        continue
                except Exception:
                    pass
                filtered_rows.append(r)
            if filtered_ct and getattr(logger,'info',None):
                try:
                    logger.info("social_ingest_filter", extra={"event":"social_ingest_filter","filtered":filtered_ct,"kept":len(filtered_rows),"value_min":self._social_ingest_value_min})
                except Exception:
                    pass
            if filtered_rows and self.prom_metrics.get('social_ingest_kept_total'):
                try: self.prom_metrics['social_ingest_kept_total'].inc(len(filtered_rows))
                except Exception: pass
            sentiment_data = filtered_rows
            await self._persist_social_sentiment(sentiment_data)
            # Optional vector indexing (aggregate items) best-effort
            if self.enable_weaviate_persist:
                try:
                    await self._index_social_to_weaviate(sentiment_data)
                except Exception as e:  # noqa: BLE001
                    logger.debug("social.weaviate.index_failed symbol=%s err=%s", symbol, e)
            # Publish to streaming topic (best-effort)
            try:
                await self._publish_social_sentiment(sentiment_data)
            except Exception as e:  # noqa: BLE001
                logger.debug("social.publish.failed symbol=%s err=%s", symbol, e)
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

    # ---------------- Value Scoring (Social) ---------------- #
    def _compute_social_value_score(self, row: SocialSentiment) -> float:
        """Compute heuristic value_score for social sentiment row.

        Components (0-1 weighted sum):
          sentiment_intensity (|sentiment_score|)         w=0.25
          volume_score                                     w=0.20
          momentum_score                                   w=0.15
          influential_mentions (scaled)                    w=0.15
          confidence                                       w=0.15
          topic_richness (#topics up to 10 normalized)     w=0.10
        """
        try:
            sent = abs(float(row.sentiment_score))
        except Exception:
            sent = 0.0
        try:
            vol = float(row.volume_score)
        except Exception:
            vol = 0.0
        try:
            mom = float(row.momentum_score)
        except Exception:
            mom = 0.0
        try:
            inf = min(int(row.influential_mentions), 50) / 50.0
        except Exception:
            inf = 0.0
        try:
            conf = float(row.confidence)
        except Exception:
            conf = 0.0
        try:
            topics = len(row.key_topics or [])
            topic_rich = min(topics, 10) / 10.0
        except Exception:
            topic_rich = 0.0
        score = 0.25*sent + 0.20*vol + 0.15*mom + 0.15*inf + 0.15*conf + 0.10*topic_rich
        if score < 0: score = 0.0
        if score > 1: score = 1.0
        return score
    
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

    # ---------------------- Persistence & Metrics Helpers ---------------------- #

    def _register_metrics(self):  # type: ignore[no-redef]
        """Register Prometheus counters (idempotent)."""
        if not Counter:
            return
        try:
            # New normalized metric names *_rows_persisted_total (keep legacy for dashboards until migrated)
            if 'social_postgres_rows_persisted_total' not in self.prom_metrics:
                try:
                    self.prom_metrics['social_postgres_rows_persisted_total'] = Counter(
                        'social_postgres_rows_persisted_total', 'Total social sentiment rows persisted to Postgres'
                    )
                except Exception:
                    self.prom_metrics['social_postgres_rows_persisted_total'] = None  # type: ignore
            if 'social_questdb_rows_persisted_total' not in self.prom_metrics:
                try:
                    self.prom_metrics['social_questdb_rows_persisted_total'] = Counter(
                        'social_questdb_rows_persisted_total', 'Total social sentiment rows persisted to QuestDB'
                    )
                except Exception:
                    self.prom_metrics['social_questdb_rows_persisted_total'] = None  # type: ignore
            # Legacy names (backward compatibility)
            if 'social_postgres_rows_total' not in self.prom_metrics:
                try:
                    self.prom_metrics['social_postgres_rows_total'] = Counter(
                        'social_postgres_rows_total', 'Total social sentiment rows persisted to Postgres (legacy)'
                    )
                except Exception:
                    self.prom_metrics['social_postgres_rows_total'] = None  # type: ignore
            if 'social_questdb_rows_total' not in self.prom_metrics:
                try:
                    self.prom_metrics['social_questdb_rows_total'] = Counter(
                        'social_questdb_rows_total', 'Total social sentiment rows persisted to QuestDB (legacy)'
                    )
                except Exception:
                    self.prom_metrics['social_questdb_rows_total'] = None  # type: ignore
            if 'social_persist_errors_total' not in self.prom_metrics:
                self.prom_metrics['social_persist_errors_total'] = Counter(
                    'social_persist_errors_total', 'Total errors during social sentiment persistence'
                )
            if 'social_weaviate_indexed_total' not in self.prom_metrics:
                self.prom_metrics['social_weaviate_indexed_total'] = Counter(
                    'social_weaviate_indexed_total', 'Total social sentiment objects indexed into Weaviate'
                )
            if 'social_value_score_observations_total' not in self.prom_metrics:
                try:
                    self.prom_metrics['social_value_score_observations_total'] = Counter(
                        'social_value_score_observations_total','Total social sentiment rows evaluated for value score'
                    )
                except Exception:
                    self.prom_metrics['social_value_score_observations_total'] = None
            if 'social_low_value_pruned_total' not in self.prom_metrics:
                try:
                    self.prom_metrics['social_low_value_pruned_total'] = Counter(
                        'social_low_value_pruned_total','Rows (social) skipped pre-persist due to very low value_score'
                    )
                except Exception:
                    self.prom_metrics['social_low_value_pruned_total'] = None
        except Exception:  # noqa: BLE001
            pass

    async def _persist_social_sentiment(self, rows: List[SocialSentiment]):  # type: ignore[no-redef]
        """Persist social sentiment to QuestDB + Postgres (feature-gated)."""
        if not rows:
            return
        # Filter out ultra-low value rows early (does not affect already persisted history)
        try:
            value_floor = float(os.getenv('SOCIAL_VALUE_SCORE_FLOOR','0.0'))
        except Exception:
            value_floor = 0.0
        filtered: List[SocialSentiment] = []
        for r in rows:
            vs = float(getattr(r, 'value_score', 0.0)) if hasattr(r, 'value_score') else 0.0
            ctr_obs = self.prom_metrics.get('social_value_score_observations_total')
            if ctr_obs:
                try: ctr_obs.inc()
                except Exception: pass
            if vs >= value_floor:
                filtered.append(r)
            else:
                ctr_pruned = self.prom_metrics.get('social_low_value_pruned_total')
                if ctr_pruned:
                    try: ctr_pruned.inc()
                    except Exception: pass
        rows = filtered
        if not rows:
            return
        # QuestDB path
        if self.enable_questdb_persist and not self.enable_hist_dry_run and Sender and self.questdb_conf:
            try:
                with Sender.from_conf(self.questdb_conf) as s:  # type: ignore[arg-type]
                    for r in rows:
                        at_ts = (
                            TimestampNanos.from_datetime(r.timestamp)
                            if TimestampNanos is not None else int(r.timestamp.timestamp() * 1_000_000_000)
                        )
                        s.row(
                            'social_events',
                            symbols={
                                'platform': r.platform,
                                'symbol': r.symbol.upper(),
                            },
                            columns={
                                'sentiment': float(r.sentiment_score),
                                # Flatten additional numeric features
                                'volume_score': float(r.volume_score),
                                'momentum_score': float(r.momentum_score),
                                'influential_mentions': int(r.influential_mentions),
                                'confidence': float(r.confidence),
                                'value_score': float(getattr(r, 'value_score', 0.0)),
                                'topics': ','.join(r.key_topics[:20]) if r.key_topics else ''
                            },
                            at=at_ts,
                        )
                    s.flush()
                ctr_new = self.prom_metrics.get('social_questdb_rows_persisted_total')
                ctr_legacy = self.prom_metrics.get('social_questdb_rows_total')
                for ctr in (ctr_new, ctr_legacy):
                    if ctr:
                        try:
                            ctr.inc(len(rows))
                        except Exception:
                            pass
            except Exception as e:  # noqa: BLE001
                err_ctr = self.prom_metrics.get('social_persist_errors_total')
                if err_ctr:
                    try:
                        err_ctr.inc()
                    except Exception:
                        pass
                logger.debug("QuestDB social persist failed: %s", e)
        # Postgres path
        if self.enable_postgres_persist and not self.enable_hist_dry_run:
            try:
                from trading_common.database_manager import get_database_manager  # type: ignore
                dbm = await get_database_manager()
                async with dbm.get_postgres() as pg:  # type: ignore[attr-defined]
                    await pg.execute("""
                        CREATE TABLE IF NOT EXISTS social_events (
                            symbol TEXT NOT NULL,
                            platform TEXT NOT NULL,
                            ts TIMESTAMPTZ NOT NULL,
                            sentiment_score DOUBLE PRECISION NOT NULL,
                            volume_score DOUBLE PRECISION NOT NULL,
                            momentum_score DOUBLE PRECISION NOT NULL,
                            key_topics JSONB,
                            influential_mentions INT,
                            confidence DOUBLE PRECISION,
                            value_score DOUBLE PRECISION DEFAULT 0,
                            inserted_at TIMESTAMPTZ DEFAULT NOW(),
                            PRIMARY KEY(symbol, platform, ts)
                        )
                    """)
                    insert_rows = [
                        (
                            r.symbol.upper(), r.platform, r.timestamp,
                            float(r.sentiment_score), float(r.volume_score), float(r.momentum_score),
                            json.dumps(r.key_topics[:50] if r.key_topics else []),
                            int(r.influential_mentions), float(r.confidence), float(getattr(r, 'value_score', 0.0))
                        ) for r in rows
                    ]
                    try:
                        await pg.executemany(
                            """
                            INSERT INTO social_events(symbol, platform, ts, sentiment_score, volume_score, momentum_score, key_topics, influential_mentions, confidence, value_score)
                            VALUES($1,$2,$3,$4,$5,$6,$7::jsonb,$8,$9,$10)
                            ON CONFLICT DO NOTHING
                            """,
                            insert_rows
                        )
                    except Exception:
                        for row in insert_rows:
                            try:
                                await pg.execute(
                                    """
                                    INSERT INTO social_events(symbol, platform, ts, sentiment_score, volume_score, momentum_score, key_topics, influential_mentions, confidence, value_score)
                                    VALUES($1,$2,$3,$4,$5,$6,$7::jsonb,$8,$9,$10) ON CONFLICT DO NOTHING
                                    """,
                                    *row
                                )
                            except Exception:
                                pass
                    ctr_new = self.prom_metrics.get('social_postgres_rows_persisted_total')
                    ctr_legacy = self.prom_metrics.get('social_postgres_rows_total')
                    for ctr in (ctr_new, ctr_legacy):
                        if ctr:
                            try:
                                ctr.inc(len(rows))
                            except Exception:
                                pass
            except Exception as e:  # noqa: BLE001
                err_ctr = self.prom_metrics.get('social_persist_errors_total')
                if err_ctr:
                    try:
                        err_ctr.inc()
                    except Exception:
                        pass
                logger.debug("Postgres social persist failed: %s", e)

    async def _index_social_to_weaviate(self, rows: List[SocialSentiment]):  # type: ignore[no-redef]
        """Send social sentiment rows to ML service, fallback to direct indexing.

        Simplified retry loop (3 attempts) then fallback on final failure.
        """
        if not rows or not self.session:
            return
        payload_items = []
        for r in rows:
            try:
                payload_items.append({
                    'symbol': r.symbol.upper(),
                    'platform': r.platform,
                    'sentiment': float(r.sentiment_score),
                    'volume_score': float(r.volume_score),
                    'momentum_score': float(r.momentum_score),
                    'influential_mentions': int(r.influential_mentions),
                    'confidence': float(r.confidence),
                    'value_score': float(getattr(r, 'value_score', 0.0)),
                    'topics': r.key_topics[:25] if r.key_topics else [],
                    'timestamp': r.timestamp.isoformat()
                })
            except Exception:
                continue
        if not payload_items:
            return
        endpoint = self.ml_service_url.rstrip('/') + '/vector/index/social'
        for attempt in range(3):
            try:
                async with self.session.post(endpoint, json={'items': payload_items}, timeout=aiohttp.ClientTimeout(total=25)) as resp:
                    if resp.status == 200:
                        ctr = self.prom_metrics.get('social_weaviate_indexed_total')
                        if ctr:
                            try:
                                ctr.inc(len(payload_items))
                            except Exception:
                                pass
                        return
                    await resp.text()  # consume body
                    raise RuntimeError(f"HTTP {resp.status}")
            except Exception:  # noqa: BLE001
                if attempt < 2:
                    await asyncio.sleep(1.2 * (attempt + 1))
                    continue
                # Fallback path (final attempt failed)
                try:
                    from shared.vector.indexing import index_social_fallback  # type: ignore
                    redis = None
                    try:
                        if self.cache:
                            redis = getattr(self.cache, 'redis', None) or self.cache
                    except Exception:  # noqa: BLE001
                        redis = None
                    inserted = 0
                    try:
                        inserted = await index_social_fallback(list(payload_items), redis=redis)
                    except Exception:
                        inserted = 0
                    if inserted:
                        try:
                            from prometheus_client import Counter as _PC  # type: ignore
                            if 'social_weaviate_fallback_indexed_total' not in self.prom_metrics:
                                try:
                                    self.prom_metrics['social_weaviate_fallback_indexed_total'] = _PC(
                                        'social_weaviate_fallback_indexed_total', 'Total social sentiment objects indexed via fallback path'
                                    )
                                except Exception:
                                    self.prom_metrics['social_weaviate_fallback_indexed_total'] = None
                            ctr_fb = self.prom_metrics.get('social_weaviate_fallback_indexed_total')
                            if ctr_fb:
                                try:
                                    ctr_fb.inc(inserted)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass
                return

    # ---------------------- Streaming (Pulsar) Helpers ---------------------- #
    def _bucket_acquire(self) -> bool:
        """Acquire a token from adaptive bucket (non-blocking)."""
        now = datetime.utcnow()
        elapsed = (now - self._bucket_last).total_seconds()
        self._bucket_last = now
        self._bucket_tokens = min(self._bucket_capacity, self._bucket_tokens + elapsed * self._bucket_rate)
        if self._bucket_tokens >= 1.0:
            self._bucket_tokens -= 1.0
            return True
        return False

    def _rate_limited_pulsar_error(self, msg: str):
        """Rate-limit noisy Pulsar error logs (token bucket)."""
        refill_interval = 30  # seconds
        now = datetime.utcnow()
        if (now - self._pulsar_error_last_refill).total_seconds() > refill_interval:
            self._pulsar_error_tokens = 15
            self._pulsar_error_last_refill = now
        if self._pulsar_error_tokens > 0:
            self._pulsar_error_tokens -= 1
            try:
                logger.warning(msg)
            except Exception:  # noqa: BLE001
                pass

    async def _publish_social_sentiment(self, rows: List[SocialSentiment]):
        """Publish social sentiment events to streaming topic (one per row).

        Best-effort: skips silently if producer unavailable or token bucket depleted.
        """
        if not rows or not self.producer:
            return
        for r in rows:
            if not self._bucket_acquire():  # backpressure skip
                try:
                    from prometheus_client import Counter as _PC  # type: ignore
                    if 'social_publish_skipped' not in self.prom_metrics:
                        try:
                            self.prom_metrics['social_publish_skipped'] = _PC('producer_publish_skipped_total','Messages skipped before publish due to backpressure',['service','reason'])
                        except Exception:
                            self.prom_metrics['social_publish_skipped'] = None
                    ctr = self.prom_metrics.get('social_publish_skipped')
                    if ctr:
                        try:
                            ctr.labels(service='social', reason='rate_limited').inc()
                        except Exception:
                            pass
                except Exception:  # noqa: BLE001
                    pass
                continue
            payload = {
                'symbol': r.symbol.upper(),
                'platform': r.platform,
                'sentiment_score': float(r.sentiment_score),
                'volume_score': float(r.volume_score),
                'momentum_score': float(r.momentum_score),
                'influential_mentions': int(r.influential_mentions),
                'confidence': float(r.confidence),
                'topics': r.key_topics[:25] if r.key_topics else [],
                'timestamp': r.timestamp.isoformat()
            }
            try:
                self.producer.send(json.dumps(payload, default=str).encode('utf-8'))
            except Exception as e:  # noqa: BLE001
                if self._pulsar_error_counter:
                    try:
                        self._pulsar_error_counter.inc()
                    except Exception:  # noqa: BLE001
                        pass
                self._rate_limited_pulsar_error(f"Failed to publish social sentiment: {e}")