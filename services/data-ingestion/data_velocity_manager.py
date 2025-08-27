#!/usr/bin/env python3
"""
Data Velocity Manager - Multi-frequency data ingestion orchestrator.
Implements tiered data collection for different trading strategies.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict

from pydantic import BaseModel
import aioredis
from prometheus_client import Counter, Gauge, Histogram
import pandas as pd

from trading_common.config import get_settings

logger = logging.getLogger(__name__)

# Metrics
data_fetch_counter = Counter('data_fetch_total', 'Total data fetches', ['source', 'tier'])
data_latency_histogram = Histogram('data_fetch_latency_seconds', 'Data fetch latency', ['source', 'tier'])
active_streams_gauge = Gauge('active_data_streams', 'Number of active data streams', ['type'])
urgent_events_counter = Counter('urgent_events_total', 'Total urgent events detected', ['type'])
api_quota_gauge = Gauge('api_quota_remaining', 'Remaining API quota', ['service'])


class DataTier(Enum):
    """Data velocity tiers for different urgency levels."""
    REALTIME = "realtime"      # WebSocket streaming (milliseconds)
    URGENT = "urgent"           # 30 seconds - Breaking news, major price moves
    FAST = "fast"               # 2 minutes - Active trading signals
    STANDARD = "standard"       # 15 minutes - Regular updates
    BACKGROUND = "background"   # 1 hour - Historical analysis


class TradingStrategy(Enum):
    """Trading strategy types that require different data velocities."""
    SCALPING = "scalping"           # Requires REALTIME
    DAY_TRADING = "day_trading"     # Requires URGENT/FAST
    SWING_TRADING = "swing_trading" # Requires FAST/STANDARD
    POSITION = "position"           # Requires STANDARD/BACKGROUND
    LONG_TERM = "long_term"         # Requires BACKGROUND


@dataclass
class DataSource:
    """Configuration for a data source."""
    name: str
    tier: DataTier
    interval_seconds: int
    max_requests_per_day: int
    current_requests: int = 0
    last_fetch: Optional[datetime] = None
    callback: Optional[Callable] = None
    enabled: bool = True
    priority: int = 1  # Higher = more important


class UrgentEvent(BaseModel):
    """Represents an urgent market event requiring immediate attention."""
    event_type: str  # price_surge, volume_spike, breaking_news, earnings
    symbol: Optional[str]
    magnitude: float  # Impact score 0-1
    timestamp: datetime
    data: Dict[str, Any]
    requires_immediate_action: bool = False


class DataVelocityManager:
    """Manages multi-frequency data collection based on market conditions and strategy."""
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        """Initialize the data velocity manager."""
        self.settings = get_settings()
        self.redis = redis_client
        self.data_sources: Dict[str, DataSource] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.urgent_queue: asyncio.Queue = asyncio.Queue()
        self.active_strategies: List[TradingStrategy] = []
        
        # Load configuration
        self._load_configuration()
        
        # Track API usage
        self.api_usage = defaultdict(lambda: {'requests': 0, 'reset_time': None})
        
        # WebSocket connections
        self.websocket_connections = {}
        
        logger.info("Data Velocity Manager initialized")
    
    def _load_configuration(self):
        """Load polling intervals and thresholds from environment."""
        # Polling intervals
        self.intervals = {
            DataTier.URGENT: int(os.getenv('DATA_POLLING_URGENT', '30')),
            DataTier.FAST: int(os.getenv('DATA_POLLING_FAST', '120')),
            DataTier.STANDARD: int(os.getenv('DATA_POLLING_STANDARD', '900')),
            DataTier.BACKGROUND: int(os.getenv('DATA_POLLING_BACKGROUND', '3600'))
        }
        
        # Thresholds for urgent events
        self.thresholds = {
            'price_change': float(os.getenv('PRICE_CHANGE_URGENT_THRESHOLD', '2.0')),
            'volume_surge': float(os.getenv('VOLUME_SURGE_THRESHOLD', '3.0')),
            'sentiment_score': float(os.getenv('NEWS_SENTIMENT_URGENT_SCORE', '0.8'))
        }
        
        # Feature flags
        self.features = {
            'streaming': os.getenv('MARKET_DATA_STREAMING', 'true').lower() == 'true',
            'websockets': os.getenv('MARKET_DATA_WEBSOCKET_ENABLED', 'true').lower() == 'true',
            'strategy_routing': os.getenv('ENABLE_STRATEGY_ROUTING', 'true').lower() == 'true',
            'day_trading': os.getenv('FEATURE_DAY_TRADING_ENABLED', 'true').lower() == 'true',
            'multi_strategy': os.getenv('FEATURE_MULTI_STRATEGY_SUPPORT', 'true').lower() == 'true'
        }
        
        # API limits
        self.api_limits = {
            'newsapi': int(os.getenv('NEWS_MAX_REQUESTS_PER_DAY', '900')),
            'polygon': 100 * 60 * 24,  # 100 req/min
            'alpaca': 200 * 60 * 24,   # 200 req/min
            'reddit': 60 * 24          # 60 req/min
        }
    
    def register_data_source(
        self, 
        name: str, 
        tier: DataTier,
        callback: Callable,
        max_requests_per_day: Optional[int] = None
    ):
        """Register a new data source with specified tier and callback."""
        interval = self.intervals.get(tier, 900)
        max_requests = max_requests_per_day or self.api_limits.get(name.lower(), 10000)
        
        self.data_sources[name] = DataSource(
            name=name,
            tier=tier,
            interval_seconds=interval,
            max_requests_per_day=max_requests,
            callback=callback,
            enabled=True,
            priority=self._get_priority_for_tier(tier)
        )
        
        logger.info(f"Registered data source: {name} at tier {tier.value} (interval: {interval}s)")
    
    def _get_priority_for_tier(self, tier: DataTier) -> int:
        """Get priority based on data tier."""
        priorities = {
            DataTier.REALTIME: 10,
            DataTier.URGENT: 8,
            DataTier.FAST: 6,
            DataTier.STANDARD: 4,
            DataTier.BACKGROUND: 2
        }
        return priorities.get(tier, 1)
    
    async def start(self):
        """Start all data collection tasks."""
        logger.info("Starting Data Velocity Manager...")
        
        # Start WebSocket connections if enabled
        if self.features['websockets']:
            await self._start_websocket_streams()
        
        # Start polling tasks for each data source
        for name, source in self.data_sources.items():
            if source.enabled and source.tier != DataTier.REALTIME:
                task = asyncio.create_task(self._polling_loop(source))
                self.active_tasks[name] = task
        
        # Start urgent event processor
        asyncio.create_task(self._process_urgent_events())
        
        # Start API quota monitor
        asyncio.create_task(self._monitor_api_quotas())
        
        logger.info(f"Started {len(self.active_tasks)} polling tasks")
        active_streams_gauge.labels(type='polling').set(len(self.active_tasks))
    
    async def _polling_loop(self, source: DataSource):
        """Main polling loop for a data source."""
        logger.info(f"Starting polling loop for {source.name} (tier: {source.tier.value})")
        
        while source.enabled:
            try:
                # Check API quota
                if not self._check_api_quota(source.name):
                    logger.warning(f"API quota exceeded for {source.name}, skipping")
                    await asyncio.sleep(source.interval_seconds)
                    continue
                
                # Fetch data with metrics
                with data_latency_histogram.labels(source=source.name, tier=source.tier.value).time():
                    if source.callback:
                        data = await source.callback()
                        
                        # Check for urgent conditions
                        if self.features['strategy_routing']:
                            await self._check_urgent_conditions(source.name, data)
                
                # Update metrics
                source.current_requests += 1
                source.last_fetch = datetime.utcnow()
                data_fetch_counter.labels(source=source.name, tier=source.tier.value).inc()
                
                # Update API usage
                self._update_api_usage(source.name)
                
                # Dynamic interval adjustment based on market conditions
                interval = await self._get_dynamic_interval(source)
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in polling loop for {source.name}: {e}")
                await asyncio.sleep(source.interval_seconds)
    
    async def _get_dynamic_interval(self, source: DataSource) -> int:
        """Dynamically adjust polling interval based on market conditions."""
        base_interval = source.interval_seconds
        
        # During market hours, increase frequency
        if self._is_market_hours():
            if source.tier == DataTier.FAST:
                return base_interval // 2  # Double the frequency
            elif source.tier == DataTier.STANDARD:
                return int(base_interval * 0.75)  # 25% faster
        
        # If day trading is active and enabled
        if self.features['day_trading'] and TradingStrategy.DAY_TRADING in self.active_strategies:
            if source.tier in [DataTier.URGENT, DataTier.FAST]:
                return base_interval // 2  # Double frequency for day trading
        
        return base_interval
    
    def _is_market_hours(self) -> bool:
        """Check if market is currently open."""
        now = datetime.utcnow()
        # NYSE hours: 9:30 AM - 4:00 PM ET (14:30 - 21:00 UTC)
        market_open = now.replace(hour=14, minute=30, second=0)
        market_close = now.replace(hour=21, minute=0, second=0)
        
        # Check if weekday and within hours
        if now.weekday() < 5:  # Monday = 0, Friday = 4
            return market_open <= now <= market_close
        return False
    
    async def _check_urgent_conditions(self, source: str, data: Dict[str, Any]):
        """Check if data contains urgent conditions requiring immediate action."""
        urgent_event = None
        
        # Check price movements
        if 'price_change_percent' in data:
            if abs(data['price_change_percent']) > self.thresholds['price_change']:
                urgent_event = UrgentEvent(
                    event_type='price_surge',
                    symbol=data.get('symbol'),
                    magnitude=min(abs(data['price_change_percent']) / 10, 1.0),
                    timestamp=datetime.utcnow(),
                    data=data,
                    requires_immediate_action=True
                )
        
        # Check volume spikes
        elif 'volume_ratio' in data:
            if data['volume_ratio'] > self.thresholds['volume_surge']:
                urgent_event = UrgentEvent(
                    event_type='volume_spike',
                    symbol=data.get('symbol'),
                    magnitude=min(data['volume_ratio'] / 10, 1.0),
                    timestamp=datetime.utcnow(),
                    data=data,
                    requires_immediate_action=True
                )
        
        # Check news sentiment
        elif 'sentiment_score' in data:
            if abs(data['sentiment_score']) > self.thresholds['sentiment_score']:
                urgent_event = UrgentEvent(
                    event_type='breaking_news',
                    symbol=data.get('symbol'),
                    magnitude=abs(data['sentiment_score']),
                    timestamp=datetime.utcnow(),
                    data=data,
                    requires_immediate_action=True
                )
        
        if urgent_event:
            await self.urgent_queue.put(urgent_event)
            urgent_events_counter.labels(type=urgent_event.event_type).inc()
            logger.info(f"Urgent event detected: {urgent_event.event_type} for {urgent_event.symbol}")
    
    async def _process_urgent_events(self):
        """Process urgent events that require immediate action."""
        logger.info("Started urgent event processor")
        
        while True:
            try:
                # Wait for urgent events
                event = await self.urgent_queue.get()
                
                logger.warning(f"Processing urgent event: {event.event_type} - {event.symbol}")
                
                # Store in Redis for immediate processing
                if self.redis:
                    await self.redis.lpush(
                        'urgent_events',
                        event.json()
                    )
                    await self.redis.expire('urgent_events', 3600)  # 1 hour TTL
                
                # Trigger immediate strategy evaluation if day trading is enabled
                if self.features['day_trading'] and event.requires_immediate_action:
                    await self._trigger_strategy_evaluation(event)
                
            except Exception as e:
                logger.error(f"Error processing urgent event: {e}")
    
    async def _trigger_strategy_evaluation(self, event: UrgentEvent):
        """Trigger immediate strategy evaluation for urgent events."""
        # This would connect to your strategy engine
        logger.info(f"Triggering strategy evaluation for {event.event_type}")
        
        # Store event for strategy engine
        if self.redis:
            await self.redis.publish(
                'strategy_triggers',
                event.json()
            )
    
    def _check_api_quota(self, source: str) -> bool:
        """Check if API quota allows another request."""
        usage = self.api_usage[source.lower()]
        limit = self.api_limits.get(source.lower(), 10000)
        
        # Reset daily counter if needed
        if usage['reset_time'] and datetime.utcnow() > usage['reset_time']:
            usage['requests'] = 0
            usage['reset_time'] = None
        
        # Check quota
        if usage['requests'] >= limit:
            return False
        
        return True
    
    def _update_api_usage(self, source: str):
        """Update API usage tracking."""
        usage = self.api_usage[source.lower()]
        usage['requests'] += 1
        
        # Set reset time if not set
        if not usage['reset_time']:
            tomorrow = datetime.utcnow() + timedelta(days=1)
            usage['reset_time'] = tomorrow.replace(hour=0, minute=0, second=0)
        
        # Update metrics
        remaining = self.api_limits.get(source.lower(), 10000) - usage['requests']
        api_quota_gauge.labels(service=source).set(remaining)
    
    async def _monitor_api_quotas(self):
        """Monitor API quotas and log warnings."""
        while True:
            try:
                for source, usage in self.api_usage.items():
                    limit = self.api_limits.get(source, 10000)
                    used_percent = (usage['requests'] / limit) * 100
                    
                    if used_percent > 90:
                        logger.warning(f"API quota warning: {source} at {used_percent:.1f}% usage")
                    elif used_percent > 75:
                        logger.info(f"API quota notice: {source} at {used_percent:.1f}% usage")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring API quotas: {e}")
                await asyncio.sleep(300)
    
    async def _start_websocket_streams(self):
        """Start WebSocket connections for real-time data."""
        logger.info("Initializing WebSocket streams...")
        
        # This would connect to your WebSocket implementations
        # Placeholder for WebSocket initialization
        active_streams_gauge.labels(type='websocket').set(0)
    
    def set_active_strategies(self, strategies: List[TradingStrategy]):
        """Update active trading strategies to optimize data collection."""
        self.active_strategies = strategies
        logger.info(f"Updated active strategies: {[s.value for s in strategies]}")
        
        # Adjust data source priorities based on strategies
        for strategy in strategies:
            self._optimize_for_strategy(strategy)
    
    def _optimize_for_strategy(self, strategy: TradingStrategy):
        """Optimize data collection for specific trading strategy."""
        optimizations = {
            TradingStrategy.SCALPING: [DataTier.REALTIME],
            TradingStrategy.DAY_TRADING: [DataTier.URGENT, DataTier.FAST],
            TradingStrategy.SWING_TRADING: [DataTier.FAST, DataTier.STANDARD],
            TradingStrategy.POSITION: [DataTier.STANDARD, DataTier.BACKGROUND],
            TradingStrategy.LONG_TERM: [DataTier.BACKGROUND]
        }
        
        preferred_tiers = optimizations.get(strategy, [])
        
        # Boost priority for preferred data sources
        for source in self.data_sources.values():
            if source.tier in preferred_tiers:
                source.priority += 2
                logger.debug(f"Boosted priority for {source.name} due to {strategy.value} strategy")
    
    async def stop(self):
        """Stop all data collection tasks."""
        logger.info("Stopping Data Velocity Manager...")
        
        # Cancel all polling tasks
        for name, task in self.active_tasks.items():
            task.cancel()
            logger.debug(f"Cancelled task: {name}")
        
        # Close WebSocket connections
        for conn in self.websocket_connections.values():
            await conn.close()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        logger.info("Data Velocity Manager stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of data velocity manager."""
        return {
            'active_sources': len([s for s in self.data_sources.values() if s.enabled]),
            'active_tasks': len(self.active_tasks),
            'active_strategies': [s.value for s in self.active_strategies],
            'market_hours': self._is_market_hours(),
            'api_usage': {
                name: {
                    'used': usage['requests'],
                    'limit': self.api_limits.get(name, 10000),
                    'percent': (usage['requests'] / self.api_limits.get(name, 10000)) * 100
                }
                for name, usage in self.api_usage.items()
            },
            'features': self.features
        }


# Example usage
async def main():
    """Example usage of DataVelocityManager."""
    manager = DataVelocityManager()
    
    # Register data sources with different tiers
    async def fetch_market_data():
        # Simulate market data fetch
        return {'symbol': 'AAPL', 'price': 150.0, 'volume': 1000000}
    
    async def fetch_news():
        # Simulate news fetch
        return {'title': 'Breaking news', 'sentiment_score': 0.9}
    
    manager.register_data_source('polygon_urgent', DataTier.URGENT, fetch_market_data)
    manager.register_data_source('newsapi_fast', DataTier.FAST, fetch_news)
    
    # Set active strategies
    manager.set_active_strategies([TradingStrategy.DAY_TRADING])
    
    # Start manager
    await manager.start()
    
    # Run for a while
    await asyncio.sleep(60)
    
    # Get status
    status = manager.get_status()
    logger.info(f"Manager status: {status}")
    
    # Stop manager
    await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())