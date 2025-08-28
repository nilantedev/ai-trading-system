#!/usr/bin/env python3
"""
Strategy-Aware Data Router
Routes data intelligently based on active trading strategies and market conditions.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict

import redis.asyncio as aioredis
from prometheus_client import Counter, Gauge, Histogram
import pandas as pd
import numpy as np

from trading_common.config import get_settings

logger = logging.getLogger(__name__)

# Metrics
data_routed_counter = Counter('data_routed_total', 'Total data points routed', ['strategy', 'priority'])
routing_latency_histogram = Histogram('routing_latency_ms', 'Data routing latency', ['strategy'])
active_strategies_gauge = Gauge('active_trading_strategies', 'Number of active strategies')
priority_queue_gauge = Gauge('priority_queue_size', 'Size of priority queues', ['priority'])
strategy_signals_counter = Counter('strategy_signals_generated', 'Trading signals generated', ['strategy', 'signal_type'])


class SignalType(Enum):
    """Types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    ALERT = "alert"


class DataPriority(Enum):
    """Priority levels for data routing."""
    CRITICAL = 1    # Immediate action required
    HIGH = 2        # Process within seconds
    MEDIUM = 3      # Process within minutes
    LOW = 4         # Background processing
    ARCHIVE = 5     # Historical data


@dataclass
class StrategyRequirements:
    """Data requirements for a trading strategy."""
    name: str
    min_data_frequency: int  # Seconds
    required_data_types: List[str]
    indicators: List[str]
    lookback_period: int  # Minutes
    risk_tolerance: float
    max_position_size: float
    enabled: bool = True


@dataclass
class DataPacket:
    """Represents a packet of data to be routed."""
    data_type: str  # trade, quote, bar, news, sentiment
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime
    priority: DataPriority
    source: str
    metadata: Dict[str, Any] = None


@dataclass
class TradingSignal:
    """Represents a trading signal generated from data."""
    strategy: str
    symbol: str
    signal_type: SignalType
    confidence: float  # 0-1
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size: Optional[float]
    timestamp: datetime
    metadata: Dict[str, Any] = None


class StrategyDataRouter:
    """Routes data intelligently to appropriate trading strategies."""
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        """Initialize the strategy data router."""
        self.settings = get_settings()
        self.redis = redis_client
        
        # Strategy configurations
        self.strategies: Dict[str, StrategyRequirements] = {}
        self.active_strategies: Set[str] = set()
        
        # Data queues by priority
        self.priority_queues: Dict[DataPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in DataPriority
        }
        
        # Strategy-specific processors
        self.strategy_processors: Dict[str, Any] = {}
        
        # Market state tracking
        self.market_state = {
            'volatility': 'normal',  # low, normal, high, extreme
            'trend': 'neutral',      # bullish, bearish, neutral
            'volume': 'average',     # low, average, high
            'market_hours': False
        }
        
        # Performance tracking
        self.routing_stats = defaultdict(lambda: {'routed': 0, 'processed': 0, 'signals': 0})
        
        self._initialize_strategies()
        logger.info("Strategy Data Router initialized")
    
    def _initialize_strategies(self):
        """Initialize trading strategy configurations."""
        # Scalping strategy
        if os.getenv('FEATURE_DAY_TRADING_ENABLED', 'true').lower() == 'true':
            self.strategies['scalping'] = StrategyRequirements(
                name='scalping',
                min_data_frequency=1,  # 1 second
                required_data_types=['trades', 'quotes', 'orderbook'],
                indicators=['spread', 'volume_profile', 'order_flow'],
                lookback_period=5,  # 5 minutes
                risk_tolerance=0.001,  # 0.1% per trade
                max_position_size=10000
            )
        
        # Day trading strategy
        if os.getenv('FEATURE_DAY_TRADING_ENABLED', 'true').lower() == 'true':
            self.strategies['day_trading'] = StrategyRequirements(
                name='day_trading',
                min_data_frequency=30,  # 30 seconds
                required_data_types=['trades', 'quotes', 'bars', 'news'],
                indicators=['rsi', 'macd', 'vwap', 'volume'],
                lookback_period=60,  # 1 hour
                risk_tolerance=0.02,  # 2% per trade
                max_position_size=25000
            )
        
        # Swing trading strategy
        self.strategies['swing_trading'] = StrategyRequirements(
            name='swing_trading',
            min_data_frequency=120,  # 2 minutes
            required_data_types=['bars', 'news', 'sentiment'],
            indicators=['sma_20', 'sma_50', 'bollinger_bands', 'adx'],
            lookback_period=1440,  # 1 day
            risk_tolerance=0.05,  # 5% per trade
            max_position_size=50000
        )
        
        # Position trading strategy
        self.strategies['position_trading'] = StrategyRequirements(
            name='position_trading',
            min_data_frequency=900,  # 15 minutes
            required_data_types=['bars', 'news', 'fundamentals'],
            indicators=['sma_50', 'sma_200', 'trend', 'support_resistance'],
            lookback_period=10080,  # 1 week
            risk_tolerance=0.10,  # 10% per trade
            max_position_size=100000
        )
    
    async def route_data(self, packet: DataPacket):
        """Route data packet to appropriate strategies based on requirements."""
        start_time = datetime.utcnow()
        
        # Determine priority based on market conditions
        priority = await self._determine_priority(packet)
        packet.priority = priority
        
        # Add to priority queue
        await self.priority_queues[priority].put(packet)
        priority_queue_gauge.labels(priority=priority.name).set(
            self.priority_queues[priority].qsize()
        )
        
        # Route to relevant strategies
        routed_strategies = []
        
        for strategy_name, requirements in self.strategies.items():
            if not requirements.enabled or strategy_name not in self.active_strategies:
                continue
            
            # Check if strategy needs this data type
            if packet.data_type in requirements.required_data_types:
                await self._route_to_strategy(strategy_name, packet)
                routed_strategies.append(strategy_name)
                
                # Update metrics
                data_routed_counter.labels(
                    strategy=strategy_name,
                    priority=priority.name
                ).inc()
        
        # Calculate routing latency
        latency = (datetime.utcnow() - start_time).total_seconds() * 1000
        for strategy in routed_strategies:
            routing_latency_histogram.labels(strategy=strategy).observe(latency)
        
        logger.debug(f"Routed {packet.data_type} for {packet.symbol} to {len(routed_strategies)} strategies")
    
    async def _determine_priority(self, packet: DataPacket) -> DataPriority:
        """Determine routing priority based on data and market conditions."""
        # Critical priority for significant events
        if packet.metadata and packet.metadata.get('urgent'):
            return DataPriority.CRITICAL
        
        # Check for market moving events
        if packet.data_type == 'news':
            sentiment = packet.data.get('sentiment_score', 0)
            if abs(sentiment) > 0.8:
                return DataPriority.CRITICAL
            elif abs(sentiment) > 0.5:
                return DataPriority.HIGH
        
        # Price/volume anomalies
        if packet.data_type == 'trade':
            if self._is_price_anomaly(packet):
                return DataPriority.HIGH
            elif self._is_volume_anomaly(packet):
                return DataPriority.HIGH
        
        # During market hours, prioritize real-time data
        if self.market_state['market_hours']:
            if packet.data_type in ['trades', 'quotes']:
                return DataPriority.HIGH
            return DataPriority.MEDIUM
        
        # Default priorities
        default_priorities = {
            'trades': DataPriority.HIGH,
            'quotes': DataPriority.HIGH,
            'bars': DataPriority.MEDIUM,
            'news': DataPriority.MEDIUM,
            'sentiment': DataPriority.LOW,
            'fundamentals': DataPriority.LOW
        }
        
        return default_priorities.get(packet.data_type, DataPriority.LOW)
    
    def _is_price_anomaly(self, packet: DataPacket) -> bool:
        """Check if price movement is anomalous."""
        if 'price_change_percent' in packet.data:
            change = abs(packet.data['price_change_percent'])
            threshold = float(os.getenv('PRICE_CHANGE_URGENT_THRESHOLD', '2.0'))
            return change > threshold
        return False
    
    def _is_volume_anomaly(self, packet: DataPacket) -> bool:
        """Check if volume is anomalous."""
        if 'volume_ratio' in packet.data:
            ratio = packet.data['volume_ratio']
            threshold = float(os.getenv('VOLUME_SURGE_THRESHOLD', '3.0'))
            return ratio > threshold
        return False
    
    async def _route_to_strategy(self, strategy_name: str, packet: DataPacket):
        """Route data to specific strategy processor."""
        # Store in Redis for strategy consumption
        if self.redis:
            key = f"strategy_data:{strategy_name}:{packet.symbol}"
            
            # Add to stream for processing
            await self.redis.xadd(
                key,
                {
                    'data_type': packet.data_type,
                    'data': json.dumps(packet.data),
                    'priority': packet.priority.value,
                    'timestamp': packet.timestamp.isoformat()
                },
                maxlen=1000  # Keep last 1000 data points
            )
            
            # Set expiry
            await self.redis.expire(key, 3600)  # 1 hour TTL
        
        # Update routing stats
        self.routing_stats[strategy_name]['routed'] += 1
        
        # Process if handler exists
        if strategy_name in self.strategy_processors:
            processor = self.strategy_processors[strategy_name]
            asyncio.create_task(processor(packet))
    
    async def process_priority_queues(self):
        """Process data from priority queues."""
        logger.info("Started priority queue processor")
        
        while True:
            try:
                # Process queues in priority order
                for priority in sorted(DataPriority, key=lambda p: p.value):
                    queue = self.priority_queues[priority]
                    
                    # Process batch based on priority
                    batch_size = self._get_batch_size(priority)
                    
                    for _ in range(min(batch_size, queue.qsize())):
                        try:
                            packet = await asyncio.wait_for(
                                queue.get(),
                                timeout=0.1
                            )
                            
                            # Process packet based on active strategies
                            await self._process_packet(packet)
                            
                        except asyncio.TimeoutError:
                            break
                        except Exception as e:
                            logger.error(f"Error processing packet: {e}")
                
                # Small delay between processing cycles
                await asyncio.sleep(0.01)  # 10ms
                
            except Exception as e:
                logger.error(f"Error in priority queue processor: {e}")
                await asyncio.sleep(1)
    
    def _get_batch_size(self, priority: DataPriority) -> int:
        """Get batch size based on priority."""
        batch_sizes = {
            DataPriority.CRITICAL: 100,
            DataPriority.HIGH: 50,
            DataPriority.MEDIUM: 20,
            DataPriority.LOW: 10,
            DataPriority.ARCHIVE: 5
        }
        return batch_sizes.get(priority, 10)
    
    async def _process_packet(self, packet: DataPacket):
        """Process a data packet and potentially generate signals."""
        # Analyze packet for each active strategy
        for strategy_name in self.active_strategies:
            requirements = self.strategies.get(strategy_name)
            
            if not requirements or not requirements.enabled:
                continue
            
            # Check if we have enough data for analysis
            if await self._has_sufficient_data(strategy_name, packet.symbol):
                signal = await self._analyze_for_strategy(
                    strategy_name,
                    packet,
                    requirements
                )
                
                if signal:
                    await self._emit_signal(signal)
                    self.routing_stats[strategy_name]['signals'] += 1
            
            self.routing_stats[strategy_name]['processed'] += 1
    
    async def _has_sufficient_data(self, strategy: str, symbol: str) -> bool:
        """Check if we have sufficient data for strategy analysis."""
        if not self.redis:
            return True
        
        requirements = self.strategies[strategy]
        key = f"strategy_data:{strategy}:{symbol}"
        
        # Check data recency
        data_count = await self.redis.xlen(key)
        min_data_points = requirements.lookback_period // requirements.min_data_frequency
        
        return data_count >= min(min_data_points, 10)  # At least 10 data points
    
    async def _analyze_for_strategy(
        self,
        strategy_name: str,
        packet: DataPacket,
        requirements: StrategyRequirements
    ) -> Optional[TradingSignal]:
        """Analyze data packet for specific strategy and generate signals."""
        
        # Strategy-specific analysis
        if strategy_name == 'scalping':
            return await self._analyze_scalping(packet, requirements)
        elif strategy_name == 'day_trading':
            return await self._analyze_day_trading(packet, requirements)
        elif strategy_name == 'swing_trading':
            return await self._analyze_swing_trading(packet, requirements)
        elif strategy_name == 'position_trading':
            return await self._analyze_position_trading(packet, requirements)
        
        return None
    
    async def _analyze_scalping(
        self,
        packet: DataPacket,
        requirements: StrategyRequirements
    ) -> Optional[TradingSignal]:
        """Analyze for scalping opportunities."""
        if packet.data_type != 'quotes':
            return None
        
        # Check bid-ask spread
        if 'bid' in packet.data and 'ask' in packet.data:
            bid = packet.data['bid']
            ask = packet.data['ask']
            spread = ask - bid
            mid_price = (ask + bid) / 2
            
            # Tight spread with momentum
            if spread / mid_price < 0.001:  # Less than 0.1% spread
                # Check for momentum (would need more historical data)
                confidence = 0.7  # Simplified confidence
                
                return TradingSignal(
                    strategy='scalping',
                    symbol=packet.symbol,
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    entry_price=ask,
                    stop_loss=bid - (spread * 2),
                    take_profit=ask + (spread * 3),
                    position_size=requirements.max_position_size * confidence,
                    timestamp=datetime.utcnow(),
                    metadata={'spread': spread, 'mid_price': mid_price}
                )
        
        return None
    
    async def _analyze_day_trading(
        self,
        packet: DataPacket,
        requirements: StrategyRequirements
    ) -> Optional[TradingSignal]:
        """Analyze for day trading opportunities."""
        # Simplified day trading signal generation
        if packet.data_type == 'bars':
            # Check for breakout patterns
            if 'close' in packet.data and 'volume' in packet.data:
                # Would implement technical indicators here
                # For now, simple momentum check
                if packet.data.get('price_change_percent', 0) > 1.0:
                    if packet.data.get('volume_ratio', 1) > 2.0:
                        return TradingSignal(
                            strategy='day_trading',
                            symbol=packet.symbol,
                            signal_type=SignalType.BUY,
                            confidence=0.65,
                            entry_price=packet.data['close'],
                            stop_loss=packet.data['close'] * 0.98,
                            take_profit=packet.data['close'] * 1.03,
                            position_size=requirements.max_position_size * 0.5,
                            timestamp=datetime.utcnow(),
                            metadata={'pattern': 'momentum_breakout'}
                        )
        
        return None
    
    async def _analyze_swing_trading(
        self,
        packet: DataPacket,
        requirements: StrategyRequirements
    ) -> Optional[TradingSignal]:
        """Analyze for swing trading opportunities."""
        # Simplified swing trading analysis
        if packet.data_type == 'news' and 'sentiment_score' in packet.data:
            sentiment = packet.data['sentiment_score']
            
            # Strong sentiment signals
            if sentiment > 0.7:
                return TradingSignal(
                    strategy='swing_trading',
                    symbol=packet.symbol,
                    signal_type=SignalType.BUY,
                    confidence=sentiment,
                    entry_price=None,  # Market order
                    stop_loss=None,    # Will be set based on ATR
                    take_profit=None,  # Will be set based on target
                    position_size=requirements.max_position_size * sentiment * 0.5,
                    timestamp=datetime.utcnow(),
                    metadata={'sentiment': sentiment, 'source': packet.source}
                )
            elif sentiment < -0.7:
                return TradingSignal(
                    strategy='swing_trading',
                    symbol=packet.symbol,
                    signal_type=SignalType.SELL,
                    confidence=abs(sentiment),
                    entry_price=None,
                    stop_loss=None,
                    take_profit=None,
                    position_size=requirements.max_position_size * abs(sentiment) * 0.5,
                    timestamp=datetime.utcnow(),
                    metadata={'sentiment': sentiment, 'source': packet.source}
                )
        
        return None
    
    async def _analyze_position_trading(
        self,
        packet: DataPacket,
        requirements: StrategyRequirements
    ) -> Optional[TradingSignal]:
        """Analyze for position trading opportunities."""
        # Position trading focuses on longer-term trends
        # Would implement trend analysis, support/resistance, etc.
        return None
    
    async def _emit_signal(self, signal: TradingSignal):
        """Emit trading signal for execution."""
        logger.info(f"Signal generated: {signal.strategy} - {signal.symbol} - {signal.signal_type.value}")
        
        # Store signal in Redis
        if self.redis:
            await self.redis.xadd(
                f"signals:{signal.strategy}",
                {
                    'symbol': signal.symbol,
                    'signal_type': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'entry_price': signal.entry_price or 0,
                    'stop_loss': signal.stop_loss or 0,
                    'take_profit': signal.take_profit or 0,
                    'position_size': signal.position_size or 0,
                    'timestamp': signal.timestamp.isoformat(),
                    'metadata': json.dumps(signal.metadata or {})
                },
                maxlen=1000
            )
            
            # Publish for real-time subscribers
            await self.redis.publish(
                'trading_signals',
                json.dumps({
                    'strategy': signal.strategy,
                    'symbol': signal.symbol,
                    'signal': signal.signal_type.value,
                    'confidence': signal.confidence
                })
            )
        
        # Update metrics
        strategy_signals_counter.labels(
            strategy=signal.strategy,
            signal_type=signal.signal_type.value
        ).inc()
    
    def activate_strategy(self, strategy_name: str):
        """Activate a trading strategy."""
        if strategy_name in self.strategies:
            self.active_strategies.add(strategy_name)
            self.strategies[strategy_name].enabled = True
            active_strategies_gauge.set(len(self.active_strategies))
            logger.info(f"Activated strategy: {strategy_name}")
    
    def deactivate_strategy(self, strategy_name: str):
        """Deactivate a trading strategy."""
        if strategy_name in self.active_strategies:
            self.active_strategies.remove(strategy_name)
            self.strategies[strategy_name].enabled = False
            active_strategies_gauge.set(len(self.active_strategies))
            logger.info(f"Deactivated strategy: {strategy_name}")
    
    def update_market_state(self, state: Dict[str, Any]):
        """Update market state for routing decisions."""
        self.market_state.update(state)
        logger.info(f"Market state updated: {self.market_state}")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            'active_strategies': list(self.active_strategies),
            'market_state': self.market_state,
            'routing_stats': dict(self.routing_stats),
            'queue_sizes': {
                priority.name: self.priority_queues[priority].qsize()
                for priority in DataPriority
            }
        }


# Required import for JSON serialization
import json


# Example usage
async def main():
    """Example usage of strategy data router."""
    router = StrategyDataRouter()
    
    # Activate strategies based on market conditions
    router.activate_strategy('day_trading')
    router.activate_strategy('swing_trading')
    
    # Update market state
    router.update_market_state({
        'market_hours': True,
        'volatility': 'high',
        'trend': 'bullish'
    })
    
    # Start processing queues
    asyncio.create_task(router.process_priority_queues())
    
    # Route some sample data
    packet = DataPacket(
        data_type='quotes',
        symbol='AAPL',
        data={'bid': 150.00, 'ask': 150.02, 'volume': 1000000},
        timestamp=datetime.utcnow(),
        priority=DataPriority.HIGH,
        source='polygon'
    )
    
    await router.route_data(packet)
    
    # Run for a while
    await asyncio.sleep(10)
    
    # Get stats
    stats = router.get_routing_stats()
    logger.info(f"Routing stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())