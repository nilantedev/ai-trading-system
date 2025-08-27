#!/usr/bin/env python3
"""
Test suite for multi-frequency data ingestion system.
Validates that all components work together properly.
"""

import asyncio
import pytest
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

import sys
sys.path.append('/home/nilante/main-nilante-server/ai-trading-system')

from services.data_ingestion.data_velocity_manager import (
    DataVelocityManager, DataTier, TradingStrategy, UrgentEvent
)
from services.data_ingestion.realtime_market_stream import (
    RealTimeMarketStream, StreamType, MarketDataBuffer
)
from services.data_ingestion.strategy_data_router import (
    StrategyDataRouter, DataPacket, DataPriority, TradingSignal, SignalType
)
from services.data_ingestion.monitoring_dashboard import MonitoringDashboard


class TestDataVelocityManager:
    """Test data velocity manager functionality."""
    
    @pytest.fixture
    async def manager(self):
        """Create a test manager instance."""
        manager = DataVelocityManager()
        yield manager
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_tiered_polling_intervals(self, manager):
        """Test that different tiers have correct polling intervals."""
        # Load configuration
        manager._load_configuration()
        
        # Verify intervals are loaded correctly
        assert manager.intervals[DataTier.URGENT] == 30
        assert manager.intervals[DataTier.FAST] == 120
        assert manager.intervals[DataTier.STANDARD] == 900
        assert manager.intervals[DataTier.BACKGROUND] == 3600
    
    @pytest.mark.asyncio
    async def test_data_source_registration(self, manager):
        """Test registering data sources with different tiers."""
        # Mock callback
        callback = AsyncMock(return_value={'test': 'data'})
        
        # Register sources at different tiers
        manager.register_data_source('test_urgent', DataTier.URGENT, callback)
        manager.register_data_source('test_standard', DataTier.STANDARD, callback)
        
        # Verify registration
        assert 'test_urgent' in manager.data_sources
        assert manager.data_sources['test_urgent'].tier == DataTier.URGENT
        assert manager.data_sources['test_urgent'].interval_seconds == 30
        
        assert 'test_standard' in manager.data_sources
        assert manager.data_sources['test_standard'].tier == DataTier.STANDARD
        assert manager.data_sources['test_standard'].interval_seconds == 900
    
    @pytest.mark.asyncio
    async def test_urgent_event_detection(self, manager):
        """Test detection of urgent market events."""
        # Test price surge detection
        data = {'price_change_percent': 3.5, 'symbol': 'AAPL'}
        await manager._check_urgent_conditions('test', data)
        
        # Check if urgent event was created
        assert not manager.urgent_queue.empty()
        event = await manager.urgent_queue.get()
        assert event.event_type == 'price_surge'
        assert event.symbol == 'AAPL'
        assert event.requires_immediate_action
    
    @pytest.mark.asyncio
    async def test_api_quota_management(self, manager):
        """Test API quota tracking and enforcement."""
        # Set up test quota
        manager.api_limits['test_api'] = 100
        
        # Simulate API usage
        for _ in range(99):
            assert manager._check_api_quota('test_api')
            manager._update_api_usage('test_api')
        
        # Should still have one request left
        assert manager._check_api_quota('test_api')
        manager._update_api_usage('test_api')
        
        # Should now be at limit
        assert not manager._check_api_quota('test_api')
    
    @pytest.mark.asyncio
    async def test_dynamic_interval_adjustment(self, manager):
        """Test dynamic adjustment of polling intervals."""
        source = manager.data_sources.get('test_urgent', Mock(
            tier=DataTier.FAST,
            interval_seconds=120
        ))
        
        # During market hours, interval should be reduced
        with patch.object(manager, '_is_market_hours', return_value=True):
            interval = await manager._get_dynamic_interval(source)
            assert interval == 60  # Half of base interval
        
        # Outside market hours, use base interval
        with patch.object(manager, '_is_market_hours', return_value=False):
            interval = await manager._get_dynamic_interval(source)
            assert interval == 120
    
    @pytest.mark.asyncio
    async def test_strategy_optimization(self, manager):
        """Test strategy-specific optimizations."""
        # Set day trading strategy
        manager.set_active_strategies([TradingStrategy.DAY_TRADING])
        
        # Register a fast-tier source
        callback = AsyncMock()
        manager.register_data_source('day_trade_data', DataTier.FAST, callback)
        
        # Priority should be boosted for day trading
        source = manager.data_sources['day_trade_data']
        assert source.priority > manager._get_priority_for_tier(DataTier.FAST)


class TestRealTimeMarketStream:
    """Test real-time market streaming functionality."""
    
    @pytest.fixture
    async def stream(self):
        """Create a test stream instance."""
        stream = RealTimeMarketStream()
        yield stream
        await stream.stop()
    
    @pytest.mark.asyncio
    async def test_market_data_buffer(self):
        """Test market data buffering."""
        buffer = MarketDataBuffer(max_size=100)
        
        # Add trades
        for i in range(10):
            buffer.add_trade('AAPL', {'price': 150 + i, 'volume': 1000})
        
        # Add quotes
        for i in range(5):
            buffer.add_quote('AAPL', {'bid': 149 + i, 'ask': 151 + i})
        
        # Test retrieval
        trades = buffer.get_latest_trades('AAPL', limit=5)
        assert len(trades) == 5
        
        quote = buffer.get_latest_quote('AAPL')
        assert quote is not None
        assert 'bid' in quote['data']
    
    @pytest.mark.asyncio
    async def test_websocket_message_processing(self, stream):
        """Test processing of WebSocket messages."""
        # Mock Redis client
        stream.redis = AsyncMock()
        stream.buffers['test'] = MarketDataBuffer()
        
        # Process trade
        await stream._process_trade('test', {
            'S': 'AAPL',
            'p': 150.50,
            'v': 1000
        })
        
        # Verify buffer updated
        assert len(stream.buffers['test'].trades) == 1
        
        # Verify Redis called
        stream.redis.xadd.assert_called()
    
    @pytest.mark.asyncio
    async def test_volume_surge_detection(self, stream):
        """Test detection of volume surges."""
        stream.redis = AsyncMock()
        stream.redis.hget.return_value = '100000'  # Average volume
        
        # Test volume surge (5x average)
        await stream._check_volume_surge('AAPL', 500000)
        
        # Should create alert
        stream.redis.lpush.assert_called()
        call_args = stream.redis.lpush.call_args[0]
        assert call_args[0] == 'alerts:volume'
        alert = json.loads(call_args[1])
        assert alert['type'] == 'volume_surge'
        assert alert['ratio'] == 5.0
    
    @pytest.mark.asyncio
    async def test_spread_monitoring(self, stream):
        """Test bid-ask spread monitoring."""
        stream.redis = AsyncMock()
        stream.buffers['test'] = MarketDataBuffer()
        
        # Wide spread scenario
        await stream._process_quote('test', {
            'S': 'AAPL',
            'bp': 150.00,
            'ap': 151.00  # $1 spread (0.66%)
        })
        
        # Should trigger wide spread alert
        stream.redis.lpush.assert_called()
        call_args = stream.redis.lpush.call_args[0]
        assert 'alerts:spreads' in call_args[0]


class TestStrategyDataRouter:
    """Test strategy-aware data routing."""
    
    @pytest.fixture
    async def router(self):
        """Create a test router instance."""
        router = StrategyDataRouter()
        return router
    
    @pytest.mark.asyncio
    async def test_priority_determination(self, router):
        """Test data priority determination."""
        # Urgent news packet
        packet = DataPacket(
            data_type='news',
            symbol='AAPL',
            data={'sentiment_score': 0.9},
            timestamp=datetime.utcnow(),
            priority=DataPriority.LOW,
            source='newsapi'
        )
        
        priority = await router._determine_priority(packet)
        assert priority == DataPriority.CRITICAL  # High sentiment = critical
        
        # Normal trade packet
        packet = DataPacket(
            data_type='trade',
            symbol='AAPL',
            data={'price': 150.00},
            timestamp=datetime.utcnow(),
            priority=DataPriority.LOW,
            source='polygon'
        )
        
        # Mock market hours
        router.market_state['market_hours'] = True
        priority = await router._determine_priority(packet)
        assert priority == DataPriority.HIGH  # Trades are high priority during market hours
    
    @pytest.mark.asyncio
    async def test_strategy_routing(self, router):
        """Test routing to specific strategies."""
        router.redis = AsyncMock()
        
        # Activate strategies
        router.activate_strategy('day_trading')
        router.activate_strategy('swing_trading')
        
        # Create packet
        packet = DataPacket(
            data_type='bars',
            symbol='AAPL',
            data={'close': 150.00, 'volume': 1000000},
            timestamp=datetime.utcnow(),
            priority=DataPriority.MEDIUM,
            source='polygon'
        )
        
        # Route packet
        await router.route_data(packet)
        
        # Should be routed to both strategies
        assert packet in router.priority_queues[DataPriority.MEDIUM]._queue
    
    @pytest.mark.asyncio
    async def test_signal_generation(self, router):
        """Test trading signal generation."""
        router.redis = AsyncMock()
        
        # Test scalping signal
        packet = DataPacket(
            data_type='quotes',
            symbol='AAPL',
            data={'bid': 150.00, 'ask': 150.05},  # Tight spread
            timestamp=datetime.utcnow(),
            priority=DataPriority.HIGH,
            source='polygon'
        )
        
        requirements = router.strategies['scalping']
        signal = await router._analyze_scalping(packet, requirements)
        
        assert signal is not None
        assert signal.strategy == 'scalping'
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_priority_queue_processing(self, router):
        """Test priority queue processing."""
        # Add packets to different priority queues
        critical_packet = DataPacket(
            data_type='news',
            symbol='AAPL',
            data={'urgent': True},
            timestamp=datetime.utcnow(),
            priority=DataPriority.CRITICAL,
            source='news'
        )
        
        low_packet = DataPacket(
            data_type='sentiment',
            symbol='AAPL',
            data={'score': 0.1},
            timestamp=datetime.utcnow(),
            priority=DataPriority.LOW,
            source='reddit'
        )
        
        await router.priority_queues[DataPriority.CRITICAL].put(critical_packet)
        await router.priority_queues[DataPriority.LOW].put(low_packet)
        
        # Critical should be processed with larger batch size
        critical_batch = router._get_batch_size(DataPriority.CRITICAL)
        low_batch = router._get_batch_size(DataPriority.LOW)
        
        assert critical_batch > low_batch
        assert critical_batch == 100
        assert low_batch == 10


class TestMonitoringDashboard:
    """Test monitoring dashboard functionality."""
    
    @pytest.fixture
    async def dashboard(self):
        """Create a test dashboard instance."""
        dashboard = MonitoringDashboard()
        return dashboard
    
    @pytest.mark.asyncio
    async def test_ingestion_rate_tracking(self, dashboard):
        """Test tracking of data ingestion rates."""
        # Track multiple data points
        for i in range(10):
            await dashboard.track_data_ingestion(
                'polygon',
                'trades',
                count=100,
                latency_ms=5.0 + i
            )
            await asyncio.sleep(0.1)
        
        # Get stats
        stats = await dashboard.get_current_stats()
        assert 'polygon:trades' in stats['ingestion_rates']
        assert stats['ingestion_rates']['polygon:trades'] > 0
    
    @pytest.mark.asyncio
    async def test_alert_creation(self, dashboard):
        """Test alert creation and storage."""
        # Create different severity alerts
        await dashboard.create_alert('warning', 'latency', 'High latency detected')
        await dashboard.create_alert('critical', 'api_health', 'API is down')
        
        # Check alerts were stored
        assert len(dashboard.alert_buffer) == 2
        assert dashboard.alert_buffer[0]['severity'] == 'warning'
        assert dashboard.alert_buffer[1]['severity'] == 'critical'
    
    @pytest.mark.asyncio
    async def test_health_checks(self, dashboard):
        """Test system health checks."""
        # Set up healthy state
        dashboard.api_health = {
            'polygon': {'healthy': True},
            'newsapi': {'healthy': True}
        }
        dashboard.last_data_times = {
            'polygon:trades': datetime.utcnow()
        }
        
        health = await dashboard.get_system_health()
        assert health['healthy']
        assert health['checks']['apis_healthy']
        assert health['checks']['data_fresh']
    
    @pytest.mark.asyncio
    async def test_stale_data_detection(self, dashboard):
        """Test detection of stale data sources."""
        # Set old data time
        dashboard.last_data_times = {
            'polygon:trades': datetime.utcnow() - timedelta(minutes=10)
        }
        dashboard.thresholds['stale_data_seconds'] = 300  # 5 minutes
        
        # Check for stale data
        await dashboard.check_stale_data()
        
        # Should create alert
        assert len(dashboard.alert_buffer) > 0
        assert 'stale_data' in dashboard.alert_buffer[-1]['category']


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_data_flow(self):
        """Test complete data flow from ingestion to signal generation."""
        # Initialize components
        velocity_manager = DataVelocityManager()
        stream = RealTimeMarketStream()
        router = StrategyDataRouter()
        dashboard = MonitoringDashboard()
        
        # Activate day trading strategy
        velocity_manager.set_active_strategies([TradingStrategy.DAY_TRADING])
        router.activate_strategy('day_trading')
        
        # Register data source
        async def fetch_market_data():
            return {
                'symbol': 'AAPL',
                'price': 150.00,
                'volume': 1000000,
                'price_change_percent': 2.5
            }
        
        velocity_manager.register_data_source(
            'test_source',
            DataTier.URGENT,
            fetch_market_data
        )
        
        # Create data packet
        packet = DataPacket(
            data_type='bars',
            symbol='AAPL',
            data={
                'close': 150.00,
                'volume': 2000000,
                'price_change_percent': 2.5,
                'volume_ratio': 3.5
            },
            timestamp=datetime.utcnow(),
            priority=DataPriority.HIGH,
            source='test'
        )
        
        # Route through system
        await router.route_data(packet)
        
        # Track in monitoring
        await dashboard.track_data_ingestion(
            'test_source',
            'bars',
            count=1,
            latency_ms=10.5
        )
        
        # Verify complete flow
        assert 'day_trading' in router.active_strategies
        assert not velocity_manager.urgent_queue.empty()
        assert packet in router.priority_queues[DataPriority.HIGH]._queue
        
        # Clean up
        await velocity_manager.stop()
        await stream.stop()
    
    @pytest.mark.asyncio
    async def test_multi_strategy_support(self):
        """Test system supports multiple strategies simultaneously."""
        router = StrategyDataRouter()
        
        # Activate multiple strategies
        router.activate_strategy('scalping')
        router.activate_strategy('day_trading')
        router.activate_strategy('swing_trading')
        
        # Create packets for different strategies
        scalping_packet = DataPacket(
            data_type='quotes',
            symbol='AAPL',
            data={'bid': 150.00, 'ask': 150.02},
            timestamp=datetime.utcnow(),
            priority=DataPriority.CRITICAL,
            source='polygon'
        )
        
        swing_packet = DataPacket(
            data_type='news',
            symbol='AAPL',
            data={'sentiment_score': 0.8},
            timestamp=datetime.utcnow(),
            priority=DataPriority.HIGH,
            source='newsapi'
        )
        
        # Route packets
        await router.route_data(scalping_packet)
        await router.route_data(swing_packet)
        
        # Verify routing stats
        stats = router.get_routing_stats()
        assert len(stats['active_strategies']) == 3
        assert router.routing_stats['scalping']['routed'] > 0
        assert router.routing_stats['swing_trading']['routed'] > 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v'])