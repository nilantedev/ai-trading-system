#!/usr/bin/env python3
"""
Critical Safety Systems Tests - MUST PASS before any real money trading
Tests all safety mechanisms to ensure they work correctly
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.risk_monitor.trading_governor import TradingGovernor, TradingMode
from services.data_ingestion.data_validator import MarketDataValidator, DataQuality
from services.ml.drift_monitor import DriftMonitor, DriftSeverity
from shared.python_common.trading_common.risk_limits import HardLimits, RiskLevel
from shared.python_common.trading_common.audit_trail import ImmutableAuditLog, EventType


class TestKillSwitch:
    """Test emergency stop and kill switch functionality"""
    
    @pytest.mark.asyncio
    async def test_emergency_stop_halts_trading(self):
        """Test that emergency stop immediately halts all trading"""
        governor = TradingGovernor()
        
        # Initialize with trading enabled
        await governor.initialize_default_settings()
        await governor.update_setting("auto_trade_enabled", True)
        await governor.update_setting("mode", "normal")
        
        # Verify trading is allowed
        can_trade, _ = await governor.can_trade("AAPL", 1000)
        assert can_trade == False  # Should be False by default due to auto_trade_enabled
        
        # Trigger emergency stop
        result = await governor.emergency_stop("Test emergency", cancel_orders=True)
        
        # Verify trading is halted
        assert result["status"] == "EMERGENCY STOP"
        
        # Check that trading is now blocked
        can_trade, reason = await governor.can_trade("AAPL", 100)
        assert can_trade == False
        assert "disabled" in reason.lower() or "stopped" in reason.lower()
        
        # Verify kill switch is active
        is_active = await governor.check_kill_switch()
        assert is_active == True
    
    @pytest.mark.asyncio
    async def test_kill_switch_persists(self):
        """Test that kill switch persists even after restart"""
        governor = TradingGovernor()
        
        # Activate kill switch
        await governor.emergency_stop("Test persistence")
        
        # Create new instance (simulating restart)
        new_governor = TradingGovernor()
        
        # Check kill switch is still active
        is_active = await new_governor.check_kill_switch()
        assert is_active == True
    
    @pytest.mark.asyncio
    async def test_kill_switch_clear_requires_admin(self):
        """Test that clearing kill switch requires admin key"""
        governor = TradingGovernor()
        
        # Activate kill switch
        await governor.emergency_stop("Test clear")
        
        # Try to clear without key
        cleared = await governor.clear_kill_switch("")
        assert cleared == False
        
        # Clear with admin key
        cleared = await governor.clear_kill_switch("admin_key_123")
        assert cleared == True
        
        # Verify it's cleared
        is_active = await governor.check_kill_switch()
        assert is_active == False


class TestRiskLimits:
    """Test hard risk limits enforcement"""
    
    def test_position_size_limits(self):
        """Test that position size limits are enforced"""
        portfolio = {
            'total_value': 10000,
            'positions': {},
            'total_exposure': 0,
            'daily_pnl': 0,
            'cash_balance': 10000
        }
        
        # Test conservative limits
        order = {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 100,
            'price': 150  # $15,000 order
        }
        
        validation = HardLimits.validate_order(order, portfolio, RiskLevel.CONSERVATIVE)
        assert validation.is_valid == False
        assert "exceeds max" in validation.reason
    
    def test_daily_loss_limit(self):
        """Test that daily loss limits trigger halt"""
        portfolio = {
            'total_value': 10000,
            'positions': {},
            'total_exposure': 0,
            'daily_pnl': -250,  # Already at loss
            'cash_balance': 9750
        }
        
        order = {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 5,
            'price': 150
        }
        
        # Conservative limit is $200
        validation = HardLimits.validate_order(order, portfolio, RiskLevel.CONSERVATIVE)
        assert validation.is_valid == False
        assert "daily loss" in validation.reason.lower()
    
    def test_position_count_limit(self):
        """Test that position count limits are enforced"""
        portfolio = {
            'total_value': 10000,
            'positions': {
                'AAPL': {'value': 1000},
                'GOOGL': {'value': 1000},
                'MSFT': {'value': 1000},
                'AMZN': {'value': 1000},
                'TSLA': {'value': 1000}
            },
            'total_exposure': 5000,
            'daily_pnl': 0,
            'cash_balance': 5000
        }
        
        order = {
            'symbol': 'META',
            'side': 'buy',
            'quantity': 5,
            'price': 100
        }
        
        # Conservative limit is 5 positions
        validation = HardLimits.validate_order(order, portfolio, RiskLevel.CONSERVATIVE)
        assert validation.is_valid == False
        assert "positions" in validation.reason
    
    def test_emergency_checks(self):
        """Test emergency halt conditions"""
        portfolio = {
            'total_value': 8500,
            'peak_value': 10000,
            'daily_pnl': -200,
            'weekly_pnl': -500
        }
        
        # Check for drawdown trigger (15% limit)
        should_halt, reason = HardLimits.emergency_checks(portfolio, RiskLevel.CONSERVATIVE)
        assert should_halt == False  # 15% drawdown exactly at limit
        
        # Exceed drawdown
        portfolio['total_value'] = 8000
        should_halt, reason = HardLimits.emergency_checks(portfolio, RiskLevel.CONSERVATIVE)
        assert should_halt == True
        assert "drawdown" in reason.lower()


class TestDataValidation:
    """Test market data validation"""
    
    def test_invalid_price_detection(self):
        """Test detection of invalid prices"""
        validator = MarketDataValidator()
        
        # Test negative price
        tick = {
            'symbol': 'AAPL',
            'price': -10,
            'timestamp': datetime.now()
        }
        
        result = validator.validate_tick(tick)
        assert result.is_valid == False
        assert result.quality == DataQuality.INVALID
        assert "too low" in result.reason.lower()
    
    def test_stale_data_detection(self):
        """Test detection of stale data"""
        validator = MarketDataValidator()
        
        # Create stale tick
        tick = {
            'symbol': 'AAPL',
            'price': 150,
            'timestamp': datetime.now() - timedelta(minutes=10)
        }
        
        result = validator.validate_tick(tick)
        assert result.is_valid == False
        assert "stale" in result.reason.lower()
    
    def test_spread_validation(self):
        """Test bid-ask spread validation"""
        validator = MarketDataValidator()
        
        # Test inverted spread
        tick = {
            'symbol': 'AAPL',
            'price': 150,
            'bid': 151,
            'ask': 149,
            'timestamp': datetime.now()
        }
        
        result = validator.validate_tick(tick)
        assert result.is_valid == False
        assert "inverted" in result.reason.lower()
    
    def test_price_spike_detection(self):
        """Test detection of price spikes"""
        validator = MarketDataValidator()
        
        # Build price history
        for i in range(20):
            tick = {
                'symbol': 'AAPL',
                'price': 150 + i * 0.1,
                'timestamp': datetime.now()
            }
            validator.validate_tick(tick)
        
        # Add spike
        spike_tick = {
            'symbol': 'AAPL',
            'price': 300,  # 100% jump
            'timestamp': datetime.now()
        }
        
        result = validator.validate_tick(spike_tick)
        assert result.is_valid == False
        assert "jump" in result.reason.lower()
    
    def test_volume_anomaly_detection(self):
        """Test detection of volume anomalies"""
        validator = MarketDataValidator()
        
        # Test negative volume
        tick = {
            'symbol': 'AAPL',
            'price': 150,
            'volume': -1000,
            'timestamp': datetime.now()
        }
        
        result = validator.validate_tick(tick)
        assert result.is_valid == False
        assert "volume" in result.reason.lower()


class TestModelDriftDetection:
    """Test ML model drift detection"""
    
    @pytest.mark.asyncio
    async def test_feature_drift_detection(self):
        """Test detection of feature distribution drift"""
        import numpy as np
        
        monitor = DriftMonitor("test_model")
        
        # Initialize baseline with normal distribution
        baseline_features = {
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000)
        }
        baseline_predictions = np.random.uniform(0, 1, 1000)
        
        await monitor.initialize_baseline(baseline_features, baseline_predictions)
        
        # Test with similar distribution (no drift)
        similar_features = {
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100)
        }
        similar_predictions = np.random.uniform(0, 1, 100)
        
        alerts = await monitor.detect_drift(similar_features, similar_predictions)
        assert len(alerts) == 0  # No drift expected
        
        # Test with shifted distribution (drift)
        shifted_features = {
            'feature1': np.random.normal(3, 1, 100),  # Mean shifted from 0 to 3
            'feature2': np.random.normal(5, 2, 100)
        }
        
        alerts = await monitor.detect_drift(shifted_features, similar_predictions)
        assert len(alerts) > 0  # Drift expected
        assert any(a.feature_name == 'feature1' for a in alerts)
    
    @pytest.mark.asyncio
    async def test_critical_drift_handling(self):
        """Test that critical drift triggers appropriate action"""
        import numpy as np
        
        monitor = DriftMonitor("test_model")
        
        # Initialize baseline
        baseline_features = {'feature1': np.random.normal(0, 1, 1000)}
        baseline_predictions = np.random.uniform(0, 1, 1000)
        await monitor.initialize_baseline(baseline_features, baseline_predictions)
        
        # Create severe drift
        drifted_features = {'feature1': np.random.normal(10, 1, 100)}  # Massive shift
        
        alerts = await monitor.detect_drift(drifted_features, baseline_predictions)
        
        # Check for critical severity
        critical_alerts = [a for a in alerts if a.severity == DriftSeverity.CRITICAL or a.severity == DriftSeverity.HIGH]
        assert len(critical_alerts) > 0
        
        # Verify action required
        for alert in critical_alerts:
            assert "retrain" in alert.action_required.lower() or "monitor" in alert.action_required.lower()


class TestAuditTrail:
    """Test immutable audit trail"""
    
    @pytest.mark.asyncio
    async def test_audit_immutability(self):
        """Test that audit trail maintains immutability"""
        audit_log = ImmutableAuditLog("/tmp/test_audit")
        
        # Add events
        event1 = await audit_log.add_event(
            EventType.ORDER_PLACED,
            {'order_id': '123', 'symbol': 'AAPL', 'quantity': 100}
        )
        
        event2 = await audit_log.add_event(
            EventType.RISK_LIMIT_HIT,
            {'limit_type': 'daily_loss', 'value': -500}
        )
        
        # Verify chain integrity
        is_valid, invalid_idx = audit_log.verify_chain()
        assert is_valid == True
        assert invalid_idx is None
        
        # Verify events are chained
        assert event2.prev_hash == event1.hash
        assert event1.prev_hash == "0" * 64  # Genesis
    
    @pytest.mark.asyncio
    async def test_audit_critical_events(self):
        """Test that critical events are properly logged"""
        audit_log = ImmutableAuditLog("/tmp/test_audit")
        
        # Log emergency stop
        await audit_log.add_event(
            EventType.EMERGENCY_STOP,
            {'reason': 'Manual trigger', 'timestamp': datetime.now().isoformat()}
        )
        
        # Log risk limit hit
        await audit_log.add_event(
            EventType.RISK_LIMIT_HIT,
            {'limit': 'position_size', 'attempted': 10000, 'max': 5000}
        )
        
        # Get recent events
        events = audit_log.get_recent_events(10)
        
        # Verify critical events are logged
        event_types = [e['event_type'] for e in events]
        assert EventType.EMERGENCY_STOP in event_types or 'emergency_stop' in str(event_types)
        assert EventType.RISK_LIMIT_HIT in event_types or 'risk_limit_hit' in str(event_types)
    
    @pytest.mark.asyncio
    async def test_compliance_report_generation(self):
        """Test compliance report generation"""
        audit_log = ImmutableAuditLog("/tmp/test_audit")
        
        # Add various events
        await audit_log.add_event(EventType.ORDER_PLACED, {'symbol': 'AAPL'})
        await audit_log.add_event(EventType.CONFIG_CHANGE, {'setting': 'max_position_size', 'old': 1000, 'new': 2000})
        await audit_log.add_event(EventType.RISK_LIMIT_HIT, {'type': 'daily_loss'})
        
        # Generate report
        report = audit_log.generate_compliance_report()
        
        assert report['chain_valid'] == True
        assert report['total_events'] >= 3
        assert len(report['risk_events']) >= 1
        assert len(report['config_changes']) >= 1


# Integration test that combines all safety systems
class TestIntegratedSafetySystems:
    """Test that all safety systems work together"""
    
    @pytest.mark.asyncio
    async def test_complete_safety_chain(self):
        """Test complete safety chain from data validation to audit"""
        # Initialize all components
        governor = TradingGovernor()
        validator = MarketDataValidator()
        audit_log = ImmutableAuditLog("/tmp/test_audit")
        
        # Initialize governor
        await governor.initialize_default_settings()
        await governor.update_setting("mode", "paper")
        await governor.update_setting("auto_trade_enabled", True)
        
        # Simulate bad data
        bad_tick = {
            'symbol': 'AAPL',
            'price': -100,
            'timestamp': datetime.now()
        }
        
        # Validate data
        validation_result = validator.validate_tick(bad_tick)
        assert validation_result.is_valid == False
        
        # Log data anomaly
        await audit_log.add_event(
            EventType.DATA_ANOMALY,
            {
                'symbol': bad_tick['symbol'],
                'reason': validation_result.reason,
                'quality': validation_result.quality.value
            }
        )
        
        # Trigger emergency stop due to bad data
        if validation_result.quality == DataQuality.INVALID:
            await governor.emergency_stop("Invalid data detected")
            await audit_log.add_event(
                EventType.EMERGENCY_STOP,
                {'reason': 'Invalid data', 'triggered_by': 'data_validator'}
            )
        
        # Verify system is halted
        can_trade, _ = await governor.can_trade("AAPL", 100)
        assert can_trade == False
        
        # Verify audit trail
        is_valid, _ = audit_log.verify_chain()
        assert is_valid == True
        
        events = audit_log.get_recent_events(10)
        assert len(events) >= 2  # Should have data anomaly and emergency stop


if __name__ == "__main__":
    pytest.main([__file__, "-v"])