"""
Critical Integration Tests for Trading Flows
Tests the most important trading operations end-to-end to ensure system safety.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
import json

# Test configuration
TEST_SYMBOL = "AAPL"
TEST_QUANTITY = 10
TEST_CAPITAL = 10000
MAX_POSITION_SIZE = 0.1  # 10% of capital


class TestCriticalTradingFlow:
    """Test complete trading flow from signal to execution."""
    
    @pytest.mark.asyncio
    async def test_complete_trade_lifecycle(self, trading_system):
        """Test full trade lifecycle: signal → validation → execution → monitoring → exit."""
        
        # 1. Generate trading signal
        signal = await trading_system.generate_signal(
            symbol=TEST_SYMBOL,
            timeframe="1h"
        )
        
        assert signal is not None
        assert signal['symbol'] == TEST_SYMBOL
        assert signal['confidence'] >= 0 and signal['confidence'] <= 1
        assert signal['direction'] in ['BUY', 'SELL', 'HOLD']
        
        # 2. Validate signal passes risk checks
        if signal['direction'] != 'HOLD':
            risk_check = await trading_system.validate_risk(
                symbol=signal['symbol'],
                direction=signal['direction'],
                quantity=TEST_QUANTITY,
                account_balance=TEST_CAPITAL
            )
            
            assert 'approved' in risk_check
            assert 'max_position_size' in risk_check
            assert 'current_exposure' in risk_check
            assert 'stop_loss' in risk_check
            assert 'take_profit' in risk_check
            
            if not risk_check['approved']:
                assert 'rejection_reason' in risk_check
                return  # Skip trade if risk check fails
        
        # 3. Place order
        if signal['direction'] == 'BUY':
            order = await trading_system.place_order(
                symbol=TEST_SYMBOL,
                side='buy',
                quantity=TEST_QUANTITY,
                order_type='market',
                stop_loss=risk_check['stop_loss'],
                take_profit=risk_check['take_profit']
            )
            
            assert order is not None
            assert order['status'] in ['pending', 'filled', 'partially_filled']
            assert order['symbol'] == TEST_SYMBOL
            assert order['quantity'] == TEST_QUANTITY
            assert 'order_id' in order
            
            # 4. Monitor order execution
            max_wait = 30  # seconds
            start_time = datetime.now()
            
            while (datetime.now() - start_time).seconds < max_wait:
                order_status = await trading_system.get_order_status(order['order_id'])
                
                assert order_status is not None
                assert 'status' in order_status
                
                if order_status['status'] in ['filled', 'cancelled', 'rejected']:
                    break
                
                await asyncio.sleep(1)
            
            # 5. Verify position
            if order_status['status'] == 'filled':
                position = await trading_system.get_position(TEST_SYMBOL)
                
                assert position is not None
                assert position['symbol'] == TEST_SYMBOL
                assert position['quantity'] == TEST_QUANTITY
                assert 'entry_price' in position
                assert 'current_price' in position
                assert 'unrealized_pnl' in position
                
                # 6. Test position monitoring
                await asyncio.sleep(5)  # Wait for price updates
                
                updated_position = await trading_system.get_position(TEST_SYMBOL)
                assert updated_position is not None
                
                # 7. Close position
                close_order = await trading_system.close_position(TEST_SYMBOL)
                
                assert close_order is not None
                assert close_order['side'] == 'sell'
                assert close_order['quantity'] == TEST_QUANTITY
                
                # 8. Verify position closed
                await asyncio.sleep(5)
                final_position = await trading_system.get_position(TEST_SYMBOL)
                assert final_position is None or final_position['quantity'] == 0
    
    @pytest.mark.asyncio
    async def test_risk_management_enforcement(self, trading_system):
        """Test that risk management rules are properly enforced."""
        
        # Test 1: Position size limit
        oversized_order = await trading_system.validate_risk(
            symbol=TEST_SYMBOL,
            direction='BUY',
            quantity=1000000,  # Huge quantity
            account_balance=TEST_CAPITAL
        )
        
        assert oversized_order['approved'] is False
        assert 'position_size' in oversized_order['rejection_reason'].lower()
        
        # Test 2: Daily loss limit
        # Simulate hitting daily loss limit
        await trading_system.set_daily_pnl(-500)  # Assume $500 loss
        
        loss_limit_order = await trading_system.validate_risk(
            symbol=TEST_SYMBOL,
            direction='BUY',
            quantity=TEST_QUANTITY,
            account_balance=TEST_CAPITAL
        )
        
        if trading_system.config.max_daily_loss and abs(-500) >= trading_system.config.max_daily_loss:
            assert loss_limit_order['approved'] is False
            assert 'daily_loss' in loss_limit_order['rejection_reason'].lower()
        
        # Test 3: Maximum positions
        # Try to open more positions than allowed
        max_positions = trading_system.config.max_positions or 10
        
        for i in range(max_positions + 1):
            result = await trading_system.validate_risk(
                symbol=f"TEST{i}",
                direction='BUY',
                quantity=1,
                account_balance=TEST_CAPITAL
            )
            
            if i >= max_positions:
                assert result['approved'] is False
                assert 'max_positions' in result['rejection_reason'].lower()
    
    @pytest.mark.asyncio
    async def test_stop_loss_execution(self, trading_system):
        """Test that stop loss orders are properly executed."""
        
        # Place order with stop loss
        entry_price = 100.0
        stop_loss_price = 95.0  # 5% stop loss
        
        order = await trading_system.place_order(
            symbol=TEST_SYMBOL,
            side='buy',
            quantity=TEST_QUANTITY,
            order_type='limit',
            limit_price=entry_price,
            stop_loss=stop_loss_price
        )
        
        assert order is not None
        assert 'stop_loss_order_id' in order
        
        # Simulate price drop below stop loss
        await trading_system.simulate_price_change(TEST_SYMBOL, 94.0)
        
        # Wait for stop loss to trigger
        await asyncio.sleep(5)
        
        # Verify position is closed
        position = await trading_system.get_position(TEST_SYMBOL)
        assert position is None or position['quantity'] == 0
        
        # Verify stop loss order was executed
        stop_loss_order = await trading_system.get_order_status(order['stop_loss_order_id'])
        assert stop_loss_order['status'] == 'filled'
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self, trading_system):
        """Test that circuit breaker stops trading when triggered."""
        
        # Simulate multiple failed trades
        for i in range(5):
            await trading_system.record_trade_result(
                symbol=f"TEST{i}",
                pnl=-100,
                success=False
            )
        
        # Check circuit breaker status
        circuit_status = await trading_system.get_circuit_breaker_status()
        assert circuit_status['is_open'] is True
        assert 'reason' in circuit_status
        
        # Try to place order with circuit breaker open
        order = await trading_system.place_order(
            symbol=TEST_SYMBOL,
            side='buy',
            quantity=TEST_QUANTITY,
            order_type='market'
        )
        
        assert order is None or order['status'] == 'rejected'
    
    @pytest.mark.asyncio
    async def test_order_validation(self, trading_system):
        """Test that orders are properly validated before execution."""
        
        # Test 1: Invalid symbol
        invalid_symbol_order = await trading_system.place_order(
            symbol="INVALID_SYMBOL_12345",
            side='buy',
            quantity=TEST_QUANTITY,
            order_type='market'
        )
        
        assert invalid_symbol_order is None or invalid_symbol_order['status'] == 'rejected'
        
        # Test 2: Invalid quantity (negative)
        with pytest.raises(ValueError):
            await trading_system.place_order(
                symbol=TEST_SYMBOL,
                side='buy',
                quantity=-10,
                order_type='market'
            )
        
        # Test 3: Invalid order type
        with pytest.raises(ValueError):
            await trading_system.place_order(
                symbol=TEST_SYMBOL,
                side='buy',
                quantity=TEST_QUANTITY,
                order_type='invalid_type'
            )
        
        # Test 4: Insufficient funds
        insufficient_funds_order = await trading_system.place_order(
            symbol=TEST_SYMBOL,
            side='buy',
            quantity=1000000,
            order_type='market'
        )
        
        assert insufficient_funds_order is None or insufficient_funds_order['status'] == 'rejected'


class TestBacktestingIntegrity:
    """Test backtesting system for accuracy and consistency."""
    
    @pytest.mark.asyncio
    async def test_backtest_reproducibility(self, backtesting_system):
        """Test that backtests produce consistent results."""
        
        # Run same backtest twice
        config = {
            'symbol': TEST_SYMBOL,
            'start_date': '2024-01-01',
            'end_date': '2024-01-31',
            'initial_capital': TEST_CAPITAL,
            'strategy': 'mean_reversion'
        }
        
        result1 = await backtesting_system.run_backtest(config)
        result2 = await backtesting_system.run_backtest(config)
        
        # Results should be identical
        assert result1['total_return'] == result2['total_return']
        assert result1['sharpe_ratio'] == result2['sharpe_ratio']
        assert result1['max_drawdown'] == result2['max_drawdown']
        assert result1['number_of_trades'] == result2['number_of_trades']
        assert result1['win_rate'] == result2['win_rate']
    
    @pytest.mark.asyncio
    async def test_backtest_metrics_accuracy(self, backtesting_system):
        """Test that backtest metrics are calculated correctly."""
        
        # Create simple test data
        trades = [
            {'entry': 100, 'exit': 105, 'quantity': 10},  # Profit: $50
            {'entry': 100, 'exit': 98, 'quantity': 10},   # Loss: -$20
            {'entry': 100, 'exit': 103, 'quantity': 10},  # Profit: $30
        ]
        
        result = await backtesting_system.calculate_metrics(
            trades=trades,
            initial_capital=TEST_CAPITAL
        )
        
        # Verify calculations
        assert result['total_pnl'] == 60  # 50 - 20 + 30
        assert result['win_rate'] == pytest.approx(0.667, rel=0.01)  # 2/3
        assert result['average_win'] == 40  # (50 + 30) / 2
        assert result['average_loss'] == 20
        assert result['profit_factor'] == 4.0  # 80 / 20
    
    @pytest.mark.asyncio
    async def test_no_look_ahead_bias(self, backtesting_system):
        """Test that backtesting doesn't have look-ahead bias."""
        
        # Create test data with future information
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h'),
            'price': np.random.randn(100).cumsum() + 100,
            'future_price': np.random.randn(100).cumsum() + 100  # This shouldn't be accessible
        })
        
        # Run backtest
        result = await backtesting_system.run_backtest_with_data(
            data=data,
            strategy='test_strategy'
        )
        
        # Verify no future data was used
        assert 'future_price' not in result['signals_used']
        assert result['look_ahead_bias_check'] == 'passed'


class TestDataIntegrityValidation:
    """Test data validation and quality checks."""
    
    @pytest.mark.asyncio
    async def test_market_data_validation(self, data_system):
        """Test that market data is properly validated."""
        
        # Fetch market data
        data = await data_system.get_market_data(
            symbol=TEST_SYMBOL,
            timeframe='1m',
            limit=100
        )
        
        assert data is not None
        assert len(data) > 0
        
        # Validate data structure
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in data.columns
        
        # Validate data quality
        assert not data['close'].isna().any()  # No missing prices
        assert (data['high'] >= data['low']).all()  # High >= Low
        assert (data['high'] >= data['close']).all()  # High >= Close
        assert (data['low'] <= data['close']).all()  # Low <= Close
        assert (data['volume'] >= 0).all()  # Non-negative volume
        
        # Check for duplicates
        assert not data['timestamp'].duplicated().any()
        
        # Check chronological order
        assert data['timestamp'].is_monotonic_increasing
    
    @pytest.mark.asyncio
    async def test_data_gap_detection(self, data_system):
        """Test that data gaps are properly detected and handled."""
        
        # Fetch data with potential gaps
        data = await data_system.get_market_data(
            symbol=TEST_SYMBOL,
            timeframe='1h',
            start='2024-01-01',
            end='2024-01-31'
        )
        
        # Check for gaps
        gaps = await data_system.detect_gaps(data)
        
        if gaps:
            for gap in gaps:
                assert 'start' in gap
                assert 'end' in gap
                assert 'duration' in gap
                assert gap['duration'] > 0
        
        # Verify gap handling
        filled_data = await data_system.fill_gaps(data)
        assert len(filled_data) >= len(data)


class TestMLModelIntegration:
    """Test ML model integration and predictions."""
    
    @pytest.mark.asyncio
    async def test_model_prediction_pipeline(self, ml_system):
        """Test complete ML prediction pipeline."""
        
        # 1. Prepare features
        features = await ml_system.prepare_features(
            symbol=TEST_SYMBOL,
            lookback_periods=20
        )
        
        assert features is not None
        assert len(features) > 0
        assert not features.isna().all().any()  # No columns with all NaN
        
        # 2. Get model prediction
        prediction = await ml_system.predict(
            features=features,
            model_id='latest'
        )
        
        assert prediction is not None
        assert 'direction' in prediction
        assert 'confidence' in prediction
        assert 'model_version' in prediction
        assert prediction['confidence'] >= 0 and prediction['confidence'] <= 1
        
        # 3. Validate prediction format
        assert prediction['direction'] in ['UP', 'DOWN', 'NEUTRAL']
        assert isinstance(prediction['confidence'], (int, float))
    
    @pytest.mark.asyncio
    async def test_model_fallback_mechanism(self, ml_system):
        """Test that system falls back gracefully when primary model fails."""
        
        # Simulate primary model failure
        await ml_system.disable_model('primary')
        
        # Should fallback to secondary model
        prediction = await ml_system.predict(
            symbol=TEST_SYMBOL,
            features={}
        )
        
        assert prediction is not None
        assert prediction['model_used'] == 'fallback'
        assert 'fallback_reason' in prediction


class TestSystemResilience:
    """Test system resilience and error recovery."""
    
    @pytest.mark.asyncio
    async def test_database_connection_recovery(self, trading_system):
        """Test system recovers from database connection loss."""
        
        # Simulate database disconnection
        await trading_system.simulate_db_disconnect()
        
        # System should queue operations
        order = await trading_system.place_order(
            symbol=TEST_SYMBOL,
            side='buy',
            quantity=TEST_QUANTITY,
            order_type='market'
        )
        
        assert order is not None
        assert order['status'] == 'queued'
        
        # Simulate reconnection
        await trading_system.simulate_db_reconnect()
        
        # Wait for queued operations to process
        await asyncio.sleep(5)
        
        # Verify order was processed
        order_status = await trading_system.get_order_status(order['order_id'])
        assert order_status['status'] != 'queued'
    
    @pytest.mark.asyncio
    async def test_api_rate_limit_handling(self, trading_system):
        """Test that system properly handles API rate limits."""
        
        # Send many requests quickly
        results = []
        for i in range(100):
            result = await trading_system.get_market_data(f"TEST{i}")
            results.append(result)
        
        # Should not all fail
        successful = sum(1 for r in results if r is not None)
        assert successful > 0
        
        # Check rate limit handling
        rate_limit_status = await trading_system.get_rate_limit_status()
        assert 'requests_remaining' in rate_limit_status
        assert 'reset_time' in rate_limit_status


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Test that system meets performance requirements."""
    
    @pytest.mark.asyncio
    async def test_order_execution_latency(self, trading_system):
        """Test that orders are executed within acceptable latency."""
        
        import time
        
        start_time = time.time()
        order = await trading_system.place_order(
            symbol=TEST_SYMBOL,
            side='buy',
            quantity=TEST_QUANTITY,
            order_type='market'
        )
        execution_time = time.time() - start_time
        
        # Should execute within 1 second
        assert execution_time < 1.0
        assert order is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_order_handling(self, trading_system):
        """Test system can handle concurrent orders."""
        
        # Place multiple orders concurrently
        orders = await asyncio.gather(*[
            trading_system.place_order(
                symbol=f"TEST{i}",
                side='buy',
                quantity=1,
                order_type='market'
            ) for i in range(10)
        ])
        
        # All should be processed
        assert all(order is not None for order in orders)
        assert len(set(order['order_id'] for order in orders)) == 10  # All unique IDs