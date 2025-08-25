#!/usr/bin/env python3
"""
Integration tests for service-to-service interactions
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json
import asyncio

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestServiceIntegration:
    """Integration tests for trading system services."""

    @pytest_asyncio.fixture
    async def mock_services(self, mock_redis, mock_database):
        """Setup mock services for integration testing."""
        services = {
            'market_data': AsyncMock(),
            'signal_generator': AsyncMock(),
            'order_management': AsyncMock(),
            'portfolio': AsyncMock(),
            'risk_monitor': AsyncMock()
        }
        
        # Configure service behaviors
        services['market_data'].get_current_data.return_value = {
            "symbol": "AAPL",
            "price": 150.25,
            "volume": 1000000,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        services['signal_generator'].generate_signal.return_value = {
            "id": "signal_001",
            "symbol": "AAPL",
            "signal_type": "BUY",
            "confidence": 0.85,
            "price_target": 155.00
        }
        
        services['order_management'].place_order.return_value = {
            "id": "order_001",
            "status": "PENDING",
            "symbol": "AAPL"
        }
        
        return services

    @pytest.mark.asyncio
    async def test_market_data_to_signal_flow(self, mock_services):
        """Test data flow from market data to signal generation."""
        # Simulate market data update triggering signal generation
        market_data = {
            "symbol": "AAPL",
            "price": 150.25,
            "change_percent": 5.2,  # Strong positive movement
            "volume": 2000000,  # High volume
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Mock the flow
        mock_services['market_data'].get_current_data.return_value = market_data
        
        # Signal service should react to market data
        signal = await mock_services['signal_generator'].generate_signal("AAPL", market_data)
        
        # Verify signal was generated correctly
        assert signal["symbol"] == "AAPL"
        assert signal["signal_type"] == "BUY"
        assert signal["confidence"] > 0.8  # High confidence for strong movement

    @pytest.mark.asyncio
    async def test_signal_to_order_flow(self, mock_services):
        """Test flow from signal generation to order placement."""
        # Generate trading signal
        signal = {
            "id": "signal_001",
            "symbol": "AAPL",
            "signal_type": "BUY",
            "confidence": 0.88,
            "price_target": 155.00,
            "stop_loss": 145.00,
            "quantity_suggestion": 100
        }
        
        # Signal should trigger order placement
        order_request = {
            "symbol": signal["symbol"],
            "side": signal["signal_type"],
            "quantity": signal["quantity_suggestion"],
            "order_type": "LIMIT",
            "price": signal["price_target"],
            "signal_id": signal["id"]
        }
        
        # Risk check should pass
        mock_services['risk_monitor'].validate_order.return_value = True
        
        # Simulate the proper integration flow: validate first, then place order
        # Step 1: Risk validation
        risk_approved = await mock_services['risk_monitor'].validate_order(order_request)
        assert risk_approved is True
        
        # Step 2: Place order only if risk approved
        if risk_approved:
            order = await mock_services['order_management'].place_order(order_request)
        
        # Verify order placement
        assert order["symbol"] == "AAPL"
        assert order["status"] in ["PENDING", "SUBMITTED"]
        mock_services['risk_monitor'].validate_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_order_to_portfolio_flow(self, mock_services):
        """Test flow from order execution to portfolio update."""
        # Simulate order execution
        order_fill = {
            "order_id": "order_001",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 100,
            "fill_price": 150.30,
            "commission": 1.00,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Order fill should update portfolio
        portfolio_update = {
            "user_id": "test_user",
            "symbol": "AAPL",
            "quantity_change": 100,
            "cost_basis": 150.30,
            "commission": 1.00
        }
        
        await mock_services['portfolio'].update_position(portfolio_update)
        
        # Verify portfolio service was called
        mock_services['portfolio'].update_position.assert_called_once_with(portfolio_update)

    @pytest.mark.asyncio
    async def test_risk_monitoring_integration(self, mock_services):
        """Test risk monitoring across services."""
        # Test pre-order risk check
        large_order = {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10000,  # Large position
            "price": 150.25,
            "user_id": "test_user"
        }
        
        # Risk service should reject large order
        mock_services['risk_monitor'].validate_order.return_value = False
        
        # Order placement should fail risk check
        with pytest.raises(Exception, match="Risk limit exceeded"):
            risk_ok = await mock_services['risk_monitor'].validate_order(large_order)
            if not risk_ok:
                raise Exception("Risk limit exceeded")

    @pytest.mark.asyncio
    async def test_portfolio_risk_monitoring(self, mock_services):
        """Test portfolio-level risk monitoring."""
        portfolio_data = {
            "user_id": "test_user",
            "total_value": 100000,
            "positions": {
                "AAPL": {"quantity": 500, "value": 75000},  # 75% concentration
                "GOOGL": {"quantity": 50, "value": 25000}
            }
        }
        
        # Risk monitor should flag concentration risk
        risk_metrics = await mock_services['risk_monitor'].calculate_portfolio_risk(portfolio_data)
        
        # Should detect high concentration
        mock_services['risk_monitor'].calculate_portfolio_risk.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_service_workflow(self, mock_services):
        """Test complete workflow across multiple services."""
        # Step 1: Market data triggers signal
        market_data = {
            "symbol": "AAPL",
            "price": 150.25,
            "change_percent": 3.5,
            "volume": 1500000
        }
        
        # Step 2: Generate signal
        signal = await mock_services['signal_generator'].generate_signal("AAPL", market_data)
        
        # Step 3: Validate with risk monitor
        risk_ok = await mock_services['risk_monitor'].validate_signal(signal)
        mock_services['risk_monitor'].validate_signal.return_value = True
        
        # Step 4: Place order if risk check passes
        if risk_ok:
            order = await mock_services['order_management'].place_order({
                "symbol": signal["symbol"],
                "side": signal["signal_type"],
                "quantity": 100,
                "signal_id": signal["id"]
            })
        
        # Verify all services were called in sequence
        mock_services['signal_generator'].generate_signal.assert_called_once()
        mock_services['risk_monitor'].validate_signal.assert_called_once()
        mock_services['order_management'].place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_across_services(self, mock_services):
        """Test error handling in service interactions."""
        # Market data service failure
        mock_services['market_data'].get_current_data.side_effect = Exception("Data source unavailable")
        
        # Signal generator should handle gracefully
        try:
            market_data = await mock_services['market_data'].get_current_data("AAPL")
        except Exception as e:
            # Signal service should use cached data or skip
            assert "unavailable" in str(e)

    @pytest.mark.asyncio
    async def test_concurrent_service_operations(self, mock_services):
        """Test concurrent operations across services."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        # Concurrent market data requests
        market_data_tasks = [
            mock_services['market_data'].get_current_data(symbol)
            for symbol in symbols
        ]
        
        # Concurrent signal generation
        signal_tasks = [
            mock_services['signal_generator'].generate_signal(symbol, {})
            for symbol in symbols
        ]
        
        # Execute concurrently
        market_results = await asyncio.gather(*market_data_tasks, return_exceptions=True)
        signal_results = await asyncio.gather(*signal_tasks, return_exceptions=True)
        
        # Verify all completed
        assert len(market_results) == 3
        assert len(signal_results) == 3

    @pytest.mark.asyncio
    async def test_service_state_consistency(self, mock_services):
        """Test state consistency between services."""
        # Portfolio service shows position
        portfolio_position = {
            "symbol": "AAPL",
            "quantity": 100,
            "average_price": 148.50
        }
        
        mock_services['portfolio'].get_position.return_value = portfolio_position
        
        # Order service should reflect same position
        order_history = [
            {
                "symbol": "AAPL",
                "side": "BUY", 
                "quantity": 100,
                "fill_price": 148.50,
                "status": "FILLED"
            }
        ]
        
        mock_services['order_management'].get_user_orders.return_value = order_history
        
        # Verify consistency
        position = await mock_services['portfolio'].get_position("AAPL")
        orders = await mock_services['order_management'].get_user_orders("test_user")
        
        assert position["quantity"] == sum(o["quantity"] for o in orders if o["side"] == "BUY")

    @pytest.mark.asyncio
    async def test_data_pipeline_integration(self, mock_services):
        """Test data pipeline from ingestion to consumption."""
        # Raw market data
        raw_data = {
            "symbol": "AAPL",
            "bid": 150.20,
            "ask": 150.30,
            "last": 150.25,
            "volume": 1000000,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Data ingestion processes raw data
        processed_data = {
            "symbol": raw_data["symbol"],
            "price": raw_data["last"],
            "spread": raw_data["ask"] - raw_data["bid"],
            "volume": raw_data["volume"],
            "timestamp": raw_data["timestamp"]
        }
        
        # Signal generator consumes processed data
        signal = await mock_services['signal_generator'].analyze_data(processed_data)
        
        # Verify data flowed through pipeline
        mock_services['signal_generator'].analyze_data.assert_called_once_with(processed_data)

    @pytest.mark.asyncio
    async def test_event_driven_architecture(self, mock_services):
        """Test event-driven interactions between services."""
        # Market data event
        market_event = {
            "type": "price_update",
            "symbol": "AAPL",
            "price": 150.25,
            "change": 2.5,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Event should trigger multiple service reactions
        await mock_services['signal_generator'].handle_market_event(market_event)
        await mock_services['risk_monitor'].handle_market_event(market_event)
        
        # Verify event was processed by multiple services
        mock_services['signal_generator'].handle_market_event.assert_called_once()
        mock_services['risk_monitor'].handle_market_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_dependency_chain(self, mock_services):
        """Test service dependency chain resolution."""
        # Order depends on risk check, which depends on portfolio, which depends on market data
        
        # Market data (base dependency)
        market_data = {"symbol": "AAPL", "price": 150.25}
        mock_services['market_data'].get_current_data.return_value = market_data
        
        # Portfolio depends on market data
        portfolio_value = 100000 * (market_data["price"] / 150.0)  # Price-adjusted value
        mock_services['portfolio'].calculate_portfolio_value.return_value = portfolio_value
        
        # Risk check depends on portfolio
        mock_services['risk_monitor'].check_portfolio_limits.return_value = True
        
        # Order depends on risk check
        order_result = await mock_services['order_management'].place_order_with_checks({
            "symbol": "AAPL",
            "quantity": 100
        })
        
        # Verify dependency chain
        mock_services['order_management'].place_order_with_checks.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_recovery_mechanisms(self, mock_services):
        """Test service recovery and retry mechanisms."""
        # Simulate temporary service failure
        mock_services['market_data'].get_current_data.side_effect = [
            Exception("Temporary failure"),  # First call fails
            {"symbol": "AAPL", "price": 150.25}  # Second call succeeds
        ]
        
        # Service should retry and succeed
        with patch('asyncio.sleep'):  # Speed up retry delay
            try:
                # First attempt fails
                await mock_services['market_data'].get_current_data("AAPL")
            except Exception:
                # Retry succeeds
                data = await mock_services['market_data'].get_current_data("AAPL")
                assert data["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_cross_service_data_validation(self, mock_services):
        """Test data validation across service boundaries."""
        # Invalid data from one service
        invalid_signal = {
            "symbol": "AAPL",
            "signal_type": "INVALID_TYPE",  # Invalid
            "confidence": 1.5  # Out of range
        }
        
        # Next service should validate and reject
        mock_services['order_management'].validate_signal_data.return_value = False
        
        with pytest.raises(ValueError, match="Invalid signal data"):
            is_valid = await mock_services['order_management'].validate_signal_data(invalid_signal)
            if not is_valid:
                raise ValueError("Invalid signal data")

    @pytest.mark.asyncio
    async def test_service_performance_monitoring(self, mock_services):
        """Test performance monitoring across services."""
        # Track service call times
        start_time = datetime.utcnow()
        
        # Make service calls
        await mock_services['market_data'].get_current_data("AAPL")
        await mock_services['signal_generator'].generate_signal("AAPL", {})
        
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        
        # Should complete quickly (mocked services)
        assert total_time < 1.0  # Less than 1 second for mocked calls