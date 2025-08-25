#!/usr/bin/env python3
"""
Unit tests for Order Management System
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json
from decimal import Decimal

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.execution.order_management_system import OrderManagementSystem


class TestOrderManagementSystem:
    """Test cases for OrderManagementSystem class."""

    @pytest_asyncio.fixture
    async def oms(self, mock_redis, mock_database, test_settings):
        """Create OrderManagementSystem instance for testing."""
        oms = OrderManagementSystem()
        oms.redis_client = mock_redis
        oms.database = mock_database
        oms.settings = test_settings
        return oms

    @pytest.mark.asyncio
    async def test_place_order_success(self, oms, sample_order):
        """Test successful order placement."""
        # Mock successful order validation and placement
        with patch.object(oms, '_validate_order', return_value=True):
            with patch.object(oms, '_check_risk_limits', return_value=True):
                with patch.object(oms, '_submit_to_broker', return_value="broker_order_123"):
                    
                    order_request = {
                        "symbol": "AAPL",
                        "side": "BUY",
                        "order_type": "MARKET",
                        "quantity": 100,
                        "user_id": "test_user"
                    }
                    
                    result = await oms.place_order(order_request)
                    
                    assert result is not None
                    assert result["status"] == "PENDING"
                    assert result["symbol"] == "AAPL"
                    assert result["quantity"] == 100
                    assert "order_id" in result

    @pytest.mark.asyncio
    async def test_place_order_validation_failure(self, oms):
        """Test order placement with validation failure."""
        invalid_order = {
            "symbol": "INVALID",
            "side": "BUY",
            "quantity": -100,  # Invalid negative quantity
            "user_id": "test_user"
        }
        
        with patch.object(oms, '_validate_order', return_value=False):
            with pytest.raises(ValueError, match="Invalid order"):
                await oms.place_order(invalid_order)

    @pytest.mark.asyncio
    async def test_place_order_risk_limit_exceeded(self, oms):
        """Test order placement with risk limit exceeded."""
        order_request = {
            "symbol": "AAPL",
            "side": "BUY",
            "order_type": "MARKET",
            "quantity": 10000,  # Large quantity
            "user_id": "test_user"
        }
        
        with patch.object(oms, '_validate_order', return_value=True):
            with patch.object(oms, '_check_risk_limits', return_value=False):
                with pytest.raises(ValueError, match="Risk limit exceeded"):
                    await oms.place_order(order_request)

    @pytest.mark.asyncio
    async def test_get_order_success(self, oms, sample_order):
        """Test successful order retrieval."""
        order_id = "order_001"
        
        oms.database.fetchrow.return_value = {
            "id": order_id,
            "symbol": "AAPL",
            "side": "BUY",
            "status": "FILLED",
            "quantity": 100,
            "filled_quantity": 100
        }
        
        result = await oms.get_order(order_id)
        
        assert result is not None
        assert result["id"] == order_id
        assert result["status"] == "FILLED"

    @pytest.mark.asyncio
    async def test_get_order_not_found(self, oms):
        """Test order retrieval for non-existent order."""
        oms.database.fetchrow.return_value = None
        
        result = await oms.get_order("non_existent_order")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, oms):
        """Test successful order cancellation."""
        order_id = "order_001"
        
        # Mock order exists and is cancellable
        oms.database.fetchrow.return_value = {
            "id": order_id,
            "status": "PENDING",
            "broker_order_id": "broker_123"
        }
        
        with patch.object(oms, '_cancel_with_broker', return_value=True):
            result = await oms.cancel_order(order_id)
            
            assert result is True
            oms.database.execute.assert_called()  # Should update order status

    @pytest.mark.asyncio
    async def test_cancel_order_already_filled(self, oms):
        """Test cancellation of already filled order."""
        order_id = "order_001"
        
        oms.database.fetchrow.return_value = {
            "id": order_id,
            "status": "FILLED"  # Already filled, can't cancel
        }
        
        with pytest.raises(ValueError, match="Cannot cancel filled order"):
            await oms.cancel_order(order_id)

    @pytest.mark.asyncio
    async def test_update_order_status(self, oms):
        """Test order status update."""
        order_id = "order_001"
        new_status = "PARTIALLY_FILLED"
        fill_data = {
            "filled_quantity": 50,
            "average_price": 150.25,
            "commission": 1.00
        }
        
        await oms.update_order_status(order_id, new_status, fill_data)
        
        # Should update database and send notification
        oms.database.execute.assert_called()

    @pytest.mark.asyncio
    async def test_get_user_orders(self, oms):
        """Test retrieval of user orders."""
        user_id = "test_user"
        
        mock_orders = [
            {
                "id": "order_001",
                "symbol": "AAPL",
                "status": "FILLED",
                "user_id": user_id
            },
            {
                "id": "order_002", 
                "symbol": "GOOGL",
                "status": "PENDING",
                "user_id": user_id
            }
        ]
        
        oms.database.fetch.return_value = mock_orders
        
        result = await oms.get_user_orders(user_id)
        
        assert len(result) == 2
        assert all(order["user_id"] == user_id for order in result)

    @pytest.mark.asyncio
    async def test_get_orders_by_symbol(self, oms):
        """Test retrieval of orders by symbol."""
        symbol = "AAPL"
        
        mock_orders = [
            {"id": "order_001", "symbol": symbol, "status": "FILLED"},
            {"id": "order_002", "symbol": symbol, "status": "PENDING"}
        ]
        
        oms.database.fetch.return_value = mock_orders
        
        result = await oms.get_orders_by_symbol(symbol)
        
        assert len(result) == 2
        assert all(order["symbol"] == symbol for order in result)

    @pytest.mark.asyncio
    async def test_get_pending_orders(self, oms):
        """Test retrieval of pending orders."""
        mock_orders = [
            {"id": "order_001", "status": "PENDING"},
            {"id": "order_002", "status": "SUBMITTED"}
        ]
        
        oms.database.fetch.return_value = mock_orders
        
        result = await oms.get_pending_orders()
        
        assert len(result) == 2
        assert all(order["status"] in ["PENDING", "SUBMITTED"] for order in result)

    @pytest.mark.asyncio
    async def test_order_validation(self, oms):
        """Test order validation logic."""
        # Valid order
        valid_order = {
            "symbol": "AAPL",
            "side": "BUY",
            "order_type": "MARKET",
            "quantity": 100,
            "user_id": "test_user"
        }
        
        is_valid = oms._validate_order(valid_order)
        assert is_valid is True
        
        # Invalid order - missing symbol
        invalid_order = {
            "side": "BUY",
            "quantity": 100
        }
        
        is_valid = oms._validate_order(invalid_order)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_risk_limit_checking(self, oms):
        """Test risk limit checking."""
        order_request = {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 1000,
            "price": 150.00,
            "user_id": "test_user"
        }
        
        # Mock user portfolio and limits
        with patch.object(oms, '_get_user_portfolio') as mock_portfolio:
            mock_portfolio.return_value = {
                "total_value": 100000,
                "available_cash": 50000,
                "positions": {}
            }
            
            # Should pass risk check
            result = oms._check_risk_limits(order_request)
            assert result is True

    @pytest.mark.asyncio
    async def test_position_size_limits(self, oms):
        """Test position size limit checking."""
        large_order = {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 50000,  # Very large position
            "user_id": "test_user"
        }
        
        with patch.object(oms, '_get_position_limits') as mock_limits:
            mock_limits.return_value = {"max_position_size": 10000}
            
            result = oms._check_position_limits(large_order)
            assert result is False

    @pytest.mark.asyncio
    async def test_order_execution_simulation(self, oms):
        """Test order execution simulation for market orders."""
        market_order = {
            "symbol": "AAPL",
            "side": "BUY",
            "order_type": "MARKET",
            "quantity": 100
        }
        
        with patch.object(oms, '_get_current_price', return_value=150.25):
            execution_price = oms._simulate_execution(market_order)
            
            # Market order should execute at current price (with some slippage)
            assert 150.0 <= execution_price <= 150.5

    @pytest.mark.asyncio
    async def test_limit_order_execution(self, oms):
        """Test limit order execution logic."""
        limit_order = {
            "symbol": "AAPL",
            "side": "BUY",
            "order_type": "LIMIT",
            "quantity": 100,
            "price": 149.00
        }
        
        # Current price above limit - should not execute
        with patch.object(oms, '_get_current_price', return_value=150.25):
            can_execute = oms._can_execute_limit_order(limit_order)
            assert can_execute is False
        
        # Current price at/below limit - should execute
        with patch.object(oms, '_get_current_price', return_value=148.50):
            can_execute = oms._can_execute_limit_order(limit_order)
            assert can_execute is True

    @pytest.mark.asyncio
    async def test_stop_loss_order_triggering(self, oms):
        """Test stop loss order triggering."""
        stop_order = {
            "symbol": "AAPL",
            "side": "SELL",
            "order_type": "STOP_LOSS",
            "quantity": 100,
            "stop_price": 145.00
        }
        
        # Price above stop - should not trigger
        with patch.object(oms, '_get_current_price', return_value=150.25):
            should_trigger = oms._should_trigger_stop_order(stop_order)
            assert should_trigger is False
        
        # Price at/below stop - should trigger
        with patch.object(oms, '_get_current_price', return_value=144.50):
            should_trigger = oms._should_trigger_stop_order(stop_order)
            assert should_trigger is True

    @pytest.mark.asyncio
    async def test_order_matching_engine(self, oms):
        """Test internal order matching for testing."""
        buy_order = {
            "side": "BUY",
            "quantity": 100,
            "price": 150.00
        }
        
        sell_order = {
            "side": "SELL", 
            "quantity": 100,
            "price": 149.50
        }
        
        # Orders should match
        match = oms._find_matching_orders(buy_order, [sell_order])
        assert match is not None
        assert match["quantity"] == 100

    @pytest.mark.asyncio
    async def test_partial_fill_handling(self, oms):
        """Test handling of partial fills."""
        order_id = "order_001"
        
        partial_fill = {
            "filled_quantity": 50,
            "remaining_quantity": 50,
            "average_price": 150.25,
            "commission": 0.50
        }
        
        await oms.handle_partial_fill(order_id, partial_fill)
        
        # Should update order with partial fill info
        oms.database.execute.assert_called()

    @pytest.mark.asyncio
    async def test_order_expiry_handling(self, oms):
        """Test handling of expired orders."""
        # Mock expired orders
        expired_orders = [
            {
                "id": "order_001",
                "expires_at": datetime.utcnow() - timedelta(hours=1)
            }
        ]
        
        with patch.object(oms, '_get_expired_orders', return_value=expired_orders):
            await oms.process_expired_orders()
            
            # Should cancel expired orders
            oms.database.execute.assert_called()

    @pytest.mark.asyncio
    async def test_order_history_retrieval(self, oms):
        """Test order history retrieval."""
        user_id = "test_user"
        
        result = await oms.get_order_history(user_id, limit=100)
        
        oms.database.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_order_performance_metrics(self, oms):
        """Test order performance metrics calculation."""
        metrics = await oms.get_order_performance_metrics("test_user")
        
        # Should calculate fill rate, average execution time, etc.
        assert "fill_rate" in metrics
        assert "average_execution_time" in metrics
        assert "total_orders" in metrics

    @pytest.mark.asyncio
    async def test_service_health_check(self, oms):
        """Test service health check."""
        health = await oms.get_service_health()
        
        assert "status" in health
        assert "pending_orders_count" in health
        assert "orders_processed_today" in health

    @pytest.mark.asyncio
    async def test_concurrent_order_processing(self, oms):
        """Test concurrent order processing."""
        import asyncio
        
        order_requests = [
            {"symbol": "AAPL", "side": "BUY", "quantity": 100, "user_id": f"user_{i}"}
            for i in range(5)
        ]
        
        with patch.object(oms, '_validate_order', return_value=True):
            with patch.object(oms, '_check_risk_limits', return_value=True):
                with patch.object(oms, '_submit_to_broker', return_value="broker_id"):
                    
                    tasks = [oms.place_order(req) for req in order_requests]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # All orders should be processed successfully
                    assert len(results) == 5
                    assert all(not isinstance(r, Exception) for r in results)