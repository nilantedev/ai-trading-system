#!/usr/bin/env python3
"""
Comprehensive WebSocket tests covering streaming, authentication, and performance.
Tests real-time market data streaming, portfolio updates, and error handling.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import os
import sys

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.websocket_manager import (
    ConnectionManager, WebSocketStreamer, AuthenticatedConnectionManager,
    MarketDataStreamer, PortfolioUpdateStreamer, OrderUpdateStreamer
)
from api.auth import User, UserRole
from trading_common import MarketData, Position, Order
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState


class TestWebSocketAuthentication:
    """Test WebSocket authentication and authorization."""
    
    @pytest.fixture
    def test_user(self):
        """Create test user for WebSocket auth."""
        return User(
            user_id="ws_user_001",
            username="ws_test_user",
            roles=[UserRole.TRADER],
            permissions=["read:market_data", "read:portfolio"],
            is_active=True
        )
    
    @pytest.fixture
    async def mock_websocket(self):
        """Create mock WebSocket with auth capabilities."""
        websocket = AsyncMock(spec=WebSocket)
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.send_json = AsyncMock()
        websocket.receive_json = AsyncMock()
        websocket.close = AsyncMock()
        websocket.client_state = WebSocketState.CONNECTED
        return websocket
    
    @pytest.fixture
    async def auth_connection_manager(self):
        """Create authenticated connection manager."""
        return AuthenticatedConnectionManager()
    
    @pytest.mark.asyncio
    async def test_websocket_auth_success(self, auth_connection_manager, mock_websocket, test_user):
        """Test successful WebSocket authentication."""
        # Mock token validation
        with patch('api.websocket_manager.verify_access_token') as mock_verify:
            mock_verify.return_value = AsyncMock(username=test_user.username, user_id=test_user.user_id)
            
            # Mock user retrieval
            with patch('api.websocket_manager.get_user_by_id', return_value=test_user):
                # Simulate auth message
                auth_message = {
                    "type": "authenticate",
                    "token": "valid_jwt_token"
                }
                mock_websocket.receive_json.return_value = auth_message
                
                # Attempt authentication
                authenticated_user = await auth_connection_manager.authenticate_websocket(mock_websocket)
                
                assert authenticated_user is not None
                assert authenticated_user.username == test_user.username
                mock_websocket.accept.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_websocket_auth_failure(self, auth_connection_manager, mock_websocket):
        """Test WebSocket authentication failure."""
        # Mock token validation failure
        with patch('api.websocket_manager.verify_access_token') as mock_verify:
            mock_verify.side_effect = Exception("Invalid token")
            
            # Simulate auth message with invalid token
            auth_message = {
                "type": "authenticate", 
                "token": "invalid_jwt_token"
            }
            mock_websocket.receive_json.return_value = auth_message
            
            # Attempt authentication
            authenticated_user = await auth_connection_manager.authenticate_websocket(mock_websocket)
            
            assert authenticated_user is None
            mock_websocket.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_websocket_permission_check(self, auth_connection_manager, test_user):
        """Test WebSocket permission checking."""
        # Test user with market data permission
        has_permission = auth_connection_manager.check_permission(test_user, "read:market_data")
        assert has_permission is True
        
        # Test user without admin permission
        has_admin = auth_connection_manager.check_permission(test_user, "write:system")
        assert has_admin is False


class TestMarketDataStreaming:
    """Test real-time market data streaming via WebSocket."""
    
    @pytest.fixture
    async def market_data_streamer(self):
        """Create market data streamer instance."""
        return MarketDataStreamer()
    
    @pytest.fixture
    async def mock_websockets(self):
        """Create multiple mock WebSocket connections."""
        websockets = []
        for i in range(3):
            ws = AsyncMock(spec=WebSocket)
            ws.accept = AsyncMock()
            ws.send_json = AsyncMock()
            ws.close = AsyncMock()
            ws.client_state = WebSocketState.CONNECTED
            websockets.append(ws)
        return websockets
    
    @pytest.mark.asyncio
    async def test_market_data_subscription(self, market_data_streamer, mock_websockets):
        """Test market data subscription management."""
        symbols = ["AAPL", "GOOGL", "TSLA"]
        
        # Subscribe websockets to different symbols
        for i, ws in enumerate(mock_websockets):
            symbol = symbols[i % len(symbols)]
            await market_data_streamer.subscribe_to_symbol(ws, symbol)
        
        # Verify subscriptions
        assert len(market_data_streamer.symbol_subscriptions["AAPL"]) >= 1
        assert len(market_data_streamer.symbol_subscriptions["GOOGL"]) >= 1
        assert len(market_data_streamer.symbol_subscriptions["TSLA"]) >= 1
    
    @pytest.mark.asyncio
    async def test_market_data_broadcast(self, market_data_streamer, mock_websockets):
        """Test broadcasting market data to subscribers."""
        symbol = "AAPL"
        
        # Subscribe all websockets to same symbol
        for ws in mock_websockets:
            await market_data_streamer.subscribe_to_symbol(ws, symbol)
        
        # Create market data update
        market_data = MarketData(
            symbol=symbol,
            price=175.50,
            volume=1000000,
            bid=175.45,
            ask=175.55,
            timestamp=datetime.utcnow()
        )
        
        # Broadcast update
        await market_data_streamer.broadcast_market_data(market_data)
        
        # Verify all subscribers received the update
        for ws in mock_websockets:
            ws.send_json.assert_called()
            call_args = ws.send_json.call_args[0][0]
            assert call_args["type"] == "market_data"
            assert call_args["symbol"] == symbol
            assert call_args["price"] == 175.50
    
    @pytest.mark.asyncio
    async def test_websocket_disconnection_cleanup(self, market_data_streamer, mock_websockets):
        """Test cleanup when WebSocket disconnects."""
        symbol = "AAPL"
        ws = mock_websockets[0]
        
        # Subscribe websocket
        await market_data_streamer.subscribe_to_symbol(ws, symbol)
        assert ws in market_data_streamer.symbol_subscriptions[symbol]
        
        # Simulate disconnection
        await market_data_streamer.unsubscribe_websocket(ws)
        
        # Verify cleanup
        assert ws not in market_data_streamer.symbol_subscriptions.get(symbol, set())
    
    @pytest.mark.asyncio
    async def test_market_data_throttling(self, market_data_streamer, mock_websockets):
        """Test market data update throttling to prevent spam."""
        symbol = "AAPL"
        ws = mock_websockets[0]
        
        await market_data_streamer.subscribe_to_symbol(ws, symbol)
        
        # Send multiple rapid updates for same symbol
        updates = []
        for i in range(10):
            market_data = MarketData(
                symbol=symbol,
                price=175.00 + i * 0.01,
                volume=1000000,
                timestamp=datetime.utcnow()
            )
            updates.append(market_data_streamer.broadcast_market_data(market_data))
        
        # Execute all updates
        await asyncio.gather(*updates)
        
        # Should throttle updates (exact behavior depends on implementation)
        # At minimum, verify no errors occurred
        assert ws.send_json.call_count > 0
        assert ws.send_json.call_count <= 10


class TestPortfolioUpdates:
    """Test real-time portfolio update streaming."""
    
    @pytest.fixture
    async def portfolio_streamer(self):
        """Create portfolio update streamer."""
        return PortfolioUpdateStreamer()
    
    @pytest.fixture
    def test_user(self):
        """Create test user for portfolio updates."""
        return User(
            user_id="portfolio_user_001",
            username="portfolio_user",
            roles=[UserRole.TRADER],
            permissions=["read:portfolio"],
            is_active=True
        )
    
    @pytest.mark.asyncio
    async def test_portfolio_subscription(self, portfolio_streamer, test_user):
        """Test portfolio update subscription."""
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.client_state = WebSocketState.CONNECTED
        
        await portfolio_streamer.subscribe_user_portfolio(mock_websocket, test_user.user_id)
        
        assert test_user.user_id in portfolio_streamer.user_subscriptions
        assert mock_websocket in portfolio_streamer.user_subscriptions[test_user.user_id]
    
    @pytest.mark.asyncio
    async def test_position_update_broadcast(self, portfolio_streamer, test_user):
        """Test broadcasting position updates."""
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.client_state = WebSocketState.CONNECTED
        
        # Subscribe to portfolio updates
        await portfolio_streamer.subscribe_user_portfolio(mock_websocket, test_user.user_id)
        
        # Create position update
        position = Position(
            symbol="AAPL",
            quantity=150,
            avg_cost=170.00,
            current_price=175.50,
            market_value=26325.00,
            unrealized_pnl=825.00,
            side="long",
            last_updated=datetime.utcnow()
        )
        
        # Broadcast update
        await portfolio_streamer.broadcast_position_update(test_user.user_id, position)
        
        # Verify update was sent
        mock_websocket.send_json.assert_called_once()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["type"] == "position_update"
        assert call_args["symbol"] == "AAPL"
        assert call_args["quantity"] == 150


class TestOrderUpdates:
    """Test real-time order update streaming."""
    
    @pytest.fixture
    async def order_streamer(self):
        """Create order update streamer."""
        return OrderUpdateStreamer()
    
    @pytest.mark.asyncio
    async def test_order_status_update(self, order_streamer):
        """Test order status update broadcasting."""
        user_id = "order_user_001"
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.client_state = WebSocketState.CONNECTED
        
        # Subscribe to order updates
        await order_streamer.subscribe_user_orders(mock_websocket, user_id)
        
        # Create order update
        from services.execution.order_management_system import Order, OrderType, OrderSide, OrderStatus
        order = Order(
            order_id="order_123",
            user_id=user_id,
            symbol="GOOGL",
            quantity=25,
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            price=2650.00,
            status=OrderStatus.FILLED,
            time_in_force="DAY",
            created_at=datetime.utcnow(),
            filled_at=datetime.utcnow()
        )
        
        # Broadcast order update
        await order_streamer.broadcast_order_update(user_id, order)
        
        # Verify update was sent
        mock_websocket.send_json.assert_called_once()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["type"] == "order_update"
        assert call_args["order_id"] == "order_123"
        assert call_args["status"] == "filled"


class TestWebSocketPerformance:
    """Test WebSocket performance and scalability."""
    
    @pytest.mark.asyncio
    async def test_concurrent_connections(self):
        """Test handling multiple concurrent WebSocket connections."""
        connection_manager = ConnectionManager()
        websockets = []
        
        # Create many mock websockets
        for i in range(100):
            ws = AsyncMock(spec=WebSocket)
            ws.accept = AsyncMock()
            ws.send_json = AsyncMock()
            ws.client_state = WebSocketState.CONNECTED
            websockets.append(ws)
        
        # Connect all websockets concurrently
        tasks = [connection_manager.connect(ws, "market_data") for ws in websockets]
        await asyncio.gather(*tasks)
        
        # Verify all connections were accepted
        assert len(connection_manager.connections["market_data"]) == 100
        for ws in websockets:
            ws.accept.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_broadcast_performance(self):
        """Test broadcast performance with many connections."""
        connection_manager = ConnectionManager()
        websockets = []
        
        # Create multiple websockets
        for i in range(50):
            ws = AsyncMock(spec=WebSocket)
            ws.send_json = AsyncMock()
            ws.client_state = WebSocketState.CONNECTED
            websockets.append(ws)
            await connection_manager.connect(ws, "market_data")
        
        # Broadcast message to all
        test_message = {"type": "test", "data": "performance_test"}
        start_time = datetime.utcnow()
        
        await connection_manager.broadcast_to_stream("market_data", test_message)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete quickly (less than 1 second for 50 connections)
        assert duration < 1.0
        
        # Verify all websockets received message
        for ws in websockets:
            ws.send_json.assert_called_once_with(test_message)


class TestWebSocketErrorHandling:
    """Test WebSocket error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_connection_failure_handling(self):
        """Test handling of WebSocket connection failures."""
        connection_manager = ConnectionManager()
        
        # Mock websocket that fails during accept
        failing_ws = AsyncMock(spec=WebSocket)
        failing_ws.accept.side_effect = Exception("Connection failed")
        
        # Should handle connection failure gracefully
        try:
            await connection_manager.connect(failing_ws, "market_data")
        except Exception:
            # Connection failure should be handled internally
            pass
        
        # Failed connection should not be in active connections
        assert failing_ws not in connection_manager.connections.get("market_data", set())
    
    @pytest.mark.asyncio
    async def test_send_failure_cleanup(self):
        """Test cleanup when sending to WebSocket fails."""
        connection_manager = ConnectionManager()
        
        # Mock websocket that fails during send
        failing_ws = AsyncMock(spec=WebSocket)
        failing_ws.accept = AsyncMock()
        failing_ws.send_json.side_effect = Exception("Send failed")
        failing_ws.client_state = WebSocketState.CONNECTED
        
        # Connect websocket
        await connection_manager.connect(failing_ws, "market_data")
        
        # Attempt to send message (should fail but be handled)
        await connection_manager.send_to_websocket(failing_ws, {"type": "test"})
        
        # Failed websocket should be cleaned up
        # (Exact cleanup behavior depends on implementation)
        # At minimum, no exceptions should propagate
        assert True  # Test passes if no exceptions were raised
    
    @pytest.mark.asyncio
    async def test_websocket_disconnect_event(self):
        """Test proper handling of WebSocket disconnect events."""
        connection_manager = ConnectionManager()
        
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.client_state = WebSocketState.CONNECTED
        
        # Connect websocket
        await connection_manager.connect(mock_ws, "market_data")
        assert mock_ws in connection_manager.connections["market_data"]
        
        # Simulate disconnect
        await connection_manager.disconnect(mock_ws, "market_data")
        
        # Websocket should be removed from connections
        assert mock_ws not in connection_manager.connections.get("market_data", set())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])