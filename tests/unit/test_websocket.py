#!/usr/bin/env python3
"""
Unit tests for WebSocket functionality
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import json
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.websocket_manager import ConnectionManager, WebSocketStreamer
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect


class TestConnectionManager:
    """Test cases for WebSocket ConnectionManager."""

    @pytest_asyncio.fixture
    async def connection_manager(self):
        """Create ConnectionManager instance for testing."""
        return ConnectionManager()

    @pytest_asyncio.fixture
    async def mock_websocket(self):
        """Create mock WebSocket connection."""
        websocket = AsyncMock(spec=WebSocket)
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.send_json = AsyncMock()
        websocket.receive_text = AsyncMock()
        websocket.receive_json = AsyncMock()
        websocket.close = AsyncMock()
        return websocket

    @pytest.mark.asyncio
    async def test_connect_websocket(self, connection_manager, mock_websocket):
        """Test WebSocket connection."""
        stream_type = "market_data"
        
        await connection_manager.connect(mock_websocket, stream_type)
        
        # Should accept connection and add to connections
        mock_websocket.accept.assert_called_once()
        assert mock_websocket in connection_manager.connections[stream_type]

    @pytest.mark.asyncio
    async def test_disconnect_websocket(self, connection_manager, mock_websocket):
        """Test WebSocket disconnection."""
        stream_type = "market_data"
        
        # First connect
        await connection_manager.connect(mock_websocket, stream_type)
        
        # Then disconnect
        await connection_manager.disconnect(mock_websocket)
        
        # Should remove from all connection sets
        for connections in connection_manager.connections.values():
            assert mock_websocket not in connections

    @pytest.mark.asyncio
    async def test_broadcast_to_stream(self, connection_manager, mock_websocket):
        """Test broadcasting message to stream."""
        stream_type = "market_data"
        message = {
            "type": "market_data_update",
            "symbol": "AAPL",
            "price": 150.25
        }
        
        # Connect websocket
        await connection_manager.connect(mock_websocket, stream_type)
        
        # Broadcast message
        await connection_manager.broadcast_to_stream(stream_type, message)
        
        # Should send message to connected websocket
        mock_websocket.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_connections(self, connection_manager):
        """Test broadcasting to multiple connections."""
        stream_type = "market_data"
        message = {"type": "test", "data": "broadcast"}
        
        # Create multiple mock websockets
        websockets = [AsyncMock(spec=WebSocket) for _ in range(3)]
        
        # Connect all websockets
        for ws in websockets:
            ws.accept = AsyncMock()
            ws.send_json = AsyncMock()
            await connection_manager.connect(ws, stream_type)
        
        # Broadcast message
        await connection_manager.broadcast_to_stream(stream_type, message)
        
        # All websockets should receive the message
        for ws in websockets:
            ws.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_handle_client_message(self, connection_manager, mock_websocket):
        """Test handling client messages."""
        stream_type = "market_data"
        client_message = json.dumps({
            "type": "subscribe",
            "symbols": ["AAPL", "GOOGL"]
        })
        
        await connection_manager.connect(mock_websocket, stream_type)
        
        with patch.object(connection_manager, '_process_subscription') as mock_process:
            await connection_manager.handle_client_message(mock_websocket, client_message)
            
            mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_with_symbols(self, connection_manager, mock_websocket):
        """Test connection with symbol subscription."""
        stream_type = "market_data"
        symbols = ["AAPL", "GOOGL"]
        
        await connection_manager.connect(
            mock_websocket, 
            stream_type, 
            symbols=symbols
        )
        
        # Should store symbol subscription
        connection_info = connection_manager.connection_info.get(mock_websocket)
        assert connection_info is not None
        assert connection_info["symbols"] == symbols

    @pytest.mark.asyncio
    async def test_connection_with_user_info(self, connection_manager, mock_websocket):
        """Test connection with user authentication."""
        stream_type = "orders"
        user_info = {
            "user_id": "test_user",
            "token": "demo_token",
            "authenticated": True
        }
        
        await connection_manager.connect(
            mock_websocket,
            stream_type,
            user_info=user_info
        )
        
        connection_info = connection_manager.connection_info.get(mock_websocket)
        assert connection_info["user_info"] == user_info

    @pytest.mark.asyncio
    async def test_get_connection_stats(self, connection_manager, mock_websocket):
        """Test connection statistics retrieval."""
        # Connect to multiple streams
        await connection_manager.connect(mock_websocket, "market_data")
        
        ws2 = AsyncMock(spec=WebSocket)
        ws2.accept = AsyncMock()
        await connection_manager.connect(ws2, "signals")
        
        stats = connection_manager.get_connection_stats()
        
        assert "total_connections" in stats
        assert "connections_by_stream" in stats
        assert stats["total_connections"] == 2

    @pytest.mark.asyncio
    async def test_broadcast_to_user(self, connection_manager, mock_websocket):
        """Test broadcasting message to specific user."""
        user_id = "test_user"
        user_info = {"user_id": user_id, "authenticated": True}
        
        await connection_manager.connect(
            mock_websocket,
            "orders",
            user_info=user_info
        )
        
        message = {"type": "order_update", "order_id": "order_001"}
        
        await connection_manager.broadcast_to_user(user_id, message)
        
        mock_websocket.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_error_handling_send_failure(self, connection_manager, mock_websocket):
        """Test error handling when sending message fails."""
        stream_type = "market_data"
        
        # Mock send_json to raise exception
        mock_websocket.send_json.side_effect = Exception("Connection lost")
        
        await connection_manager.connect(mock_websocket, stream_type)
        
        message = {"type": "test"}
        
        # Should handle the error gracefully and remove bad connection
        await connection_manager.broadcast_to_stream(stream_type, message)
        
        # Connection should be removed due to error
        assert mock_websocket not in connection_manager.connections[stream_type]


class TestWebSocketStreamer:
    """Test cases for WebSocketStreamer."""

    @pytest_asyncio.fixture
    async def streamer(self, mock_redis):
        """Create WebSocketStreamer instance."""
        return WebSocketStreamer(mock_redis)

    @pytest.mark.asyncio
    async def test_start_streaming(self, streamer):
        """Test starting data streaming."""
        with patch.object(streamer, '_stream_market_data') as mock_stream:
            await streamer.start_streaming()
            
            # Should start streaming task
            assert streamer.streaming_task is not None

    @pytest.mark.asyncio
    async def test_stop_streaming(self, streamer):
        """Test stopping data streaming."""
        # Start streaming first
        with patch.object(streamer, '_stream_market_data'):
            await streamer.start_streaming()
        
        # Then stop
        await streamer.stop_streaming()
        
        # Streaming task should be cancelled
        assert streamer.streaming_task.cancelled()

    @pytest.mark.asyncio
    async def test_market_data_streaming(self, streamer):
        """Test market data streaming."""
        mock_data = {
            "AAPL": {
                "price": 150.25,
                "volume": 1000000,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        with patch.object(streamer, '_get_latest_market_data', return_value=mock_data):
            with patch.object(streamer.connection_manager, 'broadcast_to_stream') as mock_broadcast:
                
                await streamer._stream_market_data()
                
                mock_broadcast.assert_called()

    @pytest.mark.asyncio
    async def test_signal_streaming(self, streamer):
        """Test trading signal streaming."""
        mock_signals = [
            {
                "id": "signal_001",
                "symbol": "AAPL",
                "signal_type": "BUY",
                "confidence": 0.85
            }
        ]
        
        with patch.object(streamer, '_get_latest_signals', return_value=mock_signals):
            with patch.object(streamer.connection_manager, 'broadcast_to_stream') as mock_broadcast:
                
                await streamer._stream_signals()
                
                mock_broadcast.assert_called_with("signals", mock_signals)

    @pytest.mark.asyncio
    async def test_order_update_streaming(self, streamer):
        """Test order update streaming."""
        order_update = {
            "order_id": "order_001",
            "status": "FILLED",
            "filled_quantity": 100,
            "user_id": "test_user"
        }
        
        with patch.object(streamer.connection_manager, 'broadcast_to_user') as mock_broadcast:
            
            await streamer.stream_order_update(order_update)
            
            mock_broadcast.assert_called_with("test_user", {
                "type": "order_update",
                "data": order_update
            })

    @pytest.mark.asyncio
    async def test_portfolio_update_streaming(self, streamer):
        """Test portfolio update streaming."""
        portfolio_update = {
            "user_id": "test_user",
            "total_value": 105000.0,
            "day_change": 5000.0
        }
        
        with patch.object(streamer.connection_manager, 'broadcast_to_user') as mock_broadcast:
            
            await streamer.stream_portfolio_update(portfolio_update)
            
            mock_broadcast.assert_called_with("test_user", {
                "type": "portfolio_update", 
                "data": portfolio_update
            })

    @pytest.mark.asyncio
    async def test_alert_streaming(self, streamer):
        """Test system alert streaming."""
        alert = {
            "id": "alert_001",
            "severity": "HIGH",
            "message": "System latency spike detected",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with patch.object(streamer.connection_manager, 'broadcast_to_stream') as mock_broadcast:
            
            await streamer.stream_alert(alert)
            
            mock_broadcast.assert_called_with("alerts", {
                "type": "alert",
                "data": alert
            })

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, streamer):
        """Test error handling in streaming."""
        # Mock exception in data retrieval
        with patch.object(streamer, '_get_latest_market_data', side_effect=Exception("Data error")):
            with patch.object(streamer, '_handle_streaming_error') as mock_error_handler:
                
                await streamer._stream_market_data()
                
                mock_error_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_filtering(self, streamer):
        """Test filtering connections by subscription."""
        # Test symbol-based filtering
        mock_connections = {
            "ws1": {"symbols": ["AAPL"]},
            "ws2": {"symbols": ["GOOGL"]}, 
            "ws3": {"symbols": ["AAPL", "GOOGL"]}
        }
        
        filtered = streamer._filter_connections_by_symbol(mock_connections, "AAPL")
        
        # Should return connections subscribed to AAPL
        assert len(filtered) == 2  # ws1 and ws3

    @pytest.mark.asyncio
    async def test_rate_limiting_streams(self, streamer):
        """Test rate limiting for streams."""
        # Should limit updates to prevent overwhelming clients
        with patch.object(streamer, '_should_rate_limit', return_value=True):
            with patch.object(streamer.connection_manager, 'broadcast_to_stream') as mock_broadcast:
                
                await streamer._stream_market_data()
                
                # Should not broadcast if rate limited
                mock_broadcast.assert_not_called()

    @pytest.mark.asyncio
    async def test_data_compression(self, streamer):
        """Test data compression for large messages."""
        large_data = {"data": "x" * 10000}  # Large message
        
        with patch.object(streamer, '_compress_data') as mock_compress:
            mock_compress.return_value = {"compressed": True}
            
            compressed = streamer._prepare_message(large_data)
            
            mock_compress.assert_called_once()

    @pytest.mark.asyncio
    async def test_heartbeat_mechanism(self, streamer):
        """Test WebSocket heartbeat/ping mechanism."""
        with patch.object(streamer.connection_manager, 'broadcast_to_stream') as mock_broadcast:
            
            await streamer._send_heartbeat()
            
            # Should send ping to all streams
            expected_calls = len(streamer.connection_manager.connections)
            assert mock_broadcast.call_count == expected_calls

    @pytest.mark.asyncio
    async def test_subscription_management(self, streamer, mock_websocket):
        """Test dynamic subscription management."""
        subscription_msg = {
            "type": "subscribe",
            "stream": "market_data",
            "symbols": ["AAPL", "GOOGL"]
        }
        
        await streamer.handle_subscription(mock_websocket, subscription_msg)
        
        # Should update connection subscriptions
        connection_info = streamer.connection_manager.connection_info.get(mock_websocket)
        assert connection_info is not None

    @pytest.mark.asyncio
    async def test_unsubscription(self, streamer, mock_websocket):
        """Test unsubscription from streams."""
        # First subscribe
        await streamer.connection_manager.connect(mock_websocket, "market_data", symbols=["AAPL"])
        
        unsubscribe_msg = {
            "type": "unsubscribe", 
            "symbols": ["AAPL"]
        }
        
        await streamer.handle_subscription(mock_websocket, unsubscribe_msg)
        
        # Should remove from subscriptions
        connection_info = streamer.connection_manager.connection_info.get(mock_websocket)
        assert "AAPL" not in connection_info.get("symbols", [])

    @pytest.mark.asyncio
    async def test_concurrent_streaming(self, streamer):
        """Test concurrent streaming to multiple clients."""
        import asyncio
        
        # Mock multiple data streams
        with patch.object(streamer, '_stream_market_data') as mock_market:
            with patch.object(streamer, '_stream_signals') as mock_signals:
                with patch.object(streamer, '_stream_alerts') as mock_alerts:
                    
                    # Start concurrent streaming
                    tasks = [
                        streamer._stream_market_data(),
                        streamer._stream_signals(),
                        streamer._stream_alerts()
                    ]
                    
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # All streams should have been called
                    mock_market.assert_called()
                    mock_signals.assert_called()
                    mock_alerts.assert_called()