#!/usr/bin/env python3
"""
WebSocket Manager - Real-time streaming for market data, signals, and system updates
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Set, Optional, Any
import json
import asyncio
import logging
from datetime import datetime
import uuid
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.models import (
    WebSocketMessage, MarketDataUpdate, SignalUpdate, OrderUpdate, AlertUpdate,
    MarketDataPoint, TradingSignal, Order, RiskAlert
)
from api.main import optional_auth, APIException
from trading_common import get_logger

# Import metrics (will be available after startup)
try:
    from api.metrics import metrics
except ImportError:
    metrics = None

logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""
    
    def __init__(self, enable_compression: bool = True, max_broadcast_rate: float = 100.0):
        # Active connections by stream type
        self.start_time = datetime.utcnow()  # Added for accurate uptime tracking
        self.connections: Dict[str, Set[WebSocket]] = {
            "market_data": set(),
            "signals": set(),
            "orders": set(),
            "portfolio": set(),
            "alerts": set(),
            "system": set()
        }
        
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        
        # Performance optimization settings
        self.enable_compression = enable_compression
        self.max_broadcast_rate = max_broadcast_rate  # messages per second
        self._last_broadcast_time = 0.0
        self._broadcast_rate_limiter = asyncio.Semaphore(int(max_broadcast_rate))
        
        # Stream subscriptions (connection -> subscribed streams)
        self.subscriptions: Dict[WebSocket, Set[str]] = {}
        
        # Symbol-specific subscriptions
        self.symbol_subscriptions: Dict[str, Set[WebSocket]] = {}
        
        # Statistics
        self.total_connections = 0
        self.messages_sent = 0
        self.connection_errors = 0
        
    async def connect(self, websocket: WebSocket, stream_type: str, 
                     symbols: Optional[List[str]] = None, user_info: Optional[Dict[str, Any]] = None):
        """Accept a new WebSocket connection."""
        try:
            await websocket.accept()
            
            # Add to connections
            if stream_type not in self.connections:
                self.connections[stream_type] = set()
            
            self.connections[stream_type].add(websocket)
            
            # Store metadata
            self.connection_metadata[websocket] = {
                "connection_id": str(uuid.uuid4()),
                "stream_type": stream_type,
                "connected_at": datetime.utcnow(),
                "user_info": user_info,
                "symbols": symbols or [],
                "message_count": 0
            }
            
            # Initialize subscriptions
            self.subscriptions[websocket] = {stream_type}
            
            # Handle symbol-specific subscriptions
            if symbols:
                for symbol in symbols:
                    symbol = symbol.upper()
                    if symbol not in self.symbol_subscriptions:
                        self.symbol_subscriptions[symbol] = set()
                    self.symbol_subscriptions[symbol].add(websocket)
            
            self.total_connections += 1
            
            # Send welcome message
            welcome_msg = {
                "type": "connection_established",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "connection_id": self.connection_metadata[websocket]["connection_id"],
                    "stream_type": stream_type,
                    "subscribed_symbols": symbols or [],
                    "message": f"Connected to {stream_type} stream"
                }
            }
            
            await self._send_message(websocket, welcome_msg)
            
            # Record metrics
            if metrics:
                metrics.record_websocket_connection(stream_type, True)
                metrics.update_websocket_connections(stream_type, len(self.connections[stream_type]))
            
            logger.info(f"WebSocket connected: {stream_type} stream (total: {len(self.connections[stream_type])})")
            
        except Exception as e:
            logger.error(f"Error connecting WebSocket: {e}")
            await self.disconnect(websocket)
    
    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        try:
            # Get metadata before removing
            metadata = self.connection_metadata.get(websocket, {})
            stream_type = metadata.get("stream_type", "unknown")
            symbols = metadata.get("symbols", [])
            
            # Remove from all connection sets
            for connections_set in self.connections.values():
                connections_set.discard(websocket)
            
            # Remove from symbol subscriptions
            for symbol in symbols:
                if symbol in self.symbol_subscriptions:
                    self.symbol_subscriptions[symbol].discard(websocket)
                    if not self.symbol_subscriptions[symbol]:
                        del self.symbol_subscriptions[symbol]
            
            # Clean up metadata
            self.connection_metadata.pop(websocket, None)
            self.subscriptions.pop(websocket, None)
            
            # Record metrics
            if metrics:
                metrics.record_websocket_connection(stream_type, False)
                remaining_connections = sum(1 for conns in self.connections.values() 
                                          for conn in conns if conn != websocket)
                metrics.update_websocket_connections(stream_type, remaining_connections)
            
            logger.info(f"WebSocket disconnected: {stream_type} stream")
            
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {e}")
    
    async def broadcast_to_stream(self, stream_type: str, message: Dict[str, Any]):
        """Broadcast message to all connections in a stream with optimized concurrent sending."""
        if stream_type not in self.connections:
            return
        
        connections = self.connections[stream_type].copy()
        if not connections:
            return
        
        # Use concurrent broadcasting for better performance
        await self._broadcast_concurrent(connections, message)
    
    async def broadcast_to_symbols(self, symbols: List[str], message: Dict[str, Any]):
        """Broadcast message to connections subscribed to specific symbols with optimized performance."""
        target_connections = set()
        
        for symbol in symbols:
            symbol = symbol.upper()
            if symbol in self.symbol_subscriptions:
                target_connections.update(self.symbol_subscriptions[symbol])
        
        if not target_connections:
            return
        
        # Use concurrent broadcasting for better performance
        await self._broadcast_concurrent(target_connections, message)
    
    async def send_to_connection(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to a specific connection."""
        try:
            await self._send_message(websocket, message)
            self.messages_sent += 1
        except Exception as e:
            logger.warning(f"Error sending message to WebSocket: {e}")
            await self.disconnect(websocket)
            self.connection_errors += 1
    
    async def _send_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send JSON message to WebSocket."""
        try:
            # Update message count for connection
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["message_count"] += 1
            
            await websocket.send_text(json.dumps(message, default=str))
        except WebSocketDisconnect:
            raise
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            raise
    
    async def _broadcast_concurrent(self, connections: set, message: Dict[str, Any], batch_size: int = 100):
        """
        Optimized concurrent broadcasting to multiple WebSocket connections.
        
        Features:
        - Concurrent sending with batching to prevent overwhelming the system
        - Automatic disconnection cleanup for failed connections
        - Message deduplication and compression for large broadcasts
        - Performance metrics tracking
        """
        if not connections:
            return
        
        # Prepare serialized message once for all connections
        serialized_message = json.dumps(message, default=str)
        
        # Apply compression if enabled and message is large enough
        compressed_message = None
        if self.enable_compression and len(serialized_message) > 1024:  # Compress messages > 1KB
            try:
                import gzip
                compressed_bytes = gzip.compress(serialized_message.encode('utf-8'))
                # Only use compression if it actually reduces size significantly
                if len(compressed_bytes) < len(serialized_message) * 0.8:
                    compressed_message = compressed_bytes
                    logger.debug(f"Compressed WebSocket message: {len(serialized_message)} -> {len(compressed_bytes)} bytes")
            except Exception as e:
                logger.warning(f"Message compression failed: {e}")
        
        # Choose the best message format
        message_to_send = compressed_message if compressed_message else serialized_message
        
        # Track performance
        start_time = asyncio.get_event_loop().time()
        successful_sends = 0
        failed_sends = 0
        disconnected = []
        
        # Process connections in batches to avoid overwhelming the system
        connection_list = list(connections)
        
        for i in range(0, len(connection_list), batch_size):
            batch = connection_list[i:i + batch_size]
            
            # Create concurrent tasks for this batch
            tasks = []
            for websocket in batch:
                task = self._send_message_concurrent(websocket, message_to_send, isinstance(message_to_send, bytes))
                tasks.append((websocket, task))
            
            # Wait for batch completion with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*[task for _, task in tasks], return_exceptions=True),
                    timeout=5.0  # 5 second timeout for batch
                )
                
                # Process results
                for (websocket, _), result in zip(tasks, results):
                    if isinstance(result, Exception):
                        logger.warning(f"Error broadcasting to WebSocket: {result}")
                        disconnected.append(websocket)
                        failed_sends += 1
                    else:
                        successful_sends += 1
                        self.messages_sent += 1
                        
            except asyncio.TimeoutError:
                logger.warning(f"WebSocket broadcast batch timeout - {len(batch)} connections")
                # Mark all connections in this batch as potentially failed
                disconnected.extend(batch)
                failed_sends += len(batch)
            
            # Small delay between batches to prevent overwhelming
            if i + batch_size < len(connection_list):
                await asyncio.sleep(0.001)  # 1ms delay
        
        # Clean up disconnected connections
        cleanup_tasks = [self.disconnect(websocket) for websocket in disconnected]
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Record performance metrics
        end_time = asyncio.get_event_loop().time()
        broadcast_time = end_time - start_time
        
        if metrics:
            try:
                metrics.record_websocket_broadcast(
                    connections_count=len(connections),
                    successful_sends=successful_sends,
                    failed_sends=failed_sends,
                    broadcast_time=broadcast_time
                )
            except Exception as e:
                logger.debug(f"Broadcast metrics recording failed: {e}")
        
        # Log performance information
        if len(connections) > 10:  # Only log for significant broadcasts
            logger.info(
                f"WebSocket broadcast: {successful_sends}/{len(connections)} successful, "
                f"{broadcast_time:.3f}s, { (successful_sends / broadcast_time) if broadcast_time else 0:.1f} msg/s"
            )
        
        self.connection_errors += failed_sends
    
    async def _send_message_concurrent(self, websocket: WebSocket, message_data, is_compressed: bool = False):
        """
        Send pre-serialized message to WebSocket with optimized error handling.
        
        This method is optimized for concurrent sending:
        - Uses pre-serialized message to avoid repeated JSON encoding
        - Supports compressed binary messages for large payloads
        - Minimal error handling to allow batch processing
        - Connection state validation
        """
        try:
            # Quick connection state check
            if websocket.client_state.name != 'CONNECTED':
                raise ConnectionError("WebSocket not connected")
            
            # Update message count before sending
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["message_count"] += 1
            
            # Send the message using the appropriate method
            if is_compressed:
                # Send compressed binary data
                await websocket.send_bytes(message_data)
            else:
                # Send text message
                await websocket.send_text(message_data)
            
        except WebSocketDisconnect:
            # Re-raise disconnect exceptions for proper cleanup
            raise
        except Exception as e:
            # Convert other exceptions to a standard format for batch processing
            raise ConnectionError(f"Send failed: {e}")
    
    async def handle_client_message(self, websocket: WebSocket, message: str):
        """Handle incoming message from client."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "subscribe":
                await self._handle_subscribe(websocket, data)
            elif message_type == "unsubscribe":
                await self._handle_unsubscribe(websocket, data)
            elif message_type == "ping":
                await self._handle_ping(websocket, data)
            else:
                # Send error for unknown message type
                error_msg = {
                    "type": "error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "error": "Unknown message type",
                        "received_type": message_type
                    }
                }
                await self.send_to_connection(websocket, error_msg)
            
        except json.JSONDecodeError:
            error_msg = {
                "type": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "error": "Invalid JSON format"
                }
            }
            await self.send_to_connection(websocket, error_msg)
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    async def _handle_subscribe(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle subscription request."""
        try:
            symbols = data.get("symbols", [])
            streams = data.get("streams", [])
            
            # Add symbol subscriptions
            for symbol in symbols:
                symbol = symbol.upper()
                if symbol not in self.symbol_subscriptions:
                    self.symbol_subscriptions[symbol] = set()
                self.symbol_subscriptions[symbol].add(websocket)
                
                # Update metadata
                if websocket in self.connection_metadata:
                    current_symbols = self.connection_metadata[websocket].get("symbols", [])
                    if symbol not in current_symbols:
                        current_symbols.append(symbol)
            
            # Add stream subscriptions
            if websocket in self.subscriptions:
                self.subscriptions[websocket].update(streams)
            
            # Send confirmation
            response = {
                "type": "subscription_confirmed",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "subscribed_symbols": symbols,
                    "subscribed_streams": streams,
                    "message": "Subscription updated"
                }
            }
            
            await self.send_to_connection(websocket, response)
            
        except Exception as e:
            logger.error(f"Error handling subscribe request: {e}")
    
    async def _handle_unsubscribe(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle unsubscription request."""
        try:
            symbols = data.get("symbols", [])
            streams = data.get("streams", [])
            
            # Remove symbol subscriptions
            for symbol in symbols:
                symbol = symbol.upper()
                if symbol in self.symbol_subscriptions:
                    self.symbol_subscriptions[symbol].discard(websocket)
                    if not self.symbol_subscriptions[symbol]:
                        del self.symbol_subscriptions[symbol]
                
                # Update metadata
                if websocket in self.connection_metadata:
                    current_symbols = self.connection_metadata[websocket].get("symbols", [])
                    if symbol in current_symbols:
                        current_symbols.remove(symbol)
            
            # Remove stream subscriptions
            if websocket in self.subscriptions:
                self.subscriptions[websocket].difference_update(streams)
            
            # Send confirmation
            response = {
                "type": "unsubscription_confirmed",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "unsubscribed_symbols": symbols,
                    "unsubscribed_streams": streams,
                    "message": "Unsubscription completed"
                }
            }
            
            await self.send_to_connection(websocket, response)
            
        except Exception as e:
            logger.error(f"Error handling unsubscribe request: {e}")
    
    async def _handle_ping(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle ping request."""
        try:
            response = {
                "type": "pong",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "ping_timestamp": data.get("timestamp"),
                    "server_timestamp": datetime.utcnow().isoformat()
                }
            }
            
            await self.send_to_connection(websocket, response)
            
        except Exception as e:
            logger.error(f"Error handling ping: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": sum(len(connections) for connections in self.connections.values()),
            "connections_by_stream": {
                stream: len(connections) 
                for stream, connections in self.connections.items()
            },
            "symbol_subscriptions": len(self.symbol_subscriptions),
            "messages_sent": self.messages_sent,
            "connection_errors": self.connection_errors,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()  # Fixed uptime
        }


# Global connection manager
connection_manager = ConnectionManager()


class WebSocketStreamer:
    """Handles real-time data streaming to WebSocket clients."""
    
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        self.is_streaming = False
    self.stream_tasks: List[asyncio.Task] = []
        
    async def start_streaming(self):
        """Start all streaming tasks."""
        if self.is_streaming:
            return
        
        self.is_streaming = True
        
        # Start streaming tasks
        self.stream_tasks = [
            asyncio.create_task(self._stream_market_data()),
            asyncio.create_task(self._stream_signals()),
            asyncio.create_task(self._stream_orders()),
            asyncio.create_task(self._stream_alerts()),
            asyncio.create_task(self._stream_portfolio_updates()),
            asyncio.create_task(self._stream_system_updates())
        ]
        
        logger.info("WebSocket streaming started")
        
        # Wait for all tasks
        try:
            await asyncio.gather(*self.stream_tasks)
        except Exception as e:
            logger.error(f"Streaming error: {e}")
        finally:
            self.is_streaming = False
    
    async def stop_streaming(self):
        """Stop all streaming tasks."""
        self.is_streaming = False
        
        for task in self.stream_tasks:
            task.cancel()
        
        await asyncio.gather(*self.stream_tasks, return_exceptions=True)
        logger.info("WebSocket streaming stopped")
    
    async def _stream_market_data(self):
        """Stream market data updates."""
        while self.is_streaming:
            try:
                await asyncio.sleep(1)  # Stream every second
                
                # Check if anyone is subscribed to market data
                if not self.manager.connections["market_data"]:
                    continue
                
                # Get current market data (mock)
                symbols = ["AAPL", "GOOGL", "TSLA", "SPY"]
                
                for symbol in symbols:
                    # Mock market data point
                    market_data = MarketDataPoint(
                        symbol=symbol,
                        timestamp=datetime.utcnow(),
                        open=150.00 + hash(symbol + str(datetime.utcnow().second)) % 10,
                        high=152.00 + hash(symbol + str(datetime.utcnow().second)) % 10,
                        low=148.00 + hash(symbol + str(datetime.utcnow().second)) % 10,
                        close=151.00 + hash(symbol + str(datetime.utcnow().second)) % 10,
                        volume=1000000 + hash(symbol) % 500000,
                        timeframe="1min",
                        data_source="stream"
                    )
                    
                    # Create WebSocket message
                    message = MarketDataUpdate(
                        data=market_data
                    ).dict()
                    
                    # Broadcast to symbol subscribers
                    await self.manager.broadcast_to_symbols([symbol], message)
                
            except Exception as e:
                logger.error(f"Market data streaming error: {e}")
                await asyncio.sleep(5)
    
    async def _stream_signals(self):
        """Stream trading signal updates."""
        while self.is_streaming:
            try:
                await asyncio.sleep(30)  # Stream every 30 seconds
                
                # Check if anyone is subscribed to signals
                if not self.manager.connections["signals"]:
                    continue
                
                # Mock signal generation
                signal = TradingSignal(
                    symbol="AAPL",
                    signal_type="BUY",
                    confidence=0.85,
                    strength=0.75,
                    price_target=155.0,
                    stop_loss=145.0,
                    take_profit=160.0,
                    strategy_name="momentum_strategy",
                    reasoning="Strong momentum with volume confirmation",
                    timestamp=datetime.utcnow(),
                    risk_level="MEDIUM"
                )
                
                # Create WebSocket message
                message = SignalUpdate(
                    data=signal
                ).dict()
                
                # Broadcast to signal subscribers
                await self.manager.broadcast_to_stream("signals", message)
                
            except Exception as e:
                logger.error(f"Signal streaming error: {e}")
                await asyncio.sleep(5)
    
    async def _stream_orders(self):
        """Stream order status updates."""
        while self.is_streaming:
            try:
                await asyncio.sleep(10)  # Stream every 10 seconds
                
                # Check if anyone is subscribed to orders
                if not self.manager.connections["orders"]:
                    continue
                
                # Mock order update (would come from order management system)
                # This is just for demonstration
                
            except Exception as e:
                logger.error(f"Order streaming error: {e}")
                await asyncio.sleep(5)
    
    async def _stream_alerts(self):
        """Stream system alerts."""
        while self.is_streaming:
            try:
                await asyncio.sleep(60)  # Stream every minute
                
                # Check if anyone is subscribed to alerts
                if not self.manager.connections["alerts"]:
                    continue
                
                # Mock alert (would come from risk monitoring system)
                # This is just for demonstration
                
            except Exception as e:
                logger.error(f"Alert streaming error: {e}")
                await asyncio.sleep(5)
    
    async def _stream_portfolio_updates(self):
        """Stream portfolio value updates."""
        while self.is_streaming:
            try:
                await asyncio.sleep(5)  # Stream every 5 seconds
                
                # Check if anyone is subscribed to portfolio
                if not self.manager.connections["portfolio"]:
                    continue
                
                # Mock portfolio update
                portfolio_update = {
                    "type": "portfolio_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "portfolio_value": 125000.00 + (hash(str(datetime.utcnow().minute)) % 1000 - 500),
                        "day_pnl": 125.50 + (hash(str(datetime.utcnow().second)) % 100 - 50),
                        "positions_count": 4,
                        "cash_balance": 25000.00
                    }
                }
                
                await self.manager.broadcast_to_stream("portfolio", portfolio_update)
                
            except Exception as e:
                logger.error(f"Portfolio streaming error: {e}")
                await asyncio.sleep(5)
    
    async def _stream_system_updates(self):
        """Stream system status updates."""
        while self.is_streaming:
            try:
                await asyncio.sleep(30)  # Stream every 30 seconds
                
                # Check if anyone is subscribed to system updates
                if not self.manager.connections["system"]:
                    continue
                
                # Mock system update
                system_update = {
                    "type": "system_status",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "status": "healthy",
                        "api_response_time": 85.2,
                        "active_connections": sum(len(conns) for conns in self.manager.connections.values()),
                        "messages_per_minute": self.manager.messages_sent / max(1, (datetime.utcnow().minute or 1))
                    }
                }
                
                await self.manager.broadcast_to_stream("system", system_update)
                
            except Exception as e:
                logger.error(f"System streaming error: {e}")
                await asyncio.sleep(5)