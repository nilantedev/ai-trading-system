#!/usr/bin/env python3
"""
Real-time Market Stream Handler
Manages WebSocket connections for ultra-low latency market data.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque

import websockets
from websockets.client import WebSocketClientProtocol
import redis.asyncio as aioredis
from prometheus_client import Counter, Gauge, Histogram
import pandas as pd

from trading_common.config import get_settings

logger = logging.getLogger(__name__)

# Metrics
ws_messages_counter = Counter('websocket_messages_total', 'Total WebSocket messages', ['provider', 'type'])
ws_latency_histogram = Histogram('websocket_latency_ms', 'WebSocket message latency', ['provider'])
ws_connections_gauge = Gauge('websocket_connections_active', 'Active WebSocket connections', ['provider'])
ws_errors_counter = Counter('websocket_errors_total', 'WebSocket errors', ['provider', 'error_type'])
data_points_counter = Counter('realtime_data_points_total', 'Total real-time data points', ['symbol', 'type'])


class StreamType(Enum):
    """Types of real-time data streams."""
    TRADES = "trades"
    QUOTES = "quotes"
    BARS = "bars"
    NEWS = "news"
    OPTIONS = "options"
    ORDERBOOK = "orderbook"


@dataclass
class StreamConfig:
    """Configuration for a real-time stream."""
    provider: str
    url: str
    api_key: str
    stream_types: List[StreamType]
    symbols: List[str]
    reconnect_attempts: int = 5
    heartbeat_interval: int = 30
    buffer_size: int = 1000


class MarketDataBuffer:
    """High-performance buffer for real-time market data."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.trades = deque(maxlen=max_size)
        self.quotes = deque(maxlen=max_size)
        self.bars = deque(maxlen=max_size)
        self.orderbook = {}  # Symbol -> OrderBook snapshot
        
    def add_trade(self, symbol: str, data: Dict):
        """Add trade to buffer."""
        self.trades.append({
            'symbol': symbol,
            'timestamp': datetime.utcnow(),
            'data': data
        })
        
    def add_quote(self, symbol: str, data: Dict):
        """Add quote to buffer."""
        self.quotes.append({
            'symbol': symbol,
            'timestamp': datetime.utcnow(),
            'data': data
        })
        
    def add_bar(self, symbol: str, data: Dict):
        """Add bar to buffer."""
        self.bars.append({
            'symbol': symbol,
            'timestamp': datetime.utcnow(),
            'data': data
        })
        
    def update_orderbook(self, symbol: str, data: Dict):
        """Update orderbook snapshot."""
        self.orderbook[symbol] = {
            'timestamp': datetime.utcnow(),
            'data': data
        }
    
    def get_latest_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get latest trades from buffer."""
        trades = list(self.trades)
        if symbol:
            trades = [t for t in trades if t['symbol'] == symbol]
        return trades[-limit:]
    
    def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """Get latest quote for symbol."""
        for quote in reversed(self.quotes):
            if quote['symbol'] == symbol:
                return quote
        return None
    
    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get current orderbook for symbol."""
        return self.orderbook.get(symbol)


class RealTimeMarketStream:
    """Manages real-time market data streaming via WebSockets."""
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        """Initialize the real-time stream handler."""
        self.settings = get_settings()
        self.redis = redis_client
        self.connections: Dict[str, WebSocketClientProtocol] = {}
        self.buffers: Dict[str, MarketDataBuffer] = {}
        self.handlers: Dict[str, List[Callable]] = {}
        self.running = False
        self.reconnect_tasks = {}
        
        # Load configuration
        self._load_configuration()
        
        logger.info("Real-time Market Stream handler initialized")
    
    def _load_configuration(self):
        """Load streaming configuration from environment."""
        self.config = {
            'polygon_enabled': os.getenv('POLYGON_WEBSOCKET_ENABLED', 'true').lower() == 'true',
            'polygon_key': os.getenv('POLYGON_API_KEY', ''),
            'alpaca_key': os.getenv('ALPACA_API_KEY', ''),
            'alpaca_secret': os.getenv('ALPACA_SECRET_KEY', ''),
            'buffer_size': int(os.getenv('MARKET_DATA_BUFFER_SIZE', '1000')),
            'batch_size': int(os.getenv('MARKET_DATA_BATCH_SIZE', '100'))
        }
        
        # WebSocket endpoints
        self.endpoints = {
            'polygon': 'wss://socket.polygon.io/stocks',
            'alpaca': 'wss://stream.data.alpaca.markets/v2/sip',
            'finnhub': 'wss://ws.finnhub.io'
        }
    
    async def connect_polygon(self, symbols: List[str]):
        """Connect to Polygon.io WebSocket stream."""
        if not self.config['polygon_enabled'] or not self.config['polygon_key']:
            logger.warning("Polygon WebSocket disabled or no API key")
            return
        
        try:
            url = f"{self.endpoints['polygon']}?apikey={self.config['polygon_key']}"
            
            async with websockets.connect(url) as websocket:
                self.connections['polygon'] = websocket
                ws_connections_gauge.labels(provider='polygon').inc()
                
                # Authenticate and subscribe
                await self._polygon_subscribe(websocket, symbols)
                
                # Create buffer
                self.buffers['polygon'] = MarketDataBuffer(self.config['buffer_size'])
                
                # Start listening
                await self._polygon_listen(websocket)
                
        except Exception as e:
            logger.error(f"Polygon WebSocket error: {e}")
            ws_errors_counter.labels(provider='polygon', error_type=type(e).__name__).inc()
            await self._schedule_reconnect('polygon', symbols)
    
    async def _polygon_subscribe(self, websocket: WebSocketClientProtocol, symbols: List[str]):
        """Subscribe to Polygon streams."""
        # Subscribe to trades, quotes, and aggregate bars
        subscriptions = []
        for symbol in symbols:
            subscriptions.extend([
                f"T.{symbol}",  # Trades
                f"Q.{symbol}",  # Quotes
                f"AM.{symbol}"  # Aggregate bars (minute)
            ])
        
        message = {
            "action": "subscribe",
            "params": ",".join(subscriptions)
        }
        
        await websocket.send(json.dumps(message))
        logger.info(f"Subscribed to Polygon streams for {len(symbols)} symbols")
    
    async def _polygon_listen(self, websocket: WebSocketClientProtocol):
        """Listen to Polygon WebSocket messages."""
        logger.info("Started listening to Polygon stream")
        
        while self.running:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=60)
                data = json.loads(message)
                
                # Process different message types
                for item in data:
                    msg_type = item.get('ev')  # Event type
                    
                    if msg_type == 'T':  # Trade
                        await self._process_trade('polygon', item)
                    elif msg_type == 'Q':  # Quote
                        await self._process_quote('polygon', item)
                    elif msg_type == 'AM':  # Aggregate minute bar
                        await self._process_bar('polygon', item)
                    
                    ws_messages_counter.labels(provider='polygon', type=msg_type or 'unknown').inc()
                
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.ping()
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Polygon WebSocket connection closed")
                break
                
            except Exception as e:
                logger.error(f"Error processing Polygon message: {e}")
                ws_errors_counter.labels(provider='polygon', error_type='processing').inc()
    
    async def connect_alpaca(self, symbols: List[str]):
        """Connect to Alpaca WebSocket stream."""
        if not self.config['alpaca_key'] or not self.config['alpaca_secret']:
            logger.warning("Alpaca credentials not configured")
            return
        
        try:
            url = self.endpoints['alpaca']
            
            async with websockets.connect(url) as websocket:
                self.connections['alpaca'] = websocket
                ws_connections_gauge.labels(provider='alpaca').inc()
                
                # Authenticate
                auth = {
                    "action": "auth",
                    "key": self.config['alpaca_key'],
                    "secret": self.config['alpaca_secret']
                }
                await websocket.send(json.dumps(auth))
                
                # Wait for authentication
                auth_response = await websocket.recv()
                auth_data = json.loads(auth_response)
                
                if auth_data[0].get('msg') == 'authenticated':
                    # Subscribe to streams
                    await self._alpaca_subscribe(websocket, symbols)
                    
                    # Create buffer
                    self.buffers['alpaca'] = MarketDataBuffer(self.config['buffer_size'])
                    
                    # Start listening
                    await self._alpaca_listen(websocket)
                else:
                    logger.error(f"Alpaca authentication failed: {auth_data}")
                    
        except Exception as e:
            logger.error(f"Alpaca WebSocket error: {e}")
            ws_errors_counter.labels(provider='alpaca', error_type=type(e).__name__).inc()
            await self._schedule_reconnect('alpaca', symbols)
    
    async def _alpaca_subscribe(self, websocket: WebSocketClientProtocol, symbols: List[str]):
        """Subscribe to Alpaca streams."""
        subscription = {
            "action": "subscribe",
            "trades": symbols,
            "quotes": symbols,
            "bars": symbols
        }
        
        await websocket.send(json.dumps(subscription))
        logger.info(f"Subscribed to Alpaca streams for {len(symbols)} symbols")
    
    async def _alpaca_listen(self, websocket: WebSocketClientProtocol):
        """Listen to Alpaca WebSocket messages."""
        logger.info("Started listening to Alpaca stream")
        
        while self.running:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=60)
                data = json.loads(message)
                
                for item in data:
                    msg_type = item.get('T')  # Message type
                    
                    if msg_type == 't':  # Trade
                        await self._process_trade('alpaca', item)
                    elif msg_type == 'q':  # Quote
                        await self._process_quote('alpaca', item)
                    elif msg_type == 'b':  # Bar
                        await self._process_bar('alpaca', item)
                    
                    ws_messages_counter.labels(provider='alpaca', type=msg_type or 'unknown').inc()
                
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.ping()
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Alpaca WebSocket connection closed")
                break
                
            except Exception as e:
                logger.error(f"Error processing Alpaca message: {e}")
                ws_errors_counter.labels(provider='alpaca', error_type='processing').inc()
    
    async def _process_trade(self, provider: str, data: Dict):
        """Process trade data from stream."""
        symbol = data.get('S') or data.get('sym')
        
        if symbol:
            # Add to buffer
            self.buffers[provider].add_trade(symbol, data)
            
            # Update metrics
            data_points_counter.labels(symbol=symbol, type='trade').inc()
            
            # Publish to Redis for other services
            if self.redis:
                await self.redis.xadd(
                    f'trades:{symbol}',
                    {
                        'provider': provider,
                        'data': json.dumps(data),
                        'timestamp': datetime.utcnow().isoformat()
                    },
                    maxlen=1000  # Keep last 1000 trades
                )
            
            # Call handlers
            await self._call_handlers('trade', symbol, data)
    
    async def _process_quote(self, provider: str, data: Dict):
        """Process quote data from stream."""
        symbol = data.get('S') or data.get('sym')
        
        if symbol:
            # Add to buffer
            self.buffers[provider].add_quote(symbol, data)
            
            # Update metrics
            data_points_counter.labels(symbol=symbol, type='quote').inc()
            
            # Check for significant spread changes (potential opportunity)
            if 'bp' in data and 'ap' in data:  # Bid/Ask prices
                spread = float(data['ap']) - float(data['bp'])
                mid_price = (float(data['ap']) + float(data['bp'])) / 2
                spread_percent = (spread / mid_price) * 100 if mid_price > 0 else 0
                
                # Alert on wide spreads (potential volatility)
                if spread_percent > 0.5:  # More than 0.5% spread
                    await self._alert_wide_spread(symbol, spread_percent)
            
            # Publish to Redis
            if self.redis:
                await self.redis.hset(
                    f'quotes:latest',
                    symbol,
                    json.dumps({
                        'provider': provider,
                        'data': data,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                )
            
            # Call handlers
            await self._call_handlers('quote', symbol, data)
    
    async def _process_bar(self, provider: str, data: Dict):
        """Process bar data from stream."""
        symbol = data.get('S') or data.get('sym')
        
        if symbol:
            # Add to buffer
            self.buffers[provider].add_bar(symbol, data)
            
            # Update metrics
            data_points_counter.labels(symbol=symbol, type='bar').inc()
            
            # Check for volume surges
            if 'v' in data:  # Volume
                await self._check_volume_surge(symbol, int(data['v']))
            
            # Publish to Redis
            if self.redis:
                await self.redis.xadd(
                    f'bars:{symbol}',
                    {
                        'provider': provider,
                        'data': json.dumps(data),
                        'timestamp': datetime.utcnow().isoformat()
                    },
                    maxlen=500  # Keep last 500 bars
                )
            
            # Call handlers
            await self._call_handlers('bar', symbol, data)
    
    async def _alert_wide_spread(self, symbol: str, spread_percent: float):
        """Alert on wide bid-ask spreads."""
        alert = {
            'type': 'wide_spread',
            'symbol': symbol,
            'spread_percent': spread_percent,
            'timestamp': datetime.utcnow().isoformat(),
            'message': f"Wide spread detected for {symbol}: {spread_percent:.2f}%"
        }
        
        if self.redis:
            await self.redis.lpush('alerts:spreads', json.dumps(alert))
            await self.redis.expire('alerts:spreads', 3600)
        
        logger.warning(f"Wide spread alert: {symbol} at {spread_percent:.2f}%")
    
    async def _check_volume_surge(self, symbol: str, volume: int):
        """Check for volume surges indicating potential moves."""
        # Get average volume from Redis
        if self.redis:
            avg_volume_str = await self.redis.hget('volume:averages', symbol)
            
            if avg_volume_str:
                avg_volume = float(avg_volume_str)
                volume_ratio = volume / avg_volume if avg_volume > 0 else 0
                
                if volume_ratio > 3.0:  # 3x average volume
                    alert = {
                        'type': 'volume_surge',
                        'symbol': symbol,
                        'volume': volume,
                        'average': avg_volume,
                        'ratio': volume_ratio,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    await self.redis.lpush('alerts:volume', json.dumps(alert))
                    await self.redis.expire('alerts:volume', 3600)
                    
                    logger.info(f"Volume surge detected: {symbol} at {volume_ratio:.1f}x average")
    
    def register_handler(self, event_type: str, handler: Callable):
        """Register a handler for specific event types."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type} events")
    
    async def _call_handlers(self, event_type: str, symbol: str, data: Dict):
        """Call registered handlers for an event."""
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                try:
                    await handler(symbol, data)
                except Exception as e:
                    logger.error(f"Handler error for {event_type}: {e}")
    
    async def _schedule_reconnect(self, provider: str, symbols: List[str]):
        """Schedule reconnection after disconnect."""
        async def reconnect():
            await asyncio.sleep(5)  # Wait 5 seconds
            logger.info(f"Attempting to reconnect to {provider}")
            
            if provider == 'polygon':
                await self.connect_polygon(symbols)
            elif provider == 'alpaca':
                await self.connect_alpaca(symbols)
        
        if provider not in self.reconnect_tasks or self.reconnect_tasks[provider].done():
            self.reconnect_tasks[provider] = asyncio.create_task(reconnect())
    
    async def start(self, symbols: List[str]):
        """Start real-time streaming for specified symbols."""
        logger.info(f"Starting real-time streams for {len(symbols)} symbols")
        self.running = True
        
        # Start connections based on configuration
        tasks = []
        
        if self.config['polygon_enabled'] and self.config['polygon_key']:
            tasks.append(asyncio.create_task(self.connect_polygon(symbols)))
        
        if self.config['alpaca_key'] and self.config['alpaca_secret']:
            tasks.append(asyncio.create_task(self.connect_alpaca(symbols)))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            logger.warning("No streaming providers configured")
    
    async def stop(self):
        """Stop all real-time streams."""
        logger.info("Stopping real-time streams")
        self.running = False
        
        # Close all connections
        for provider, connection in self.connections.items():
            try:
                await connection.close()
                ws_connections_gauge.labels(provider=provider).dec()
            except Exception as e:
                logger.error(f"Error closing {provider} connection: {e}")
        
        # Cancel reconnect tasks
        for task in self.reconnect_tasks.values():
            if not task.done():
                task.cancel()
        
        logger.info("Real-time streams stopped")
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about buffered data."""
        stats = {}
        
        for provider, buffer in self.buffers.items():
            stats[provider] = {
                'trades': len(buffer.trades),
                'quotes': len(buffer.quotes),
                'bars': len(buffer.bars),
                'orderbook_symbols': len(buffer.orderbook)
            }
        
        return stats


# Example usage
async def main():
    """Example usage of real-time market stream."""
    stream = RealTimeMarketStream()
    
    # Define symbols to track
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'SPY']
    
    # Register a handler for trades
    async def trade_handler(symbol: str, data: Dict):
        logger.info(f"Trade for {symbol}: {data}")
    
    stream.register_handler('trade', trade_handler)
    
    # Start streaming
    await stream.start(symbols)
    
    # Run for a while
    await asyncio.sleep(60)
    
    # Get buffer stats
    stats = stream.get_buffer_stats()
    logger.info(f"Buffer stats: {stats}")
    
    # Stop streaming
    await stream.stop()


if __name__ == "__main__":
    asyncio.run(main())