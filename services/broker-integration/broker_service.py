#!/usr/bin/env python3
"""Broker Integration Service - Multi-broker API integration for trade execution."""

import asyncio
import json
import logging
import aiohttp
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod

from trading_common import get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.messaging import get_message_consumer, get_message_producer

logger = get_logger(__name__)
settings = get_settings()


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"


class BrokerType(Enum):
    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "interactive_brokers"
    TD_AMERITRADE = "td_ameritrade"
    PAPER_TRADING = "paper_trading"


@dataclass
class Order:
    """Trading order representation."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    broker: Optional[str] = None
    broker_order_id: Optional[str] = None


@dataclass
class Position:
    """Position representation."""
    symbol: str
    quantity: float
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_pl_percent: float
    current_price: float
    last_updated: datetime


@dataclass
class Account:
    """Account information."""
    account_id: str
    broker: str
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    day_trade_buying_power: float
    pattern_day_trader: bool
    trade_suspended: bool
    account_blocked: bool
    last_updated: datetime


class BrokerAPI(ABC):
    """Abstract base class for broker API implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = asyncio.Semaphore(10)  # 10 concurrent requests
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
    async def initialize(self):
        """Initialize the broker API connection."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'AI-Trading-System/1.0'}
        )
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Apply rate limiting."""
        async with self.rate_limiter:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last)
            
            self.last_request_time = time.time()
    
    @abstractmethod
    async def get_account(self) -> Account:
        """Get account information."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Place a trading order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order details."""
        pass
    
    @abstractmethod
    async def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get orders with optional status filter."""
        pass


class AlpacaAPI(BrokerAPI):
    """Alpaca broker API implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', 'https://paper-api.alpaca.markets')
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        
    async def initialize(self):
        """Initialize Alpaca API."""
        await super().initialize()
        
        # Set authentication headers
        if self.session:
            self.session.headers.update({
                'APCA-API-KEY-ID': self.api_key,
                'APCA-API-SECRET-KEY': self.secret_key
            })
    
    async def get_account(self) -> Account:
        """Get Alpaca account information."""
        await self._rate_limit()
        
        try:
            async with self.session.get(f"{self.base_url}/v2/account") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return Account(
                        account_id=data['id'],
                        broker='alpaca',
                        equity=float(data['equity']),
                        cash=float(data['cash']),
                        buying_power=float(data['buying_power']),
                        portfolio_value=float(data['portfolio_value']),
                        day_trade_buying_power=float(data['daytrading_buying_power']),
                        pattern_day_trader=data['pattern_day_trader'],
                        trade_suspended=data['trade_suspended_by_user'],
                        account_blocked=data['account_blocked'],
                        last_updated=datetime.utcnow()
                    )
                else:
                    raise Exception(f"Alpaca API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Failed to get Alpaca account: {e}")
            raise
    
    async def get_positions(self) -> List[Position]:
        """Get Alpaca positions."""
        await self._rate_limit()
        
        try:
            async with self.session.get(f"{self.base_url}/v2/positions") as response:
                if response.status == 200:
                    data = await response.json()
                    positions = []
                    
                    for pos_data in data:
                        position = Position(
                            symbol=pos_data['symbol'],
                            quantity=float(pos_data['qty']),
                            market_value=float(pos_data['market_value']),
                            cost_basis=float(pos_data['cost_basis']),
                            unrealized_pl=float(pos_data['unrealized_pl']),
                            unrealized_pl_percent=float(pos_data['unrealized_plpc']),
                            current_price=float(pos_data['current_price']),
                            last_updated=datetime.utcnow()
                        )
                        positions.append(position)
                    
                    return positions
                else:
                    raise Exception(f"Alpaca API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Failed to get Alpaca positions: {e}")
            return []
    
    async def place_order(self, order: Order) -> Order:
        """Place order with Alpaca."""
        await self._rate_limit()
        
        try:
            order_data = {
                'symbol': order.symbol,
                'qty': str(order.quantity),
                'side': order.side.value,
                'type': order.order_type.value,
                'time_in_force': order.time_in_force
            }
            
            if order.price:
                order_data['limit_price'] = str(order.price)
            
            if order.stop_price:
                order_data['stop_price'] = str(order.stop_price)
            
            async with self.session.post(
                f"{self.base_url}/v2/orders",
                json=order_data
            ) as response:
                if response.status == 201:
                    data = await response.json()
                    
                    order.broker_order_id = data['id']
                    order.status = OrderStatus(data['status'])
                    order.created_at = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
                    order.broker = 'alpaca'
                    
                    return order
                else:
                    error_data = await response.json()
                    raise Exception(f"Alpaca order failed: {error_data}")
                    
        except Exception as e:
            logger.error(f"Failed to place Alpaca order: {e}")
            order.status = OrderStatus.REJECTED
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel Alpaca order."""
        await self._rate_limit()
        
        try:
            async with self.session.delete(f"{self.base_url}/v2/orders/{order_id}") as response:
                return response.status == 204
                
        except Exception as e:
            logger.error(f"Failed to cancel Alpaca order {order_id}: {e}")
            return False
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get Alpaca order details."""
        await self._rate_limit()
        
        try:
            async with self.session.get(f"{self.base_url}/v2/orders/{order_id}") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return Order(
                        order_id=order_id,
                        broker_order_id=data['id'],
                        symbol=data['symbol'],
                        side=OrderSide(data['side']),
                        order_type=OrderType(data['order_type']),
                        quantity=float(data['qty']),
                        price=float(data['limit_price']) if data['limit_price'] else None,
                        stop_price=float(data['stop_price']) if data['stop_price'] else None,
                        time_in_force=data['time_in_force'],
                        status=OrderStatus(data['status']),
                        filled_quantity=float(data['filled_qty']),
                        filled_price=float(data['filled_avg_price']) if data['filled_avg_price'] else None,
                        created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
                        updated_at=datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00')),
                        broker='alpaca'
                    )
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get Alpaca order {order_id}: {e}")
            return None
    
    async def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get Alpaca orders."""
        await self._rate_limit()
        
        try:
            params = {}
            if status:
                params['status'] = status.value
            
            async with self.session.get(f"{self.base_url}/v2/orders", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    orders = []
                    
                    for order_data in data:
                        order = Order(
                            order_id=order_data['id'],
                            broker_order_id=order_data['id'],
                            symbol=order_data['symbol'],
                            side=OrderSide(order_data['side']),
                            order_type=OrderType(order_data['order_type']),
                            quantity=float(order_data['qty']),
                            price=float(order_data['limit_price']) if order_data['limit_price'] else None,
                            stop_price=float(order_data['stop_price']) if order_data['stop_price'] else None,
                            time_in_force=order_data['time_in_force'],
                            status=OrderStatus(order_data['status']),
                            filled_quantity=float(order_data['filled_qty']),
                            filled_price=float(order_data['filled_avg_price']) if order_data['filled_avg_price'] else None,
                            created_at=datetime.fromisoformat(order_data['created_at'].replace('Z', '+00:00')),
                            updated_at=datetime.fromisoformat(order_data['updated_at'].replace('Z', '+00:00')),
                            broker='alpaca'
                        )
                        orders.append(order)
                    
                    return orders
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"Failed to get Alpaca orders: {e}")
            return []


class PaperTradingAPI(BrokerAPI):
    """Paper trading simulation API."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.account_data = {
            'equity': 100000.0,
            'cash': 100000.0,
            'buying_power': 400000.0,  # 4x leverage
            'portfolio_value': 100000.0
        }
        self.positions = {}
        self.orders = {}
        self.order_counter = 1
    
    async def get_account(self) -> Account:
        """Get paper trading account."""
        return Account(
            account_id="paper_account_1",
            broker="paper_trading",
            equity=self.account_data['equity'],
            cash=self.account_data['cash'],
            buying_power=self.account_data['buying_power'],
            portfolio_value=self.account_data['portfolio_value'],
            day_trade_buying_power=self.account_data['buying_power'],
            pattern_day_trader=False,
            trade_suspended=False,
            account_blocked=False,
            last_updated=datetime.utcnow()
        )
    
    async def get_positions(self) -> List[Position]:
        """Get paper trading positions."""
        positions = []
        for symbol, pos_data in self.positions.items():
            position = Position(
                symbol=symbol,
                quantity=pos_data['quantity'],
                market_value=pos_data['market_value'],
                cost_basis=pos_data['cost_basis'],
                unrealized_pl=pos_data['unrealized_pl'],
                unrealized_pl_percent=pos_data['unrealized_pl_percent'],
                current_price=pos_data['current_price'],
                last_updated=datetime.utcnow()
            )
            positions.append(position)
        return positions
    
    async def place_order(self, order: Order) -> Order:
        """Simulate order placement."""
        order.broker_order_id = f"paper_{self.order_counter}"
        order.status = OrderStatus.FILLED  # Instant fill for simulation
        order.filled_quantity = order.quantity
        order.filled_price = order.price if order.price else 100.0  # Mock price
        order.created_at = datetime.utcnow()
        order.updated_at = datetime.utcnow()
        order.broker = 'paper_trading'
        
        self.orders[order.broker_order_id] = order
        self.order_counter += 1
        
        # Update positions
        await self._update_position(order)
        
        return order
    
    async def _update_position(self, order: Order):
        """Update position after order fill."""
        symbol = order.symbol
        quantity_delta = order.filled_quantity if order.side == OrderSide.BUY else -order.filled_quantity
        
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0.0,
                'cost_basis': 0.0,
                'market_value': 0.0,
                'unrealized_pl': 0.0,
                'unrealized_pl_percent': 0.0,
                'current_price': order.filled_price
            }
        
        pos = self.positions[symbol]
        
        # Update quantity
        new_quantity = pos['quantity'] + quantity_delta
        
        if new_quantity == 0:
            # Position closed
            del self.positions[symbol]
        else:
            # Update cost basis
            if (pos['quantity'] >= 0 and quantity_delta > 0) or (pos['quantity'] < 0 and quantity_delta < 0):
                # Adding to position
                total_cost = pos['cost_basis'] * pos['quantity'] + order.filled_price * quantity_delta
                pos['cost_basis'] = total_cost / new_quantity
            
            pos['quantity'] = new_quantity
            pos['current_price'] = order.filled_price
            pos['market_value'] = new_quantity * order.filled_price
            pos['unrealized_pl'] = pos['market_value'] - (pos['cost_basis'] * new_quantity)
            
            if pos['cost_basis'] * new_quantity != 0:
                pos['unrealized_pl_percent'] = pos['unrealized_pl'] / (pos['cost_basis'] * new_quantity)
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel paper trading order."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.PENDING, OrderStatus.OPEN]:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.utcnow()
                return True
        return False
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get paper trading order."""
        return self.orders.get(order_id)
    
    async def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get paper trading orders."""
        orders = list(self.orders.values())
        if status:
            orders = [o for o in orders if o.status == status]
        return orders


class BrokerService:
    """Service for managing multiple broker integrations."""
    
    def __init__(self):
        self.consumer = None
        self.producer = None
        self.cache = None
        
        self.brokers: Dict[str, BrokerAPI] = {}
        self.is_running = False
        
        # Order tracking
        self.active_orders = {}  # order_id -> Order
        self.order_counter = 1
        
        # Performance metrics
        self.orders_placed = 0
        self.orders_filled = 0
        self.orders_cancelled = 0
        self.api_errors = 0
        
        # Message queues
        self.order_queue = asyncio.Queue(maxsize=1000)
        
    async def start(self):
        """Initialize and start broker service."""
        logger.info("Starting Broker Integration Service")
        
        try:
            # Initialize connections
            self.consumer = await get_message_consumer()
            self.producer = await get_message_producer()
            self.cache = get_trading_cache()
            
            # Initialize brokers
            await self._initialize_brokers()
            
            # Subscribe to trading signals
            await self._setup_subscriptions()
            
            # Start processing tasks
            self.is_running = True
            
            tasks = [
                asyncio.create_task(self._process_order_queue()),
                asyncio.create_task(self._consume_messages()),
                asyncio.create_task(self._monitor_orders()),
                asyncio.create_task(self._sync_positions()),
                asyncio.create_task(self._health_check())
            ]
            
            logger.info("Broker service started with 5 concurrent tasks")
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Failed to start broker service: {e}")
            raise
    
    async def stop(self):
        """Stop broker service gracefully."""
        logger.info("Stopping Broker Integration Service")
        self.is_running = False
        
        # Clean up broker connections
        for broker in self.brokers.values():
            await broker.cleanup()
        
        if self.consumer:
            await self.consumer.close()
        if self.producer:
            await self.producer.close()
        
        logger.info("Broker Integration Service stopped")
    
    async def _initialize_brokers(self):
        """Initialize configured brokers."""
        try:
            broker_configs = settings.get('brokers', {})
            
            for broker_name, config in broker_configs.items():
                broker_type = BrokerType(config.get('type', 'paper_trading'))
                
                if broker_type == BrokerType.ALPACA:
                    broker = AlpacaAPI(config)
                elif broker_type == BrokerType.PAPER_TRADING:
                    broker = PaperTradingAPI(config)
                else:
                    logger.warning(f"Unsupported broker type: {broker_type}")
                    continue
                
                await broker.initialize()
                self.brokers[broker_name] = broker
                
                logger.info(f"Initialized {broker_type.value} broker: {broker_name}")
            
            if not self.brokers:
                # Initialize default paper trading broker
                paper_broker = PaperTradingAPI({'type': 'paper_trading'})
                await paper_broker.initialize()
                self.brokers['paper'] = paper_broker
                logger.info("Initialized default paper trading broker")
                
        except Exception as e:
            logger.error(f"Failed to initialize brokers: {e}")
            raise
    
    async def _setup_subscriptions(self):
        """Subscribe to trading signals and order management."""
        try:
            await self.consumer.subscribe_trading_signals(
                self._handle_trading_signal,
                subscription_name="broker-service-signals"
            )
            
            await self.consumer.subscribe_order_requests(
                self._handle_order_request,
                subscription_name="broker-service-orders"
            )
            
            logger.info("Subscribed to trading signals and order requests")
        except Exception as e:
            logger.warning(f"Subscription setup failed: {e}")
    
    async def _consume_messages(self):
        """Consume messages from subscribed topics."""
        try:
            await self.consumer.start_consuming()
        except Exception as e:
            logger.error(f"Message consumption failed: {e}")
    
    async def _handle_trading_signal(self, message):
        """Handle incoming trading signal."""
        try:
            signal_data = json.loads(message) if isinstance(message, str) else message
            
            # Convert signal to order request
            order_request = await self._signal_to_order(signal_data)
            
            if order_request:
                await self.order_queue.put(order_request)
                
        except Exception as e:
            logger.error(f"Failed to handle trading signal: {e}")
    
    async def _handle_order_request(self, message):
        """Handle direct order request."""
        try:
            order_data = json.loads(message) if isinstance(message, str) else message
            await self.order_queue.put(order_data)
        except Exception as e:
            logger.error(f"Failed to handle order request: {e}")
    
    async def _signal_to_order(self, signal_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert trading signal to order request."""
        try:
            action = signal_data.get('recommended_action')
            if action not in ['BUY', 'SELL']:
                return None
            
            symbol = signal_data.get('symbol')
            position_size = signal_data.get('position_size', 0.01)  # Default 1% position
            
            # Calculate order quantity (simplified)
            # Would integrate with portfolio manager for proper sizing
            account = await self.get_account()
            if account:
                order_value = account.portfolio_value * position_size
                # Would get current market price to calculate quantity
                estimated_price = 100.0  # Mock price
                quantity = order_value / estimated_price
                
                return {
                    'symbol': symbol,
                    'side': action.lower(),
                    'order_type': 'market',
                    'quantity': quantity,
                    'signal_id': signal_data.get('signal_id'),
                    'confidence': signal_data.get('confidence', 0.5)
                }
            
        except Exception as e:
            logger.error(f"Failed to convert signal to order: {e}")
        
        return None
    
    async def _process_order_queue(self):
        """Process order requests."""
        while self.is_running:
            try:
                # Wait for order request
                order_data = await asyncio.wait_for(
                    self.order_queue.get(),
                    timeout=1.0
                )
                
                # Create order object
                order = Order(
                    order_id=f"order_{self.order_counter}",
                    symbol=order_data['symbol'],
                    side=OrderSide(order_data['side']),
                    order_type=OrderType(order_data.get('order_type', 'market')),
                    quantity=order_data['quantity'],
                    price=order_data.get('price'),
                    stop_price=order_data.get('stop_price'),
                    time_in_force=order_data.get('time_in_force', 'DAY')
                )
                
                self.order_counter += 1
                
                # Place order with primary broker
                result = await self._place_order_with_broker(order)
                
                if result:
                    self.active_orders[order.order_id] = order
                    await self._cache_order(order)
                    await self._publish_order_status(order)
                    self.orders_placed += 1
                
                self.order_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Order processing error: {e}")
                self.api_errors += 1
    
    async def _place_order_with_broker(self, order: Order) -> bool:
        """Place order with the appropriate broker."""
        try:
            # Use first available broker (would implement broker selection logic)
            broker_name = next(iter(self.brokers.keys()))
            broker = self.brokers[broker_name]
            
            result_order = await broker.place_order(order)
            
            # Update order with broker response
            order.broker_order_id = result_order.broker_order_id
            order.status = result_order.status
            order.created_at = result_order.created_at
            order.broker = result_order.broker
            
            logger.info(f"Order placed: {order.symbol} {order.side.value} {order.quantity} via {broker_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            order.status = OrderStatus.REJECTED
            return False
    
    async def _monitor_orders(self):
        """Monitor active orders for status updates."""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                for order_id, order in list(self.active_orders.items()):
                    if order.status in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
                        # Check order status with broker
                        updated_order = await self._get_order_from_broker(order)
                        
                        if updated_order and updated_order.status != order.status:
                            # Status changed
                            order.status = updated_order.status
                            order.filled_quantity = updated_order.filled_quantity
                            order.filled_price = updated_order.filled_price
                            order.updated_at = datetime.utcnow()
                            
                            await self._cache_order(order)
                            await self._publish_order_status(order)
                            
                            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                                # Remove from active orders
                                del self.active_orders[order_id]
                                
                                if order.status == OrderStatus.FILLED:
                                    self.orders_filled += 1
                                elif order.status == OrderStatus.CANCELLED:
                                    self.orders_cancelled += 1
                
            except Exception as e:
                logger.warning(f"Order monitoring error: {e}")
    
    async def _get_order_from_broker(self, order: Order) -> Optional[Order]:
        """Get updated order details from broker."""
        try:
            if order.broker and order.broker_order_id:
                broker = self.brokers.get(order.broker.replace('_', ''))  # Handle broker name mapping
                if broker:
                    return await broker.get_order(order.broker_order_id)
        except Exception as e:
            logger.warning(f"Failed to get order from broker: {e}")
        return None
    
    async def _sync_positions(self):
        """Sync positions from all brokers."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Sync every minute
                
                all_positions = {}
                
                for broker_name, broker in self.brokers.items():
                    try:
                        positions = await broker.get_positions()
                        for position in positions:
                            key = f"{broker_name}_{position.symbol}"
                            all_positions[key] = position
                    except Exception as e:
                        logger.warning(f"Failed to get positions from {broker_name}: {e}")
                
                # Cache aggregated positions
                await self._cache_positions(all_positions)
                
            except Exception as e:
                logger.warning(f"Position sync error: {e}")
    
    async def _health_check(self):
        """Perform periodic health checks on broker connections."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                for broker_name, broker in self.brokers.items():
                    try:
                        # Test broker connection by getting account info
                        account = await broker.get_account()
                        if account:
                            logger.debug(f"Broker {broker_name} health check: OK")
                        else:
                            logger.warning(f"Broker {broker_name} health check: Failed")
                    except Exception as e:
                        logger.error(f"Broker {broker_name} health check error: {e}")
                        self.api_errors += 1
                
            except Exception as e:
                logger.warning(f"Health check error: {e}")
    
    async def _cache_order(self, order: Order):
        """Cache order information."""
        try:
            if self.cache:
                cache_key = f"order:{order.order_id}"
                order_data = asdict(order)
                
                # Convert datetime fields to ISO format
                if order.created_at:
                    order_data['created_at'] = order.created_at.isoformat()
                if order.updated_at:
                    order_data['updated_at'] = order.updated_at.isoformat()
                
                # Convert enums to values
                order_data['side'] = order.side.value
                order_data['order_type'] = order.order_type.value
                order_data['status'] = order.status.value
                
                await self.cache.set_json(cache_key, order_data, ttl=86400)  # 24 hours
        except Exception as e:
            logger.warning(f"Failed to cache order: {e}")
    
    async def _cache_positions(self, positions: Dict[str, Position]):
        """Cache position information."""
        try:
            if self.cache:
                cache_key = "positions:all"
                positions_data = {}
                
                for key, position in positions.items():
                    pos_data = asdict(position)
                    pos_data['last_updated'] = position.last_updated.isoformat()
                    positions_data[key] = pos_data
                
                await self.cache.set_json(cache_key, positions_data, ttl=300)  # 5 minutes
        except Exception as e:
            logger.warning(f"Failed to cache positions: {e}")
    
    async def _publish_order_status(self, order: Order):
        """Publish order status update."""
        try:
            if self.producer:
                order_message = {
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'status': order.status.value,
                    'quantity': order.quantity,
                    'filled_quantity': order.filled_quantity,
                    'filled_price': order.filled_price,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Would publish to order status topic
                logger.debug(f"Publishing order status: {order.order_id} - {order.status.value}")
                
        except Exception as e:
            logger.warning(f"Failed to publish order status: {e}")
    
    async def get_account(self, broker_name: Optional[str] = None) -> Optional[Account]:
        """Get account information from specified or primary broker."""
        try:
            if broker_name and broker_name in self.brokers:
                broker = self.brokers[broker_name]
            else:
                # Use first available broker
                broker = next(iter(self.brokers.values()))
            
            return await broker.get_account()
            
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return None
    
    async def get_positions(self, broker_name: Optional[str] = None) -> List[Position]:
        """Get positions from specified or all brokers."""
        try:
            if broker_name and broker_name in self.brokers:
                return await self.brokers[broker_name].get_positions()
            else:
                # Get positions from all brokers
                all_positions = []
                for broker in self.brokers.values():
                    positions = await broker.get_positions()
                    all_positions.extend(positions)
                return all_positions
                
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                
                # Find broker and cancel
                for broker_name, broker in self.brokers.items():
                    if order.broker and broker_name in order.broker:
                        result = await broker.cancel_order(order.broker_order_id)
                        
                        if result:
                            order.status = OrderStatus.CANCELLED
                            order.updated_at = datetime.utcnow()
                            
                            await self._cache_order(order)
                            await self._publish_order_status(order)
                            
                            del self.active_orders[order_id]
                            self.orders_cancelled += 1
                            
                            return True
                        break
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health status."""
        broker_status = {}
        for broker_name in self.brokers.keys():
            try:
                account = await self.get_account(broker_name)
                broker_status[broker_name] = 'connected' if account else 'disconnected'
            except Exception:
                broker_status[broker_name] = 'error'
        
        return {
            'service': 'broker_integration_service',
            'status': 'healthy' if self.is_running else 'stopped',
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {
                'orders_placed': self.orders_placed,
                'orders_filled': self.orders_filled,
                'orders_cancelled': self.orders_cancelled,
                'api_errors': self.api_errors,
                'active_orders': len(self.active_orders)
            },
            'brokers': broker_status,
            'connections': {
                'consumer': self.consumer is not None,
                'producer': self.producer is not None,
                'cache': self.cache is not None
            }
        }


# Global service instance
broker_service: Optional[BrokerService] = None


async def get_broker_service() -> BrokerService:
    """Get or create broker service instance."""
    global broker_service
    if broker_service is None:
        broker_service = BrokerService()
    return broker_service