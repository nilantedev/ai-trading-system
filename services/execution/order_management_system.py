#!/usr/bin/env python3
"""Order Management System - Centralized order processing and execution."""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import time

from trading_common import get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.messaging import get_message_consumer, get_message_producer
from trading_common.resilience import CircuitBreaker, CircuitBreakerConfig, RetryStrategy

logger = get_logger(__name__)
settings = get_settings()


class OrderStatus(Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(Enum):
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill


class RejectionReason(Enum):
    INSUFFICIENT_FUNDS = "insufficient_funds"
    POSITION_LIMIT = "position_limit"
    RISK_LIMIT = "risk_limit"
    INVALID_SYMBOL = "invalid_symbol"
    MARKET_CLOSED = "market_closed"
    SYSTEM_ERROR = "system_error"
    VALIDATION_FAILED = "validation_failed"


@dataclass
class OrderRequest:
    """Order request from trading signals or manual input."""
    request_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    client_order_id: Optional[str] = None
    source: str = "system"  # system, manual, api
    strategy_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    requested_at: datetime = None


@dataclass
class Order:
    """Complete order representation."""
    order_id: str
    client_order_id: Optional[str]
    broker_order_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    time_in_force: TimeInForce
    status: OrderStatus
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    
    # Timing
    created_at: datetime = None
    submitted_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Metadata
    source: str = "system"
    strategy_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    
    # Risk and validation
    risk_validated: bool = False
    rejection_reason: Optional[RejectionReason] = None
    
    # Execution tracking
    fills: List[Dict[str, Any]] = None
    commission: float = 0.0
    
    def __post_init__(self):
        if self.fills is None:
            self.fills = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.quantity


@dataclass
class Fill:
    """Order fill execution."""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    exchange: Optional[str] = None
    execution_id: Optional[str] = None


@dataclass
class RiskCheck:
    """Risk validation result."""
    passed: bool
    risk_score: float
    checks_performed: List[str]
    failures: List[str]
    warnings: List[str]
    recommendations: List[str]


class OrderValidator:
    """Order validation and risk checking."""
    
    def __init__(self, risk_service=None, portfolio_service=None):
        self.risk_service = risk_service
        self.portfolio_service = portfolio_service
        
        # Risk limits
        self.max_order_value = settings.get('risk_limits', {}).get('max_order_value', 100000)
        self.max_position_size = settings.get('risk_limits', {}).get('max_position_size', 0.1)
        self.max_daily_loss = settings.get('risk_limits', {}).get('max_daily_loss', 0.02)
        
    async def validate_order(self, order_request: OrderRequest) -> RiskCheck:
        """Perform comprehensive order validation."""
        checks_performed = []
        failures = []
        warnings = []
        recommendations = []
        risk_score = 0.0
        
        try:
            # Basic validation
            if not order_request.symbol or len(order_request.symbol) < 1:
                failures.append("Invalid symbol")
            checks_performed.append("symbol_validation")
            
            if order_request.quantity <= 0:
                failures.append("Invalid quantity")
            checks_performed.append("quantity_validation")
            
            if order_request.order_type == OrderType.LIMIT and not order_request.price:
                failures.append("Limit order requires price")
            checks_performed.append("price_validation")
            
            # Market hours check (simplified)
            current_time = datetime.utcnow()
            market_open = current_time.replace(hour=14, minute=30, second=0, microsecond=0)  # 9:30 AM EST
            market_close = current_time.replace(hour=21, minute=0, second=0, microsecond=0)  # 4:00 PM EST
            
            if not (market_open <= current_time <= market_close):
                warnings.append("Market is closed - order will be queued")
            checks_performed.append("market_hours")
            
            # Order value check
            estimated_value = order_request.quantity * (order_request.price or 100.0)  # Mock price if not provided
            if estimated_value > self.max_order_value:
                failures.append(f"Order value ${estimated_value:,.2f} exceeds limit ${self.max_order_value:,.2f}")
                risk_score += 0.3
            checks_performed.append("order_value")
            
            # Position size check
            if self.portfolio_service:
                try:
                    account = await self.portfolio_service.get_account()
                    if account:
                        position_value = estimated_value
                        position_percentage = position_value / account.portfolio_value
                        
                        if position_percentage > self.max_position_size:
                            failures.append(f"Position size {position_percentage:.2%} exceeds limit {self.max_position_size:.2%}")
                            risk_score += 0.4
                        elif position_percentage > self.max_position_size * 0.8:
                            warnings.append(f"Large position size: {position_percentage:.2%}")
                            risk_score += 0.2
                            
                        checks_performed.append("position_size")
                except Exception as e:
                    warnings.append("Could not validate position size")
            
            # Buying power check
            if self.portfolio_service and order_request.side == OrderSide.BUY:
                try:
                    account = await self.portfolio_service.get_account()
                    if account and estimated_value > account.buying_power:
                        failures.append("Insufficient buying power")
                        risk_score += 0.5
                    checks_performed.append("buying_power")
                except Exception as e:
                    warnings.append("Could not validate buying power")
            
            # Risk service validation
            if self.risk_service:
                try:
                    symbol_risk = await self.risk_service.get_risk_metrics(order_request.symbol)
                    if symbol_risk:
                        if symbol_risk.risk_level.value == "CRITICAL":
                            warnings.append("Symbol has critical risk level")
                            risk_score += 0.3
                        elif symbol_risk.risk_level.value == "HIGH":
                            warnings.append("Symbol has high risk level")
                            risk_score += 0.2
                            
                        checks_performed.append("symbol_risk")
                except Exception as e:
                    warnings.append("Could not validate symbol risk")
            
            # Generate recommendations
            if risk_score > 0.5:
                recommendations.append("Consider reducing position size")
            if warnings:
                recommendations.append("Review all warnings before proceeding")
            if order_request.order_type == OrderType.MARKET:
                recommendations.append("Consider using limit order for better price control")
            
            # Overall risk assessment
            if risk_score > 0.7:
                risk_score = 0.8  # Cap at high risk
            
            passed = len(failures) == 0
            
            return RiskCheck(
                passed=passed,
                risk_score=risk_score,
                checks_performed=checks_performed,
                failures=failures,
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Order validation error: {e}")
            return RiskCheck(
                passed=False,
                risk_score=1.0,
                checks_performed=checks_performed,
                failures=[f"Validation error: {str(e)}"],
                warnings=[],
                recommendations=["Contact system administrator"]
            )


class OrderManagementSystem:
    """Centralized order management and execution system."""
    
    def __init__(self):
        self.consumer = None
        self.producer = None
        self.cache = None
        
        self.is_running = False
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.order_requests: Dict[str, OrderRequest] = {}
        self.active_orders: Dict[str, Order] = {}  # Orders that can still be modified/cancelled
        
        # Processing queues
        self.order_request_queue = asyncio.Queue(maxsize=10000)
        self.order_update_queue = asyncio.Queue(maxsize=10000)
        self.fill_queue = asyncio.Queue(maxsize=10000)
        
        # Services
        self.validator = OrderValidator()
        self.broker_service = None
        self.risk_service = None
        self.portfolio_service = None
        
        # Performance metrics
        self.orders_processed = 0
        self.orders_filled = 0
        self.orders_rejected = 0
        
        # Resilience patterns
        self.circuit_breakers = {
            'broker_service': CircuitBreaker(
                name='broker_service',
                config=CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout=30,
                    success_threshold=2
                )
            ),
            'risk_service': CircuitBreaker(
                name='risk_service',
                config=CircuitBreakerConfig(
                    failure_threshold=3,
                    recovery_timeout=15,
                    success_threshold=2
                )
            ),
            'validation_service': CircuitBreaker(
                name='validation_service',
                config=CircuitBreakerConfig(
                    failure_threshold=10,
                    recovery_timeout=10,
                    success_threshold=3
                )
            )
        }
        
        self.retry_strategy = RetryStrategy(
            max_attempts=3,
            base_delay=0.5,
            max_delay=5.0,
            exponential_base=2.0
        )
        self.total_fill_value = 0.0
        
        # Order ID generation
        self.order_counter = 1
        
    async def start(self):
        """Initialize and start order management system."""
        logger.info("Starting Order Management System")
        
        try:
            # Initialize connections
            self.consumer = await get_message_consumer()
            self.producer = await get_message_producer()
            self.cache = get_trading_cache()
            
            # Initialize services
            await self._initialize_services()
            
            # Subscribe to order flows
            await self._setup_subscriptions()
            
            # Start processing tasks
            self.is_running = True
            
            tasks = [
                asyncio.create_task(self._process_order_requests()),
                asyncio.create_task(self._process_order_updates()),
                asyncio.create_task(self._process_fills()),
                asyncio.create_task(self._consume_messages()),
                asyncio.create_task(self._monitor_orders()),
                asyncio.create_task(self._periodic_cleanup())
            ]
            
            logger.info("Order management system started with 6 concurrent tasks")
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Failed to start order management system: {e}")
            raise
    
    async def stop(self):
        """Stop order management system gracefully."""
        logger.info("Stopping Order Management System")
        self.is_running = False
        
        if self.consumer:
            await self.consumer.close()
        if self.producer:
            await self.producer.close()
        
        logger.info("Order Management System stopped")
    
    async def _initialize_services(self):
        """Initialize dependent services."""
        try:
            # Import services dynamically to avoid circular imports
            from broker_service import get_broker_service
            from risk_monitoring_service import get_risk_service
            # from portfolio_service import get_portfolio_service  # Would be implemented
            
            self.broker_service = await get_broker_service()
            self.risk_service = await get_risk_service()
            # self.portfolio_service = await get_portfolio_service()
            
            # Update validator with services
            self.validator = OrderValidator(
                risk_service=self.risk_service,
                portfolio_service=self.portfolio_service
            )
            
            logger.info("Initialized dependent services")
            
        except Exception as e:
            logger.warning(f"Failed to initialize some services: {e}")
    
    async def _setup_subscriptions(self):
        """Subscribe to order-related message streams."""
        try:
            await self.consumer.subscribe_trading_signals(
                self._handle_trading_signal,
                subscription_name="oms-trading-signals"
            )
            
            await self.consumer.subscribe_order_requests(
                self._handle_order_request,
                subscription_name="oms-order-requests"
            )
            
            await self.consumer.subscribe_broker_updates(
                self._handle_broker_update,
                subscription_name="oms-broker-updates"
            )
            
            logger.info("Subscribed to order management streams")
        except Exception as e:
            logger.warning(f"Order subscription setup failed: {e}")
    
    async def _consume_messages(self):
        """Consume messages from subscribed topics."""
        try:
            await self.consumer.start_consuming()
        except Exception as e:
            logger.error(f"Message consumption failed: {e}")
    
    async def _handle_trading_signal(self, message):
        """Handle trading signal and convert to order request."""
        try:
            signal_data = json.loads(message) if isinstance(message, str) else message
            
            # Convert signal to order request
            order_request = self._signal_to_order_request(signal_data)
            
            if order_request:
                await self.order_request_queue.put(order_request)
                
        except Exception as e:
            logger.error(f"Failed to handle trading signal: {e}")
    
    async def _handle_order_request(self, message):
        """Handle direct order request."""
        try:
            request_data = json.loads(message) if isinstance(message, str) else message
            
            order_request = OrderRequest(
                request_id=request_data.get('request_id', str(uuid.uuid4())),
                symbol=request_data['symbol'],
                side=OrderSide(request_data['side']),
                order_type=OrderType(request_data.get('order_type', 'market')),
                quantity=float(request_data['quantity']),
                price=float(request_data['price']) if request_data.get('price') else None,
                stop_price=float(request_data['stop_price']) if request_data.get('stop_price') else None,
                time_in_force=TimeInForce(request_data.get('time_in_force', 'day')),
                client_order_id=request_data.get('client_order_id'),
                source=request_data.get('source', 'api'),
                strategy_id=request_data.get('strategy_id'),
                requested_at=datetime.utcnow()
            )
            
            await self.order_request_queue.put(order_request)
            
        except Exception as e:
            logger.error(f"Failed to handle order request: {e}")
    
    async def _handle_broker_update(self, message):
        """Handle order updates from broker."""
        try:
            update_data = json.loads(message) if isinstance(message, str) else message
            await self.order_update_queue.put(update_data)
        except Exception as e:
            logger.error(f"Failed to handle broker update: {e}")
    
    def _signal_to_order_request(self, signal_data: Dict[str, Any]) -> Optional[OrderRequest]:
        """Convert trading signal to order request."""
        try:
            action = signal_data.get('recommended_action')
            if action not in ['BUY', 'SELL']:
                return None
            
            symbol = signal_data.get('symbol')
            position_size = signal_data.get('position_size', 0.01)
            confidence = signal_data.get('confidence', 0.5)
            
            # Calculate quantity based on position size and confidence
            # This is simplified - would integrate with portfolio management
            base_quantity = 100  # Base quantity
            adjusted_quantity = base_quantity * position_size * confidence
            
            return OrderRequest(
                request_id=f"signal_{signal_data.get('timestamp', time.time())}",
                symbol=symbol,
                side=OrderSide(action.lower()),
                order_type=OrderType.MARKET,  # Default to market orders for signals
                quantity=adjusted_quantity,
                source="signal",
                strategy_id=signal_data.get('strategy_name'),
                requested_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to convert signal to order request: {e}")
            return None
    
    async def _process_order_requests(self):
        """Process incoming order requests."""
        while self.is_running:
            try:
                # Wait for order request
                order_request = await asyncio.wait_for(
                    self.order_request_queue.get(),
                    timeout=1.0
                )
                
                # Store request
                self.order_requests[order_request.request_id] = order_request
                
                # Create order from request
                order = self._create_order_from_request(order_request)
                
                # Validate order
                risk_check = await self.validator.validate_order(order_request)
                order.risk_validated = risk_check.passed
                
                if not risk_check.passed:
                    # Reject order
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = RejectionReason.VALIDATION_FAILED
                    
                    logger.warning(f"Order rejected: {order.order_id} - {', '.join(risk_check.failures)}")
                    self.orders_rejected += 1
                    
                    await self._publish_order_update(order)
                    await self._cache_order(order)
                    
                else:
                    # Submit order to broker
                    order.status = OrderStatus.VALIDATED
                    success = await self._submit_order_to_broker(order)
                    
                    if success:
                        order.status = OrderStatus.SUBMITTED
                        order.submitted_at = datetime.utcnow()
                        self.active_orders[order.order_id] = order
                    else:
                        order.status = OrderStatus.REJECTED
                        order.rejection_reason = RejectionReason.SYSTEM_ERROR
                        self.orders_rejected += 1
                    
                    await self._publish_order_update(order)
                    await self._cache_order(order)
                
                self.orders[order.order_id] = order
                self.orders_processed += 1
                self.order_request_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Order request processing error: {e}")
    
    def _create_order_from_request(self, request: OrderRequest) -> Order:
        """Create order object from request."""
        order_id = f"OMS_{self.order_counter:06d}"
        self.order_counter += 1
        
        return Order(
            order_id=order_id,
            client_order_id=request.client_order_id,
            broker_order_id=None,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            price=request.price,
            stop_price=request.stop_price,
            time_in_force=request.time_in_force,
            status=OrderStatus.PENDING,
            source=request.source,
            strategy_id=request.strategy_id,
            parent_order_id=request.parent_order_id,
            created_at=datetime.utcnow()
        )
    
    async def _submit_order_to_broker(self, order: Order) -> bool:
        """Submit order to broker service."""
        try:
            if self.broker_service:
                # Convert to broker order format
                broker_order_request = {
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'order_type': order.order_type.value,
                    'quantity': order.quantity,
                    'price': order.price,
                    'stop_price': order.stop_price,
                    'time_in_force': order.time_in_force.value,
                    'client_order_id': order.order_id
                }
                
                # Submit to broker (would call actual broker service)
                # result = await self.broker_service.place_order(broker_order_request)
                
                # Mock successful submission
                order.broker_order_id = f"broker_{order.order_id}"
                logger.info(f"Order submitted to broker: {order.order_id}")
                return True
            else:
                logger.warning("No broker service available")
                return False
                
        except Exception as e:
            logger.error(f"Failed to submit order to broker: {e}")
            return False
    
    async def _process_order_updates(self):
        """Process order status updates from broker."""
        while self.is_running:
            try:
                # Wait for order update
                update_data = await asyncio.wait_for(
                    self.order_update_queue.get(),
                    timeout=1.0
                )
                
                order_id = update_data.get('order_id')
                if order_id not in self.orders:
                    logger.warning(f"Received update for unknown order: {order_id}")
                    continue
                
                order = self.orders[order_id]
                old_status = order.status
                
                # Update order status
                new_status = OrderStatus(update_data.get('status', order.status.value))
                order.status = new_status
                order.updated_at = datetime.utcnow()
                
                # Update fill information
                if 'filled_quantity' in update_data:
                    order.filled_quantity = float(update_data['filled_quantity'])
                    order.remaining_quantity = order.quantity - order.filled_quantity
                
                if 'average_fill_price' in update_data:
                    order.average_fill_price = float(update_data['average_fill_price'])
                
                # Handle status transitions
                if old_status != new_status:
                    await self._handle_status_change(order, old_status)
                
                # Cache and publish update
                await self._cache_order(order)
                await self._publish_order_update(order)
                
                self.order_update_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Order update processing error: {e}")
    
    async def _handle_status_change(self, order: Order, old_status: OrderStatus):
        """Handle order status changes."""
        try:
            if order.status == OrderStatus.FILLED:
                # Order fully filled
                if order.order_id in self.active_orders:
                    del self.active_orders[order.order_id]
                
                self.orders_filled += 1
                self.total_fill_value += order.filled_quantity * (order.average_fill_price or order.price or 0)
                
                logger.info(f"Order filled: {order.order_id} - {order.filled_quantity} shares at {order.average_fill_price}")
                
            elif order.status == OrderStatus.CANCELLED:
                # Order cancelled
                if order.order_id in self.active_orders:
                    del self.active_orders[order.order_id]
                
                logger.info(f"Order cancelled: {order.order_id}")
                
            elif order.status == OrderStatus.REJECTED:
                # Order rejected
                if order.order_id in self.active_orders:
                    del self.active_orders[order.order_id]
                
                self.orders_rejected += 1
                logger.warning(f"Order rejected: {order.order_id}")
                
            elif order.status == OrderStatus.OPEN:
                # Order accepted by exchange
                logger.info(f"Order open: {order.order_id}")
                
        except Exception as e:
            logger.error(f"Failed to handle status change for order {order.order_id}: {e}")
    
    async def _process_fills(self):
        """Process order fill notifications."""
        while self.is_running:
            try:
                # Wait for fill notification
                fill_data = await asyncio.wait_for(
                    self.fill_queue.get(),
                    timeout=1.0
                )
                
                order_id = fill_data.get('order_id')
                if order_id not in self.orders:
                    logger.warning(f"Received fill for unknown order: {order_id}")
                    continue
                
                order = self.orders[order_id]
                
                # Create fill record
                fill = Fill(
                    fill_id=str(uuid.uuid4()),
                    order_id=order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=float(fill_data['quantity']),
                    price=float(fill_data['price']),
                    timestamp=datetime.fromisoformat(fill_data.get('timestamp', datetime.utcnow().isoformat())),
                    commission=float(fill_data.get('commission', 0)),
                    exchange=fill_data.get('exchange'),
                    execution_id=fill_data.get('execution_id')
                )
                
                # Add fill to order
                order.fills.append(asdict(fill))
                order.filled_quantity += fill.quantity
                order.remaining_quantity = order.quantity - order.filled_quantity
                order.commission += fill.commission
                
                # Update average fill price
                if order.filled_quantity > 0:
                    total_value = sum(f['quantity'] * f['price'] for f in order.fills)
                    order.average_fill_price = total_value / order.filled_quantity
                
                # Update order status
                if order.remaining_quantity <= 0.001:  # Account for floating point precision
                    order.status = OrderStatus.FILLED
                    if order.order_id in self.active_orders:
                        del self.active_orders[order.order_id]
                else:
                    order.status = OrderStatus.PARTIALLY_FILLED
                
                order.updated_at = datetime.utcnow()
                
                # Cache and publish
                await self._cache_order(order)
                await self._cache_fill(fill)
                await self._publish_fill(fill)
                await self._publish_order_update(order)
                
                logger.info(f"Fill processed: {fill.quantity} shares of {fill.symbol} at {fill.price}")
                
                self.fill_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Fill processing error: {e}")
    
    async def _monitor_orders(self):
        """Monitor active orders for timeouts and status updates."""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                current_time = datetime.utcnow()
                
                for order_id, order in list(self.active_orders.items()):
                    # Check for expiration
                    if order.expires_at and current_time > order.expires_at:
                        await self._expire_order(order)
                        continue
                    
                    # Check for day order expiration
                    if (order.time_in_force == TimeInForce.DAY and 
                        (current_time - order.created_at).total_seconds() > 86400):  # 24 hours
                        await self._expire_order(order)
                        continue
                    
                    # Request status update from broker if stale
                    if (current_time - (order.updated_at or order.created_at)).total_seconds() > 300:  # 5 minutes
                        await self._request_order_status(order)
                
            except Exception as e:
                logger.warning(f"Order monitoring error: {e}")
    
    async def _expire_order(self, order: Order):
        """Expire an order."""
        try:
            order.status = OrderStatus.EXPIRED
            order.updated_at = datetime.utcnow()
            
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
            
            await self._cache_order(order)
            await self._publish_order_update(order)
            
            logger.info(f"Order expired: {order.order_id}")
            
        except Exception as e:
            logger.error(f"Failed to expire order {order.order_id}: {e}")
    
    async def _request_order_status(self, order: Order):
        """Request order status update from broker."""
        try:
            if self.broker_service and order.broker_order_id:
                # Would request status from broker service
                logger.debug(f"Requesting status update for order: {order.order_id}")
        except Exception as e:
            logger.warning(f"Failed to request order status: {e}")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of old orders and data."""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Clean up every hour
                
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                
                # Clean up old completed orders
                old_orders = []
                for order_id, order in self.orders.items():
                    if (order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                                       OrderStatus.REJECTED, OrderStatus.EXPIRED] and
                        (order.updated_at or order.created_at) < cutoff_time):
                        old_orders.append(order_id)
                
                for order_id in old_orders:
                    del self.orders[order_id]
                
                # Clean up old requests
                old_requests = []
                for request_id, request in self.order_requests.items():
                    if request.requested_at < cutoff_time:
                        old_requests.append(request_id)
                
                for request_id in old_requests:
                    del self.order_requests[request_id]
                
                if old_orders or old_requests:
                    logger.debug(f"Cleaned up {len(old_orders)} old orders and {len(old_requests)} old requests")
                
            except Exception as e:
                logger.warning(f"Periodic cleanup error: {e}")
    
    async def _cache_order(self, order: Order):
        """Cache order information."""
        try:
            if self.cache:
                cache_key = f"order:{order.order_id}"
                order_data = asdict(order)
                
                # Convert datetime fields
                order_data['created_at'] = order.created_at.isoformat()
                if order.submitted_at:
                    order_data['submitted_at'] = order.submitted_at.isoformat()
                if order.updated_at:
                    order_data['updated_at'] = order.updated_at.isoformat()
                if order.expires_at:
                    order_data['expires_at'] = order.expires_at.isoformat()
                
                # Convert enums
                order_data['side'] = order.side.value
                order_data['order_type'] = order.order_type.value
                order_data['time_in_force'] = order.time_in_force.value
                order_data['status'] = order.status.value
                if order.rejection_reason:
                    order_data['rejection_reason'] = order.rejection_reason.value
                
                await self.cache.set_json(cache_key, order_data, ttl=604800)  # 7 days
        except Exception as e:
            logger.warning(f"Failed to cache order: {e}")
    
    async def _cache_fill(self, fill: Fill):
        """Cache fill information."""
        try:
            if self.cache:
                cache_key = f"fill:{fill.fill_id}"
                fill_data = asdict(fill)
                fill_data['timestamp'] = fill.timestamp.isoformat()
                fill_data['side'] = fill.side.value
                
                await self.cache.set_json(cache_key, fill_data, ttl=2592000)  # 30 days
        except Exception as e:
            logger.warning(f"Failed to cache fill: {e}")
    
    async def _publish_order_update(self, order: Order):
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
                    'remaining_quantity': order.remaining_quantity,
                    'average_fill_price': order.average_fill_price,
                    'timestamp': (order.updated_at or order.created_at).isoformat(),
                    'source': order.source
                }
                
                # Would publish to order updates topic
                logger.debug(f"Publishing order update: {order.order_id} - {order.status.value}")
                
        except Exception as e:
            logger.warning(f"Failed to publish order update: {e}")
    
    async def _publish_fill(self, fill: Fill):
        """Publish fill notification."""
        try:
            if self.producer:
                fill_message = {
                    'fill_id': fill.fill_id,
                    'order_id': fill.order_id,
                    'symbol': fill.symbol,
                    'side': fill.side.value,
                    'quantity': fill.quantity,
                    'price': fill.price,
                    'timestamp': fill.timestamp.isoformat(),
                    'commission': fill.commission
                }
                
                # Would publish to fills topic
                logger.debug(f"Publishing fill: {fill.quantity} shares of {fill.symbol} at {fill.price}")
                
        except Exception as e:
            logger.warning(f"Failed to publish fill: {e}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Cannot cancel order - not active: {order_id}")
                return False
            
            order = self.active_orders[order_id]
            
            # Request cancellation from broker
            if self.broker_service and order.broker_order_id:
                # success = await self.broker_service.cancel_order(order.broker_order_id)
                success = True  # Mock success
                
                if success:
                    order.status = OrderStatus.CANCELLED
                    order.updated_at = datetime.utcnow()
                    
                    del self.active_orders[order_id]
                    
                    await self._cache_order(order)
                    await self._publish_order_update(order)
                    
                    logger.info(f"Order cancelled: {order_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    async def get_orders(self, symbol: Optional[str] = None, 
                        status: Optional[OrderStatus] = None) -> List[Order]:
        """Get orders with optional filtering."""
        orders = list(self.orders.values())
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol.upper()]
        
        if status:
            orders = [o for o in orders if o.status == status]
        
        return sorted(orders, key=lambda o: o.created_at, reverse=True)
    
    async def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        return list(self.active_orders.values())
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health status."""
        return {
            'service': 'order_management_system',
            'status': 'healthy' if self.is_running else 'stopped',
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {
                'orders_processed': self.orders_processed,
                'orders_filled': self.orders_filled,
                'orders_rejected': self.orders_rejected,
                'total_fill_value': self.total_fill_value,
                'active_orders': len(self.active_orders),
                'total_orders': len(self.orders)
            },
            'queue_sizes': {
                'order_requests': self.order_request_queue.qsize(),
                'order_updates': self.order_update_queue.qsize(),
                'fills': self.fill_queue.qsize()
            },
            'connections': {
                'consumer': self.consumer is not None,
                'producer': self.producer is not None,
                'cache': self.cache is not None,
                'broker_service': self.broker_service is not None
            }
        }


# Global service instance
oms: Optional[OrderManagementSystem] = None


async def get_order_management_system() -> OrderManagementSystem:
    """Get or create order management system instance."""
    global oms
    if oms is None:
        oms = OrderManagementSystem()
    return oms