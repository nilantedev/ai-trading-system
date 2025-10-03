#!/usr/bin/env python3
"""Order Management System - Centralized order processing and execution."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import time
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from trading_common import get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.messaging import get_message_consumer, get_message_producer
from trading_common.resilience import CircuitBreaker, CircuitBreakerConfig, RetryStrategy, RetryConfig

# Import advanced broker service
try:
    from .advanced_broker_service import get_advanced_broker_service, SmartOrder, ExecutionAlgo, ExecutionMetrics
except ImportError:
    from advanced_broker_service import get_advanced_broker_service, SmartOrder, ExecutionAlgo, ExecutionMetrics

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
        
        # Risk limits - use getattr with defaults since settings is a Pydantic object
        risk_limits = getattr(settings, 'risk_limits', {})
        self.max_order_value = risk_limits.get('max_order_value', 100000) if isinstance(risk_limits, dict) else 100000
        self.max_position_size = risk_limits.get('max_position_size', 0.1) if isinstance(risk_limits, dict) else 0.1
        self.max_daily_loss = risk_limits.get('max_daily_loss', 0.02) if isinstance(risk_limits, dict) else 0.02
        
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


class OptionsModeling:
    """Black-Scholes options pricing and Greeks calculation."""
    
    @staticmethod
    def black_scholes(S: float, K: float, T: float, r: float, sigma: float, 
                     option_type: str = 'call') -> float:
        """Calculate option price using Black-Scholes model.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        if T <= 0:
            # Option has expired
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float,
                        option_type: str = 'call') -> Dict[str, float]:
        """Calculate all Greeks for an option.
        
        Returns dictionary with:
            - delta: Rate of change of option price with respect to stock price
            - gamma: Rate of change of delta with respect to stock price
            - theta: Rate of change of option price with respect to time
            - vega: Rate of change of option price with respect to volatility
            - rho: Rate of change of option price with respect to interest rate
        """
        if T <= 0:
            # Option has expired, Greeks are zero or special values
            return {
                'delta': 1.0 if option_type == 'call' and S > K else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        if option_type == 'call':
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            theta = (term1 + term2) / 365  # Convert to daily theta
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta = (term1 + term2) / 365
        
        # Vega (same for calls and puts)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Divide by 100 for 1% change
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Divide by 100 for 1% change
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    @staticmethod
    def implied_volatility(option_price: float, S: float, K: float, T: float, 
                          r: float, option_type: str = 'call') -> float:
        """Calculate implied volatility from option price.
        
        Uses Brent's method for root finding.
        """
        if T <= 0:
            return 0.0
        
        # Check if option price is valid
        if option_type == 'call':
            min_price = max(S - K * np.exp(-r * T), 0)
            max_price = S
        else:
            min_price = max(K * np.exp(-r * T) - S, 0)
            max_price = K * np.exp(-r * T)
        
        if option_price < min_price or option_price > max_price:
            return 0.0  # Invalid option price
        
        def objective(sigma):
            return OptionsModeling.black_scholes(S, K, T, r, sigma, option_type) - option_price
        
        try:
            # Use Brent's method to find implied volatility
            iv = brentq(objective, 0.001, 5.0)
            return iv
        except ValueError:
            # If root finding fails, return a default volatility
            return 0.2
    
    @staticmethod
    def calculate_volatility_surface(strikes: np.ndarray, expirations: np.ndarray,
                                    option_prices: np.ndarray, S: float, r: float,
                                    option_type: str = 'call') -> np.ndarray:
        """Calculate volatility surface from option prices.
        
        Args:
            strikes: Array of strike prices
            expirations: Array of expiration times (years)
            option_prices: 2D array of option prices [strikes x expirations]
            S: Current stock price
            r: Risk-free rate
            option_type: 'call' or 'put'
            
        Returns:
            2D array of implied volatilities
        """
        vol_surface = np.zeros_like(option_prices)
        
        for i, K in enumerate(strikes):
            for j, T in enumerate(expirations):
                if option_prices[i, j] > 0:
                    vol_surface[i, j] = OptionsModeling.implied_volatility(
                        option_prices[i, j], S, K, T, r, option_type
                    )
        
        return vol_surface
    
    @staticmethod
    def delta_hedging_quantity(position_delta: float, option_delta: float) -> float:
        """Calculate hedge quantity for delta-neutral portfolio.
        
        Args:
            position_delta: Current portfolio delta
            option_delta: Delta of the option to hedge with
            
        Returns:
            Number of options to trade (negative for sell, positive for buy)
        """
        if option_delta == 0:
            return 0
        
        return -position_delta / option_delta
    
    @staticmethod
    def calculate_option_payoff(S_T: np.ndarray, K: float, option_type: str = 'call',
                               premium: float = 0, position: str = 'long') -> np.ndarray:
        """Calculate option payoff at expiration.
        
        Args:
            S_T: Array of possible stock prices at expiration
            K: Strike price
            option_type: 'call' or 'put'
            premium: Option premium paid/received
            position: 'long' or 'short'
            
        Returns:
            Array of payoffs
        """
        if option_type == 'call':
            intrinsic = np.maximum(S_T - K, 0)
        else:
            intrinsic = np.maximum(K - S_T, 0)
        
        if position == 'long':
            payoff = intrinsic - premium
        else:
            payoff = premium - intrinsic
        
        return payoff
    
    @staticmethod
    def monte_carlo_option_price(S: float, K: float, T: float, r: float, 
                                sigma: float, n_simulations: int = 10000,
                                option_type: str = 'call') -> Tuple[float, float]:
        """Calculate option price using Monte Carlo simulation.
        
        Returns:
            Tuple of (price, standard_error)
        """
        if T <= 0:
            if option_type == 'call':
                return max(S - K, 0), 0
            else:
                return max(K - S, 0), 0
        
        # Generate random price paths
        Z = np.random.standard_normal(n_simulations)
        S_T = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
        
        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        standard_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)
        
        return option_price, standard_error


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
            RetryConfig(
                max_attempts=3,
                initial_delay=0.5,
                max_delay=5.0,
                exponential_base=2.0
            )
        )
        self.total_fill_value = 0.0
        
        # Order ID generation
        self.order_counter = 1
        
        # Options modeling instance
        self.options_modeler = OptionsModeling()
        
        # Store for ML predictions
        self.ml_predictions = {}  # symbol -> predictions
        
        # Advanced broker service
        self.advanced_broker = None
        
        # Position tracking
        self.positions = {}  # symbol -> position
        self.position_limits = {
            'max_position_size': 100000,  # Max $ per position
            'max_positions': 20,           # Max number of positions
            'max_concentration': 0.2,       # Max 20% in one position
            'max_sector_exposure': 0.4     # Max 40% in one sector
        }
        
    async def start(self):
        """Initialize and start order management system."""
        logger.info("Starting Order Management System")
        
        try:
            # Initialize connections
            self.consumer = await get_message_consumer()
            self.producer = await get_message_producer()
            # Await the async TradingCache factory to get a realized cache instance
            self.cache = await get_trading_cache()
            
            # Initialize services
            await self._initialize_services()
            
            # Initialize advanced broker
            try:
                self.advanced_broker = await get_advanced_broker_service()
                logger.info("Advanced broker service initialized with smart order routing")
            except Exception as e:
                logger.warning(f"Advanced broker initialization failed, using basic: {e}")
            
            # Subscribe to order flows
            await self._setup_subscriptions()
            
            # Start processing tasks
            self.is_running = True
            
            # Create background tasks without blocking
            self.background_tasks = [
                asyncio.create_task(self._process_order_requests()),
                asyncio.create_task(self._process_order_updates()),
                asyncio.create_task(self._process_fills()),
                asyncio.create_task(self._consume_messages()),
                asyncio.create_task(self._monitor_orders()),
                asyncio.create_task(self._periodic_cleanup())
            ]
            
            logger.info("Order management system started with 6 concurrent tasks")
            # Don't await gather - let tasks run in background
            
        except Exception as e:
            logger.error(f"Failed to start order management system: {e}")
            raise
    
    async def stop(self):
        """Stop order management system gracefully."""
        logger.info("Stopping Order Management System")
        self.is_running = False
        
        # Cancel background tasks
        if hasattr(self, 'background_tasks'):
            for task in self.background_tasks:
                task.cancel()
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        if self.consumer:
            await self.consumer.close()
        if self.producer:
            await self.producer.close()
        
        logger.info("Order Management System stopped")
    
    async def _initialize_services(self):
        """Initialize dependent services."""
        try:
            # Import services dynamically to avoid circular imports and path issues
            from broker_service import get_broker_service
            # Ensure the risk-monitor module is importable both in and out of package context
            try:
                from services.risk_monitor.risk_monitoring_service import get_risk_service  # type: ignore
            except Exception:
                # Add ../risk-monitor to path and retry
                import sys as _sys
                from pathlib import Path as _Path
                _sys.path.insert(0, str(_Path(__file__).parent.parent / 'risk-monitor'))
                try:
                    from risk_monitoring_service import get_risk_service  # type: ignore
                except Exception as e:
                    raise ImportError(f"risk service import failed: {e}")
            # from portfolio_service import get_portfolio_service  # Would be implemented

            self.broker_service = await get_broker_service()
            try:
                # Some broker services require async connect (Alpaca/HTTP), best-effort
                if hasattr(self.broker_service, 'connect'):
                    await self.broker_service.connect()  # type: ignore[misc]
            except Exception as e:  # noqa: BLE001
                logger.warning(f"broker.connect failed (continuing in degraded mode): {e}")
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

    async def place_order(self, order_request: OrderRequest, user: Optional[dict] = None) -> Order:
        """Public API to place an order flowing through validation and broker.

        Returns the created Order object (tracked internally).
        """
        # Normalize request metadata
        if not order_request.request_id:
            order_request.request_id = str(uuid.uuid4())
        if not order_request.requested_at:
            order_request.requested_at = datetime.utcnow()

        # Create order and validate
        order = self._create_order_from_request(order_request)
        risk_check = await self.validator.validate_order(order_request)
        order.risk_validated = risk_check.passed
        if not risk_check.passed:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = RejectionReason.VALIDATION_FAILED
            self.orders_rejected += 1
            await self._publish_order_update(order)
            await self._cache_order(order)
            self.orders[order.order_id] = order
            return order

        # Submit to broker
        order.status = OrderStatus.VALIDATED
        submitted = await self._submit_order_to_broker(order)
        if submitted:
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
        return order
    
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
            # Normalize message to dict shape our converter expects
            signal_data = self._normalize_trading_signal(message)
            
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

    def _normalize_trading_signal(self, raw: Any) -> Dict[str, Any]:
        """Normalize incoming trading-signal (Avro Record or dict/JSON string) to a dict.

        Expected output keys: symbol, recommended_action, position_size, confidence, strategy_name, timestamp
        """
        try:
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    # Fallback: wrap as minimal dict
                    return {'recommended_action': 'HOLD', 'symbol': raw, 'position_size': 0.0, 'confidence': 0.0}
            # Avro Record (TradingSignalMessage) case: has attributes
            if hasattr(raw, 'symbol') and hasattr(raw, 'signal_type'):
                symbol = getattr(raw, 'symbol', None)
                sig_type = str(getattr(raw, 'signal_type', '') or '').upper()
                confidence = float(getattr(raw, 'confidence', 0.5) or 0.0)
                strategy_name = getattr(raw, 'strategy_name', None)
                ts = getattr(raw, 'timestamp', None)
                reasoning = str(getattr(raw, 'reasoning', '') or '')
                # Parse size from reasoning if present: e.g., "size=0.0420"
                size = 0.0
                try:
                    if 'size=' in reasoning:
                        after = reasoning.split('size=', 1)[1]
                        num = ''
                        for ch in after:
                            if ch in '0123456789.+-eE':
                                num += ch
                            else:
                                break
                        size = float(num) if num else 0.0
                except Exception:
                    size = 0.0
                if size <= 0.0:
                    # Derive a conservative size from confidence (1%-10%)
                    size = max(0.01, min(0.1, 0.02 + 0.08 * confidence))
                action = 'BUY' if sig_type == 'BUY' else ('SELL' if sig_type == 'SELL' else 'HOLD')
                return {
                    'symbol': symbol,
                    'recommended_action': action,
                    'position_size': size,
                    'confidence': confidence,
                    'strategy_name': strategy_name,
                    'timestamp': ts,
                }
            # Dict-like
            if isinstance(raw, dict):
                # Map signal_type -> recommended_action if needed
                if 'recommended_action' not in raw and 'signal_type' in raw:
                    st = str(raw.get('signal_type') or '').upper()
                    raw['recommended_action'] = 'BUY' if st == 'BUY' else ('SELL' if st == 'SELL' else 'HOLD')
                # Ensure position_size
                if 'position_size' not in raw:
                    conf = float(raw.get('confidence', 0.5) or 0.0)
                    raw['position_size'] = max(0.01, min(0.1, 0.02 + 0.08 * conf))
                return raw
        except Exception as e:
            logger.debug(f"signal.normalize.failed err={e}")
        # Fallback minimal HOLD
        return {'recommended_action': 'HOLD', 'symbol': None, 'position_size': 0.0, 'confidence': 0.0}
    
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
        """Submit order to broker service with PhD-level execution."""
        try:
            # Use advanced broker if available for smart execution
            if self.advanced_broker and order.quantity >= 100:
                # Determine execution algorithm based on order characteristics
                exec_algo = await self._select_execution_algorithm(order)
                
                # Create smart order
                smart_order = SmartOrder(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side.value,
                    total_quantity=order.quantity,
                    limit_price=order.price,
                    execution_algo=exec_algo,
                    urgency=self._calculate_urgency(order),
                    max_participation_rate=0.15 if order.quantity > 1000 else 0.25,
                    min_fill_size=100,
                    use_dark_pools=order.quantity > 5000,  # Use dark pools for large orders
                    avoid_detection=order.quantity > 10000  # Anti-gaming for very large orders
                )
                
                # Execute with advanced algorithms
                metrics = await self.advanced_broker.execute_order(smart_order)
                
                # Update order with execution results
                if metrics:
                    order.broker_order_id = f"adv_{order.order_id}"
                    order.average_fill_price = metrics.execution_price
                    order.filled_quantity = smart_order.executed_quantity
                    order.remaining_quantity = smart_order.remaining_quantity
                    
                    # Store execution quality metrics
                    await self._store_execution_metrics(order, metrics)
                    
                    logger.info(
                        f"Advanced execution for {order.order_id}: "
                        f"IS={metrics.implementation_shortfall:.4f}, "
                        f"Impact={metrics.market_impact:.4f}"
                    )
                    return True
                    
            elif self.broker_service:
                # Fall back to basic broker for small orders
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
                
                # Submit to basic broker
                result = await self.broker_service.submit_order(broker_order_request)
                
                if result and result.get('status') == 'submitted':
                    order.broker_order_id = result.get('order_id', f"broker_{order.order_id}")
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
                
                # Update position tracking with PhD-level analytics
                await self.update_position_tracking(order, fill)
                
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
                
                try:
                    await self.producer.send_order_update(order_message)
                except Exception as e:
                    logger.debug(f"Order update publish failed: {e}")
                
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
                
                try:
                    await self.producer.send_fill(fill_message)
                except Exception as e:
                    logger.debug(f"Fill publish failed: {e}")
                
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
    
    async def calculate_options_hedge(self, symbol: str, position_size: float, 
                                     current_price: float) -> Dict[str, Any]:
        """Calculate optimal options hedge for a position."""
        try:
            # Get ML predictions for volatility if available
            ml_vol = self.ml_predictions.get(symbol, {}).get('volatility', 0.2)
            
            # Calculate options parameters
            strike_price = current_price * 1.05  # 5% OTM for protection
            time_to_expiry = 30 / 365  # 30 days
            risk_free_rate = 0.05  # 5% risk-free rate
            
            # Calculate put option price for hedging
            put_price = self.options_modeler.black_scholes(
                S=current_price,
                K=strike_price,
                T=time_to_expiry,
                r=risk_free_rate,
                sigma=ml_vol,
                option_type='put'
            )
            
            # Calculate Greeks
            greeks = self.options_modeler.calculate_greeks(
                S=current_price,
                K=strike_price,
                T=time_to_expiry,
                r=risk_free_rate,
                sigma=ml_vol,
                option_type='put'
            )
            
            # Calculate hedge ratio
            hedge_ratio = abs(greeks['delta'])
            contracts_needed = position_size / (100 * hedge_ratio)  # Options are per 100 shares
            
            return {
                'hedge_type': 'protective_put',
                'strike': strike_price,
                'premium': put_price,
                'contracts': int(contracts_needed),
                'total_cost': put_price * contracts_needed * 100,
                'greeks': greeks,
                'hedge_effectiveness': hedge_ratio,
                'max_loss': (strike_price - current_price) * position_size + put_price * contracts_needed * 100
            }
            
        except Exception as e:
            logger.error(f"Options hedge calculation failed for {symbol}: {e}")
            return {'hedge_type': 'none', 'error': str(e)}
    
    async def update_position_tracking(self, order: Order, fill: Fill):
        """Update position tracking with sophisticated analytics."""
        try:
            symbol = order.symbol
            
            # Initialize position if needed
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': 0,
                    'avg_price': 0,
                    'market_value': 0,
                    'unrealized_pnl': 0,
                    'realized_pnl': 0,
                    'total_cost': 0,
                    'vwap': 0,
                    'high_water_mark': 0,
                    'drawdown': 0,
                    'sharpe_ratio': 0,
                    'fills': [],
                    'risk_metrics': {}
                }
            
            position = self.positions[symbol]
            
            # Update position based on fill
            if order.side == OrderSide.BUY:
                # Calculate new average price
                new_total_cost = (position['quantity'] * position['avg_price']) + (fill.quantity * fill.price)
                new_quantity = position['quantity'] + fill.quantity
                position['avg_price'] = new_total_cost / new_quantity if new_quantity > 0 else 0
                position['quantity'] = new_quantity
                position['total_cost'] = new_total_cost
            else:  # SELL
                # Calculate realized P&L
                if position['quantity'] > 0:
                    realized = (fill.price - position['avg_price']) * fill.quantity
                    position['realized_pnl'] += realized
                
                position['quantity'] -= fill.quantity
                if position['quantity'] <= 0:
                    # Position closed
                    position['avg_price'] = 0
                    position['total_cost'] = 0
            
            # Update VWAP
            position['fills'].append({
                'price': fill.price,
                'quantity': fill.quantity,
                'timestamp': fill.timestamp
            })
            total_value = sum(f['price'] * f['quantity'] for f in position['fills'])
            total_qty = sum(f['quantity'] for f in position['fills'])
            position['vwap'] = total_value / total_qty if total_qty > 0 else 0
            
            # Calculate current metrics
            current_price = fill.price  # Use latest fill as current price
            position['market_value'] = position['quantity'] * current_price
            position['unrealized_pnl'] = (current_price - position['avg_price']) * position['quantity']
            
            # Update high water mark and drawdown
            total_value = position['market_value'] + position['realized_pnl']
            if total_value > position['high_water_mark']:
                position['high_water_mark'] = total_value
            position['drawdown'] = (position['high_water_mark'] - total_value) / position['high_water_mark'] if position['high_water_mark'] > 0 else 0
            
            # Calculate risk metrics
            position['risk_metrics'] = await self._calculate_position_risk(symbol, position)
            
            # Check position limits
            await self._check_position_limits(symbol, position)
            
            # Store position update
            if self.cache:
                await self.cache.set_json(f"position:{symbol}", position, ttl=86400)
            
        except Exception as e:
            logger.error(f"Failed to update position tracking: {e}")
    
    async def _calculate_position_risk(self, symbol: str, position: Dict) -> Dict[str, float]:
        """Calculate comprehensive risk metrics for position."""
        try:
            risk_metrics = {}
            
            # Position exposure
            risk_metrics['exposure'] = abs(position['market_value'])
            
            # Calculate beta if we have market data
            if self.cache:
                market_data = await self.cache.get_json(f"market_beta:{symbol}")
                beta = market_data.get('beta', 1.0) if market_data else 1.0
            else:
                beta = 1.0
            
            risk_metrics['beta_adjusted_exposure'] = risk_metrics['exposure'] * beta
            
            # VaR calculation (simplified)
            position_volatility = 0.02  # 2% daily vol assumption
            confidence_level = 0.95
            z_score = 1.645  # 95% confidence
            risk_metrics['var_95'] = position['market_value'] * position_volatility * z_score
            
            # Expected Shortfall (CVaR)
            risk_metrics['cvar_95'] = risk_metrics['var_95'] * 1.25  # Approximation
            
            # Sharpe ratio calculation (simplified)
            if position['realized_pnl'] != 0:
                returns = position['realized_pnl'] / max(position['total_cost'], 1)
                risk_metrics['sharpe'] = returns / position_volatility if position_volatility > 0 else 0
            else:
                risk_metrics['sharpe'] = 0
            
            # Greeks for options positions (if applicable)
            if 'option' in symbol.lower() or position.get('is_option'):
                greeks = self.options_modeler.calculate_greeks(
                    S=position.get('underlying_price', 100),
                    K=position.get('strike', 100),
                    T=position.get('time_to_expiry', 0.1),
                    r=0.05,
                    sigma=0.2
                )
                risk_metrics['delta_exposure'] = greeks['delta'] * position['quantity'] * 100
                risk_metrics['gamma_exposure'] = greeks['gamma'] * position['quantity'] * 100
                risk_metrics['vega_exposure'] = greeks['vega'] * position['quantity']
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate position risk: {e}")
            return {}
    
    async def _check_position_limits(self, symbol: str, position: Dict):
        """Check if position exceeds risk limits."""
        try:
            warnings = []
            
            # Check max position size
            if abs(position['market_value']) > self.position_limits['max_position_size']:
                warnings.append(f"Position size ${abs(position['market_value']):,.0f} exceeds limit")
            
            # Check concentration
            total_portfolio_value = sum(
                p.get('market_value', 0) for p in self.positions.values()
            )
            if total_portfolio_value > 0:
                concentration = abs(position['market_value']) / total_portfolio_value
                if concentration > self.position_limits['max_concentration']:
                    warnings.append(f"Position concentration {concentration:.1%} exceeds limit")
            
            # Check number of positions
            if len(self.positions) > self.position_limits['max_positions']:
                warnings.append(f"Too many positions: {len(self.positions)}")
            
            # Log warnings
            if warnings:
                logger.warning(f"Position limit warnings for {symbol}: {', '.join(warnings)}")
                
                # Publish risk alert
                if self.producer:
                    await self.producer.send_risk_alert({
                        'symbol': symbol,
                        'warnings': warnings,
                        'position': position,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
        except Exception as e:
            logger.error(f"Failed to check position limits: {e}")
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary with PhD-level analytics."""
        try:
            if not self.positions:
                return {'status': 'no_positions'}
            
            # Calculate portfolio metrics
            total_value = sum(p.get('market_value', 0) for p in self.positions.values())
            total_unrealized = sum(p.get('unrealized_pnl', 0) for p in self.positions.values())
            total_realized = sum(p.get('realized_pnl', 0) for p in self.positions.values())
            
            # Position breakdown
            long_positions = {s: p for s, p in self.positions.items() if p['quantity'] > 0}
            short_positions = {s: p for s, p in self.positions.items() if p['quantity'] < 0}
            
            # Risk metrics
            total_var = sum(p.get('risk_metrics', {}).get('var_95', 0) for p in self.positions.values())
            total_beta_exposure = sum(
                p.get('risk_metrics', {}).get('beta_adjusted_exposure', 0) 
                for p in self.positions.values()
            )
            
            # Calculate portfolio Sharpe
            if total_value > 0:
                portfolio_return = (total_realized + total_unrealized) / total_value
                portfolio_volatility = total_var / total_value if total_value > 0 else 0
                portfolio_sharpe = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            else:
                portfolio_sharpe = 0
            
            # Top positions by risk
            positions_by_risk = sorted(
                self.positions.items(),
                key=lambda x: abs(x[1].get('risk_metrics', {}).get('var_95', 0)),
                reverse=True
            )[:5]
            
            return {
                'total_value': total_value,
                'unrealized_pnl': total_unrealized,
                'realized_pnl': total_realized,
                'total_pnl': total_unrealized + total_realized,
                'num_positions': len(self.positions),
                'num_long': len(long_positions),
                'num_short': len(short_positions),
                'risk_metrics': {
                    'portfolio_var_95': total_var,
                    'beta_adjusted_exposure': total_beta_exposure,
                    'portfolio_sharpe': portfolio_sharpe,
                    'max_drawdown': max(p.get('drawdown', 0) for p in self.positions.values()) if self.positions else 0
                },
                'top_positions': [
                    {
                        'symbol': symbol,
                        'value': pos['market_value'],
                        'pnl': pos['unrealized_pnl'],
                        'var_95': pos.get('risk_metrics', {}).get('var_95', 0)
                    }
                    for symbol, pos in positions_by_risk
                ],
                'execution_quality': {
                    'orders_today': self.orders_processed,
                    'fill_rate': self.orders_filled / max(self.orders_processed, 1),
                    'rejection_rate': self.orders_rejected / max(self.orders_processed, 1)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate portfolio summary: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def optimize_order_with_ml(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Optimize order parameters using ML predictions."""
        try:
            symbol = order_request.symbol
            
            # Get ML predictions if available
            if symbol in self.ml_predictions:
                predictions = self.ml_predictions[symbol]
                
                # Adjust order based on predicted price movement
                predicted_move = predictions.get('price_change_1h', 0)
                volatility = predictions.get('volatility', 0.02)
                
                optimization = {
                    'original_price': order_request.price,
                    'original_quantity': order_request.quantity,
                    'ml_confidence': predictions.get('confidence', 0.5)
                }
                
                # For limit orders, adjust price based on prediction
                if order_request.order_type == OrderType.LIMIT:
                    if order_request.side == OrderSide.BUY:
                        # If price predicted to go down, lower our bid
                        if predicted_move < 0:
                            adjusted_price = order_request.price * (1 + predicted_move * 0.5)
                            optimization['adjusted_price'] = adjusted_price
                            order_request.price = adjusted_price
                    else:  # SELL
                        # If price predicted to go up, raise our ask
                        if predicted_move > 0:
                            adjusted_price = order_request.price * (1 + predicted_move * 0.5)
                            optimization['adjusted_price'] = adjusted_price
                            order_request.price = adjusted_price
                
                # Adjust quantity based on confidence
                confidence_adjustment = 0.5 + predictions.get('confidence', 0.5)
                adjusted_quantity = order_request.quantity * confidence_adjustment
                optimization['adjusted_quantity'] = adjusted_quantity
                order_request.quantity = adjusted_quantity
                
                # Add stop loss based on volatility
                if not order_request.stop_price:
                    stop_distance = 2 * volatility  # 2 standard deviations
                    if order_request.side == OrderSide.BUY:
                        order_request.stop_price = order_request.price * (1 - stop_distance)
                    else:
                        order_request.stop_price = order_request.price * (1 + stop_distance)
                    optimization['stop_price'] = order_request.stop_price
                
                logger.info(f"ML optimization for {symbol}: {optimization}")
                return optimization
            
            return {'status': 'no_ml_predictions'}
            
        except Exception as e:
            logger.error(f"ML optimization failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _select_execution_algorithm(self, order: Order) -> ExecutionAlgo:
        """Select optimal execution algorithm based on order and market conditions."""
        try:
            # Get market conditions
            if self.cache:
                market_data = await self.cache.get_json(f"market_state:{order.symbol}")
                volatility = market_data.get('volatility', 0.02) if market_data else 0.02
                spread = market_data.get('spread', 0.001) if market_data else 0.001
                volume = market_data.get('volume', 100000) if market_data else 100000
            else:
                volatility = 0.02
                spread = 0.001
                volume = 100000
            
            # Algorithm selection logic
            if order.quantity > volume * 0.1:
                # Large order relative to volume - use VWAP to minimize impact
                return ExecutionAlgo.VWAP
            elif volatility > 0.03:
                # High volatility - use TWAP to spread risk
                return ExecutionAlgo.TWAP
            elif spread > 0.002:
                # Wide spread - use Iceberg to minimize crossing
                return ExecutionAlgo.ICEBERG
            elif order.time_in_force == TimeInForce.IOC:
                # Immediate execution needed - use Sniper
                return ExecutionAlgo.SNIPER
            elif order.quantity < 500:
                # Small order - use aggressive execution
                return ExecutionAlgo.SNIPER
            else:
                # Default to adaptive for most cases
                return ExecutionAlgo.ADAPTIVE
                
        except Exception as e:
            logger.warning(f"Error selecting execution algorithm: {e}")
            return ExecutionAlgo.ADAPTIVE
    
    def _calculate_urgency(self, order: Order) -> float:
        """Calculate order urgency score (0=patient, 1=aggressive)."""
        urgency = 0.5  # Default moderate urgency
        
        # Increase urgency for certain conditions
        if order.time_in_force == TimeInForce.IOC:
            urgency = 1.0
        elif order.time_in_force == TimeInForce.FOK:
            urgency = 0.9
        elif order.order_type == OrderType.MARKET:
            urgency = 0.8
        elif order.order_type == OrderType.STOP:
            urgency = 0.7
        
        # Adjust based on source
        if order.source == "manual":
            urgency = min(1.0, urgency + 0.2)
        elif order.source == "signal":
            urgency = min(1.0, urgency + 0.1)
        
        return urgency
    
    async def _store_execution_metrics(self, order: Order, metrics: ExecutionMetrics):
        """Store execution quality metrics for analysis."""
        try:
            if self.cache:
                metrics_data = {
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'implementation_shortfall': metrics.implementation_shortfall,
                    'market_impact': metrics.market_impact,
                    'total_cost': metrics.total_cost,
                    'price_improvement': metrics.price_improvement,
                    'execution_time': metrics.execution_time,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Store metrics
                await self.cache.set_json(
                    f"execution_metrics:{order.order_id}",
                    metrics_data,
                    ttl=2592000  # 30 days
                )
                
                # Update symbol execution statistics
                symbol_key = f"symbol_exec_stats:{order.symbol}"
                stats = await self.cache.get_json(symbol_key) or {
                    'total_orders': 0,
                    'avg_shortfall': 0,
                    'avg_impact': 0
                }
                
                # Update running averages
                n = stats['total_orders']
                stats['avg_shortfall'] = (stats['avg_shortfall'] * n + metrics.implementation_shortfall) / (n + 1)
                stats['avg_impact'] = (stats['avg_impact'] * n + metrics.market_impact) / (n + 1)
                stats['total_orders'] = n + 1
                
                await self.cache.set_json(symbol_key, stats, ttl=604800)  # 7 days
                
        except Exception as e:
            logger.warning(f"Failed to store execution metrics: {e}")
    
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
                'broker_service': self.broker_service is not None,
                'advanced_broker': self.advanced_broker is not None
            },
            'execution_capabilities': {
                'smart_order_routing': self.advanced_broker is not None,
                'dark_pool_access': self.advanced_broker is not None,
                'algorithms': ['TWAP', 'VWAP', 'ICEBERG', 'SNIPER', 'ADAPTIVE'] if self.advanced_broker else [],
                'options_modeling': True,
                'position_limits': self.position_limits
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