#!/usr/bin/env python3
"""
Resilience patterns for external API calls.
Provides retry logic, circuit breakers, and backoff strategies.
"""

import asyncio
import time
import random
import logging
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: int = 60  # Seconds before trying half-open
    expected_exception: type = Exception  # Exception types to track
    success_threshold: int = 2  # Successes needed to close from half-open


@dataclass
class CircuitBreakerState:
    """State tracking for circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.now)


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker."""
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState()
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        async with self._lock:
            # Check if we should attempt the call
            if not self._should_attempt_call():
                raise Exception(f"Circuit breaker {self.name} is OPEN")
            
            # Update state if moving from OPEN to HALF_OPEN
            if self.state.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state.state = CircuitState.HALF_OPEN
                    self.state.success_count = 0
                    logger.info(f"Circuit breaker {self.name} moving to HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        # Attempt the call
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await self._record_success()
            return result
            
        except self.config.expected_exception as e:
            await self._record_failure()
            raise
    
    def _should_attempt_call(self) -> bool:
        """Check if we should attempt a call."""
        if self.state.state == CircuitState.CLOSED:
            return True
        elif self.state.state == CircuitState.HALF_OPEN:
            return True
        elif self.state.state == CircuitState.OPEN:
            return self._should_attempt_reset()
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try resetting."""
        if self.state.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.state.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    async def _record_success(self):
        """Record a successful call."""
        async with self._lock:
            if self.state.state == CircuitState.HALF_OPEN:
                self.state.success_count += 1
                if self.state.success_count >= self.config.success_threshold:
                    self.state.state = CircuitState.CLOSED
                    self.state.failure_count = 0
                    self.state.success_count = 0
                    logger.info(f"Circuit breaker {self.name} CLOSED (recovered)")
            elif self.state.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.state.failure_count = 0
    
    async def _record_failure(self):
        """Record a failed call."""
        async with self._lock:
            self.state.failure_count += 1
            self.state.last_failure_time = datetime.now()
            
            if self.state.state == CircuitState.CLOSED:
                if self.state.failure_count >= self.config.failure_threshold:
                    self.state.state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker {self.name} OPEN after {self.state.failure_count} failures")
            elif self.state.state == CircuitState.HALF_OPEN:
                # Single failure in half-open goes back to open
                self.state.state = CircuitState.OPEN
                self.state.failure_count = 0
                logger.warning(f"Circuit breaker {self.name} reopened from HALF_OPEN")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state.state.value,
            "failure_count": self.state.failure_count,
            "success_count": self.state.success_count,
            "last_failure": self.state.last_failure_time.isoformat() if self.state.last_failure_time else None
        }


class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """Initialize retry configuration."""
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


class RetryStrategy:
    """Retry strategy with exponential backoff."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """Initialize retry strategy."""
        self.config = config or RetryConfig()
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        # Exponential backoff
        delay = min(
            self.config.initial_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            delay = delay * (0.5 + random.random())
        
        return delay
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.max_attempts - 1:
                    delay = self.get_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.config.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_attempts} attempts failed")
        
        raise last_exception


def with_retry(max_attempts: int = 3, backoff_base: float = 2.0):
    """Decorator for adding retry logic to functions."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            strategy = RetryStrategy(RetryConfig(max_attempts=max_attempts, exponential_base=backoff_base))
            return await strategy.execute(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            strategy = RetryStrategy(RetryConfig(max_attempts=max_attempts, exponential_base=backoff_base))
            return asyncio.run(strategy.execute(func, *args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def with_circuit_breaker(name: str, failure_threshold: int = 5):
    """Decorator for adding circuit breaker to functions."""
    breaker = CircuitBreaker(name, CircuitBreakerConfig(failure_threshold=failure_threshold))
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(breaker.call(func, *args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: int, per: float = 1.0):
        """
        Initialize rate limiter.
        
        Args:
            rate: Number of allowed requests
            per: Time period in seconds
        """
        self.rate = rate
        self.per = per
        self.tokens = rate
        self.updated_at = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.updated_at
            self.updated_at = now
            
            # Refill tokens based on elapsed time
            self.tokens = min(
                self.rate,
                self.tokens + elapsed * (self.rate / self.per)
            )
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_and_acquire(self, tokens: int = 1):
        """Wait until tokens are available and acquire them."""
        while not await self.acquire(tokens):
            # Calculate wait time
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed * (self.per / self.rate)
            await asyncio.sleep(wait_time)


class BulkheadPool:
    """Bulkhead pattern for resource isolation."""
    
    def __init__(self, max_concurrent: int = 10):
        """Initialize bulkhead pool."""
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_count = 0
        self._lock = asyncio.Lock()
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with bulkhead protection."""
        async with self.semaphore:
            async with self._lock:
                self.active_count += 1
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result
            finally:
                async with self._lock:
                    self.active_count -= 1
    
    def get_status(self) -> Dict[str, int]:
        """Get bulkhead status."""
        return {
            "active": self.active_count,
            "available": self.semaphore._value
        }


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a named circuit breaker."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def get_all_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers."""
    return {name: breaker.get_state() for name, breaker in _circuit_breakers.items()}


# Example usage for external API calls
class ResilientAPIClient:
    """Example of using resilience patterns for API calls."""
    
    def __init__(self, base_url: str):
        """Initialize resilient API client."""
        self.base_url = base_url
        self.retry_strategy = RetryStrategy()
        self.circuit_breaker = CircuitBreaker(f"api_{base_url}")
        self.rate_limiter = RateLimiter(rate=100, per=60)  # 100 requests per minute
        self.bulkhead = BulkheadPool(max_concurrent=10)
    
    @with_retry(max_attempts=3)
    @with_circuit_breaker("external_api", failure_threshold=5)
    async def make_request(self, endpoint: str, **kwargs) -> Any:
        """Make resilient API request."""
        # Rate limiting
        await self.rate_limiter.wait_and_acquire()
        
        # Bulkhead protection
        async def _request():
            # Your actual HTTP request logic here
            # For example: async with aiohttp.ClientSession() as session:
            #     async with session.get(f"{self.base_url}/{endpoint}") as response:
            #         return await response.json()
            pass
        
        return await self.bulkhead.execute(_request)