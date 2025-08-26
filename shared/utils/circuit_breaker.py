#!/usr/bin/env python3
"""
Production-ready circuit breaker implementation for API resilience.
Prevents cascade failures by stopping calls to failing services.
"""

import asyncio
import time
import logging
from enum import Enum
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import redis
import json

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    expected_exception: type = Exception
    success_threshold: int = 2  # successes needed to close from half-open
    redis_client: Optional[redis.Redis] = None
    name: str = "default"


class CircuitBreaker:
    """
    Circuit breaker pattern implementation with Redis persistence.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self._state = CircuitState.CLOSED
        self.redis_client = config.redis_client
        self.circuit_key = f"circuit_breaker:{config.name}"
        
        # Load state from Redis if available
        if self.redis_client:
            self._load_state()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for automatic transitions."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker {self.config.name} moved to HALF_OPEN")
        return self._state
    
    @state.setter
    def state(self, value: CircuitState):
        """Set circuit state and persist to Redis."""
        self._state = value
        self._persist_state()
        logger.info(f"Circuit breaker {self.config.name} state changed to {value.value}")
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _persist_state(self):
        """Persist circuit breaker state to Redis."""
        if not self.redis_client:
            return
        
        try:
            state_data = {
                "state": self._state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time,
                "timestamp": time.time()
            }
            self.redis_client.setex(
                self.circuit_key,
                self.config.recovery_timeout * 2,
                json.dumps(state_data)
            )
        except Exception as e:
            logger.error(f"Failed to persist circuit breaker state: {e}")
    
    def _load_state(self):
        """Load circuit breaker state from Redis."""
        if not self.redis_client:
            return
        
        try:
            data = self.redis_client.get(self.circuit_key)
            if data:
                state_data = json.loads(data)
                self._state = CircuitState(state_data["state"])
                self.failure_count = state_data["failure_count"]
                self.success_count = state_data["success_count"]
                self.last_failure_time = state_data["last_failure_time"]
                logger.info(f"Loaded circuit breaker {self.config.name} state: {self._state.value}")
        except Exception as e:
            logger.error(f"Failed to load circuit breaker state: {e}")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If function fails
        """
        if self.state == CircuitState.OPEN:
            raise CircuitOpenError(f"Circuit breaker {self.config.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function through circuit breaker.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If function fails
        """
        if self.state == CircuitState.OPEN:
            raise CircuitOpenError(f"Circuit breaker {self.config.name} is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker {self.config.name} closed after successful recovery")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset failure count on success
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker {self.config.name} reopened after failure in HALF_OPEN state")
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker {self.config.name} opened after {self.failure_count} failures")
    
    def reset(self):
        """Manually reset the circuit breaker."""
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        logger.info(f"Circuit breaker {self.config.name} manually reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "failure_threshold": self.config.failure_threshold,
            "recovery_timeout": self.config.recovery_timeout
        }


class CircuitOpenError(Exception):
    """Exception raised when circuit is open."""
    pass


class CircuitBreakerManager:
    """Manage multiple circuit breakers for different services."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        success_threshold: int = 2
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        if name not in self.breakers:
            config = CircuitBreakerConfig(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception,
                success_threshold=success_threshold,
                redis_client=self.redis_client
            )
            self.breakers[name] = CircuitBreaker(config)
        return self.breakers[name]
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {name: breaker.get_status() for name, breaker in self.breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.reset()
    
    def reset(self, name: str):
        """Reset specific circuit breaker."""
        if name in self.breakers:
            self.breakers[name].reset()


# Decorator for applying circuit breaker to functions
def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type = Exception,
    manager: Optional[CircuitBreakerManager] = None
):
    """
    Decorator to apply circuit breaker pattern to a function.
    
    Args:
        name: Circuit breaker name
        failure_threshold: Number of failures before opening
        recovery_timeout: Seconds before attempting reset
        expected_exception: Exception type to catch
        manager: Circuit breaker manager instance
    """
    def decorator(func):
        # Create default manager if not provided
        nonlocal manager
        if manager is None:
            manager = CircuitBreakerManager()
        
        # Get or create circuit breaker
        breaker = manager.get_or_create(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception
        )
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                return await breaker.async_call(func, *args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                return breaker.call(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator


# Global circuit breaker manager instance
_global_manager = None


def get_circuit_breaker_manager(redis_client: Optional[redis.Redis] = None) -> CircuitBreakerManager:
    """Get global circuit breaker manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = CircuitBreakerManager(redis_client)
    return _global_manager