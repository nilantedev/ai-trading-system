#!/usr/bin/env python3
"""
Production-ready retry mechanism with exponential backoff and jitter.
Prevents thundering herd problem and handles transient failures gracefully.
"""

import asyncio
import random
import time
import logging
from typing import Optional, Callable, Any, TypeVar, Union
from functools import wraps
import httpx

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: tuple = (Exception,),
        retry_on_status_codes: tuple = (429, 500, 502, 503, 504),
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on = retry_on
        self.retry_on_status_codes = retry_on_status_codes


def calculate_backoff_delay(
    attempt: int,
    config: RetryConfig
) -> float:
    """
    Calculate delay with exponential backoff and optional jitter.
    
    Args:
        attempt: Current attempt number (0-based)
        config: Retry configuration
        
    Returns:
        Delay in seconds
    """
    # Exponential backoff
    delay = min(
        config.initial_delay * (config.exponential_base ** attempt),
        config.max_delay
    )
    
    # Add jitter to prevent thundering herd
    if config.jitter:
        delay = delay * (0.5 + random.random())
    
    return delay


def should_retry(exception: Exception, config: RetryConfig) -> bool:
    """
    Determine if an exception should trigger a retry.
    
    Args:
        exception: The exception that occurred
        config: Retry configuration
        
    Returns:
        True if should retry, False otherwise
    """
    # Check if exception type should be retried
    if not any(isinstance(exception, exc_type) for exc_type in config.retry_on):
        return False
    
    # Check HTTP status codes for HTTPError
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code in config.retry_on_status_codes
    
    return True


def retry_with_backoff(
    func: Optional[Callable] = None,
    *,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: tuple = (Exception,),
    retry_on_status_codes: tuple = (429, 500, 502, 503, 504),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        func: Function to wrap
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Add randomization to prevent thundering herd
        retry_on: Tuple of exception types to retry on
        retry_on_status_codes: HTTP status codes to retry on
        on_retry: Callback function called on each retry
    
    Returns:
        Decorated function
    """
    config = RetryConfig(
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retry_on=retry_on,
        retry_on_status_codes=retry_on_status_codes
    )
    
    def decorator(f):
        if asyncio.iscoroutinefunction(f):
            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                return await async_retry_with_backoff(
                    f, args, kwargs, config, on_retry
                )
            return async_wrapper
        else:
            @wraps(f)
            def sync_wrapper(*args, **kwargs):
                return sync_retry_with_backoff(
                    f, args, kwargs, config, on_retry
                )
            return sync_wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


def sync_retry_with_backoff(
    func: Callable,
    args: tuple,
    kwargs: dict,
    config: RetryConfig,
    on_retry: Optional[Callable] = None
) -> Any:
    """
    Synchronous retry with exponential backoff.
    
    Args:
        func: Function to retry
        args: Function arguments
        kwargs: Function keyword arguments
        config: Retry configuration
        on_retry: Callback for retry events
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries exhausted
    """
    last_exception = None
    
    for attempt in range(config.max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if attempt >= config.max_retries:
                logger.error(f"All {config.max_retries} retries exhausted for {func.__name__}")
                raise
            
            if not should_retry(e, config):
                logger.debug(f"Exception {type(e).__name__} not retryable")
                raise
            
            delay = calculate_backoff_delay(attempt, config)
            logger.warning(
                f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} "
                f"after {delay:.2f}s delay. Error: {str(e)}"
            )
            
            if on_retry:
                try:
                    on_retry(e, attempt)
                except Exception as callback_error:
                    logger.error(f"Error in retry callback: {callback_error}")
            
            time.sleep(delay)
    
    raise last_exception


async def async_retry_with_backoff(
    func: Callable,
    args: tuple,
    kwargs: dict,
    config: RetryConfig,
    on_retry: Optional[Callable] = None
) -> Any:
    """
    Asynchronous retry with exponential backoff.
    
    Args:
        func: Async function to retry
        args: Function arguments
        kwargs: Function keyword arguments
        config: Retry configuration
        on_retry: Callback for retry events
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries exhausted
    """
    last_exception = None
    
    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if attempt >= config.max_retries:
                logger.error(f"All {config.max_retries} retries exhausted for {func.__name__}")
                raise
            
            if not should_retry(e, config):
                logger.debug(f"Exception {type(e).__name__} not retryable")
                raise
            
            delay = calculate_backoff_delay(attempt, config)
            logger.warning(
                f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} "
                f"after {delay:.2f}s delay. Error: {str(e)}"
            )
            
            if on_retry:
                try:
                    if asyncio.iscoroutinefunction(on_retry):
                        await on_retry(e, attempt)
                    else:
                        on_retry(e, attempt)
                except Exception as callback_error:
                    logger.error(f"Error in retry callback: {callback_error}")
            
            await asyncio.sleep(delay)
    
    raise last_exception


class RetryableHTTPClient:
    """HTTP client with built-in retry logic."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        retry_config: Optional[RetryConfig] = None
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            follow_redirects=True
        )
    
    @retry_with_backoff()
    async def get(self, url: str, **kwargs) -> httpx.Response:
        """GET request with retry logic."""
        response = await self.client.get(url, **kwargs)
        response.raise_for_status()
        return response
    
    @retry_with_backoff()
    async def post(self, url: str, **kwargs) -> httpx.Response:
        """POST request with retry logic."""
        response = await self.client.post(url, **kwargs)
        response.raise_for_status()
        return response
    
    @retry_with_backoff()
    async def put(self, url: str, **kwargs) -> httpx.Response:
        """PUT request with retry logic."""
        response = await self.client.put(url, **kwargs)
        response.raise_for_status()
        return response
    
    @retry_with_backoff()
    async def delete(self, url: str, **kwargs) -> httpx.Response:
        """DELETE request with retry logic."""
        response = await self.client.delete(url, **kwargs)
        response.raise_for_status()
        return response
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Specialized retry configurations for different scenarios
class RetryProfiles:
    """Pre-configured retry profiles for common scenarios."""
    
    # Fast retry for low-latency operations
    FAST = RetryConfig(
        max_retries=2,
        initial_delay=0.1,
        max_delay=1.0,
        exponential_base=2.0
    )
    
    # Standard retry for API calls
    STANDARD = RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0
    )
    
    # Aggressive retry for critical operations
    AGGRESSIVE = RetryConfig(
        max_retries=5,
        initial_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0
    )
    
    # Rate-limited API retry
    RATE_LIMITED = RetryConfig(
        max_retries=5,
        initial_delay=5.0,
        max_delay=300.0,
        exponential_base=2.0,
        retry_on_status_codes=(429,)  # Only retry on rate limit
    )
    
    # Database connection retry
    DATABASE = RetryConfig(
        max_retries=3,
        initial_delay=0.5,
        max_delay=5.0,
        exponential_base=2.0,
        retry_on=(ConnectionError, TimeoutError)
    )