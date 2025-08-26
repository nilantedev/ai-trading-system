#!/usr/bin/env python3
"""
Centralized resilient HTTP client for external API calls.
Provides consistent retry, circuit breaking, rate limiting, and monitoring.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

from .resilience import (
    CircuitBreaker, CircuitBreakerConfig, RetryStrategy, RetryConfig,
    RateLimiter, BulkheadPool, get_circuit_breaker
)
try:
    from .metrics import increment_counter, histogram_observe
    from .logging import get_logger
except ImportError:
    # Fallback for import issues
    def increment_counter(name, labels=None):
        pass
    def histogram_observe(name, value, labels=None):
        pass
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)


class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class HTTPClientConfig:
    """Configuration for resilient HTTP client."""
    # Basic settings
    timeout: float = 30.0
    max_redirects: int = 10
    
    # Resilience settings
    circuit_breaker: Optional[CircuitBreakerConfig] = None
    retry_config: Optional[RetryConfig] = None
    rate_limit_per_minute: Optional[int] = None
    max_concurrent_requests: int = 50
    
    # Monitoring
    enable_metrics: bool = True
    enable_request_logging: bool = True
    
    # Headers
    default_headers: Dict[str, str] = field(default_factory=dict)
    
    # SSL/TLS settings
    verify_ssl: bool = True
    ssl_cert_file: Optional[str] = None
    ssl_key_file: Optional[str] = None
    

@dataclass
class HTTPResponse:
    """HTTP response wrapper with additional metadata."""
    status: int
    headers: Dict[str, str]
    body: Union[str, bytes, dict]
    url: str
    method: str
    elapsed_time: float
    attempt_count: int
    from_cache: bool = False
    
    @property
    def is_success(self) -> bool:
        """Check if response is successful (2xx)."""
        return 200 <= self.status < 300
    
    @property
    def is_client_error(self) -> bool:
        """Check if response is client error (4xx)."""
        return 400 <= self.status < 500
    
    @property
    def is_server_error(self) -> bool:
        """Check if response is server error (5xx)."""
        return 500 <= self.status < 600


class ResilientHTTPClient:
    """Centralized resilient HTTP client with comprehensive fault tolerance."""
    
    def __init__(self, name: str, config: Optional[HTTPClientConfig] = None):
        """Initialize resilient HTTP client."""
        self.name = name
        self.config = config or HTTPClientConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Initialize resilience components
        self._init_resilience_components()
        
        # Request/response cache for debugging
        self._request_cache: Dict[str, Any] = {}
        self._response_cache: Dict[str, HTTPResponse] = {}
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'circuit_breaker_opens': 0,
            'retries_attempted': 0,
            'rate_limit_hits': 0
        }
    
    def _init_resilience_components(self):
        """Initialize resilience components."""
        # Circuit breaker
        cb_config = self.config.circuit_breaker or CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60,
            success_threshold=2
        )
        self.circuit_breaker = get_circuit_breaker(f"{self.name}_http_client", cb_config)
        
        # Retry strategy
        retry_config = self.config.retry_config or RetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True
        )
        self.retry_strategy = RetryStrategy(retry_config)
        
        # Rate limiter (if configured)
        self.rate_limiter = None
        if self.config.rate_limit_per_minute:
            self.rate_limiter = RateLimiter(
                rate=self.config.rate_limit_per_minute,
                per=60.0  # per minute
            )
        
        # Bulkhead for concurrent request limiting
        self.bulkhead = BulkheadPool(self.config.max_concurrent_requests)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def start(self):
        """Initialize HTTP session and components."""
        if self.session is None:
            # Configure SSL context
            ssl_context = None
            if not self.config.verify_ssl:
                ssl_context = False
            elif self.config.ssl_cert_file and self.config.ssl_key_file:
                import ssl
                ssl_context = ssl.create_default_context()
                ssl_context.load_cert_chain(self.config.ssl_cert_file, self.config.ssl_key_file)
            
            # Configure timeout
            timeout = aiohttp.ClientTimeout(
                total=self.config.timeout,
                connect=10.0,
                sock_read=self.config.timeout
            )
            
            # Create session
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.config.default_headers,
                max_redirects=self.config.max_redirects,
                connector=aiohttp.TCPConnector(
                    ssl=ssl_context,
                    limit=self.config.max_concurrent_requests,
                    limit_per_host=10
                )
            )
            
            logger.info(f"HTTP client '{self.name}' initialized")
    
    async def close(self):
        """Close HTTP session and cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info(f"HTTP client '{self.name}' closed")
    
    async def get(self, url: str, **kwargs) -> HTTPResponse:
        """Make GET request."""
        return await self.request(HTTPMethod.GET, url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> HTTPResponse:
        """Make POST request."""
        return await self.request(HTTPMethod.POST, url, **kwargs)
    
    async def put(self, url: str, **kwargs) -> HTTPResponse:
        """Make PUT request.""" 
        return await self.request(HTTPMethod.PUT, url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> HTTPResponse:
        """Make DELETE request."""
        return await self.request(HTTPMethod.DELETE, url, **kwargs)
    
    async def request(
        self,
        method: HTTPMethod,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        **kwargs
    ) -> HTTPResponse:
        """Make resilient HTTP request with all fault tolerance patterns applied."""
        if not self.session:
            await self.start()
        
        # Apply rate limiting if configured
        if self.rate_limiter:
            if not await self.rate_limiter.acquire():
                self.stats['rate_limit_hits'] += 1
                if self.config.enable_metrics:
                    increment_counter('http_client_rate_limit_hits', {'client': self.name})
                await self.rate_limiter.wait_and_acquire()
        
        # Execute request through bulkhead pattern
        return await self.bulkhead.execute(self._make_resilient_request, method, url, headers, params, json_data, data, **kwargs)
    
    async def _make_resilient_request(
        self,
        method: HTTPMethod,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        **kwargs
    ) -> HTTPResponse:
        """Make request with circuit breaker and retry protection."""
        start_time = time.time()
        
        # Merge headers
        request_headers = {**self.config.default_headers}
        if headers:
            request_headers.update(headers)
        
        # Log request if enabled
        if self.config.enable_request_logging:
            logger.debug(f"HTTP {method.value} {url}", extra={
                'http_method': method.value,
                'http_url': url,
                'client_name': self.name
            })
        
        # Define the actual HTTP call
        async def make_http_call():
            return await self._execute_http_request(
                method, url, request_headers, params, json_data, data, **kwargs
            )
        
        # Apply circuit breaker and retry patterns
        try:
            response = await self.circuit_breaker.call(
                lambda: self.retry_strategy.execute(make_http_call)
            )
            
            self.stats['total_requests'] += 1
            self.stats['successful_requests'] += 1
            
            # Record metrics
            if self.config.enable_metrics:
                elapsed = time.time() - start_time
                increment_counter('http_client_requests_total', {
                    'client': self.name, 
                    'method': method.value, 
                    'status': str(response.status)
                })
                histogram_observe('http_client_request_duration_seconds', elapsed, {
                    'client': self.name,
                    'method': method.value
                })
            
            return response
            
        except Exception as e:
            self.stats['total_requests'] += 1
            self.stats['failed_requests'] += 1
            
            # Record failure metrics
            if self.config.enable_metrics:
                increment_counter('http_client_requests_failed_total', {
                    'client': self.name,
                    'method': method.value,
                    'error_type': type(e).__name__
                })
            
            logger.error(f"HTTP {method.value} {url} failed: {e}", extra={
                'http_method': method.value,
                'http_url': url,
                'client_name': self.name,
                'error': str(e)
            })
            raise
    
    async def _execute_http_request(
        self,
        method: HTTPMethod,
        url: str,
        headers: Dict[str, str],
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        **kwargs
    ) -> HTTPResponse:
        """Execute the actual HTTP request."""
        start_time = time.time()
        
        try:
            # Prepare request arguments
            request_kwargs = {
                'headers': headers,
                'params': params,
                **kwargs
            }
            
            if json_data is not None:
                request_kwargs['json'] = json_data
            elif data is not None:
                request_kwargs['data'] = data
            
            # Make the HTTP request
            async with self.session.request(method.value, url, **request_kwargs) as response:
                # Read response body
                content_type = response.headers.get('content-type', '').lower()
                
                if 'application/json' in content_type:
                    try:
                        body = await response.json()
                    except (json.JSONDecodeError, aiohttp.ContentTypeError):
                        body = await response.text()
                elif 'text/' in content_type:
                    body = await response.text()
                else:
                    body = await response.read()
                
                elapsed_time = time.time() - start_time
                
                # Create response object
                http_response = HTTPResponse(
                    status=response.status,
                    headers=dict(response.headers),
                    body=body,
                    url=str(response.url),
                    method=method.value,
                    elapsed_time=elapsed_time,
                    attempt_count=1  # Will be updated by retry strategy
                )
                
                # Log response
                if self.config.enable_request_logging:
                    logger.debug(f"HTTP {method.value} {url} -> {response.status} in {elapsed_time:.3f}s")
                
                # Check for HTTP errors
                if http_response.is_server_error:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"Server error: {response.status}"
                    )
                
                return http_response
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # These are retryable errors
            elapsed_time = time.time() - start_time
            logger.warning(f"HTTP {method.value} {url} error: {e} (elapsed: {elapsed_time:.3f}s)")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            'name': self.name,
            'stats': self.stats.copy(),
            'circuit_breaker': self.circuit_breaker.get_state(),
            'bulkhead': self.bulkhead.get_status(),
            'rate_limiter': {
                'tokens_available': self.rate_limiter.tokens if self.rate_limiter else None,
                'rate_per_minute': self.config.rate_limit_per_minute
            } if self.rate_limiter else None
        }
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {key: 0 for key in self.stats}


# Global client registry
_http_clients: Dict[str, ResilientHTTPClient] = {}


def get_http_client(name: str, config: Optional[HTTPClientConfig] = None) -> ResilientHTTPClient:
    """Get or create a named HTTP client."""
    if name not in _http_clients:
        _http_clients[name] = ResilientHTTPClient(name, config)
    return _http_clients[name]


def get_all_http_clients() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all HTTP clients."""
    return {name: client.get_stats() for name, client in _http_clients.items()}


async def close_all_http_clients():
    """Close all HTTP clients - useful for cleanup."""
    for client in _http_clients.values():
        await client.close()
    _http_clients.clear()


# Convenience functions for common configurations
def get_trading_api_client() -> ResilientHTTPClient:
    """Get HTTP client optimized for trading APIs."""
    config = HTTPClientConfig(
        timeout=15.0,
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30,
            success_threshold=2
        ),
        retry_config=RetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=15.0,
            exponential_base=2.0
        ),
        rate_limit_per_minute=100,  # Conservative for trading APIs
        max_concurrent_requests=20
    )
    return get_http_client("trading_api", config)


def get_market_data_client() -> ResilientHTTPClient:
    """Get HTTP client optimized for market data APIs."""
    config = HTTPClientConfig(
        timeout=10.0,
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=20,
            success_threshold=2
        ),
        retry_config=RetryConfig(
            max_attempts=2,  # Fast retry for real-time data
            initial_delay=0.5,
            max_delay=5.0,
            exponential_base=1.5
        ),
        rate_limit_per_minute=300,  # Higher limits for data feeds
        max_concurrent_requests=50
    )
    return get_http_client("market_data", config)


def get_news_api_client() -> ResilientHTTPClient:
    """Get HTTP client optimized for news APIs."""
    config = HTTPClientConfig(
        timeout=20.0,
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60,
            success_threshold=1
        ),
        retry_config=RetryConfig(
            max_attempts=2,
            initial_delay=2.0,
            max_delay=30.0,
            exponential_base=2.0
        ),
        rate_limit_per_minute=60,  # News APIs often have strict limits
        max_concurrent_requests=10
    )
    return get_http_client("news_api", config)