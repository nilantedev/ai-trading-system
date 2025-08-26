#!/usr/bin/env python3
"""
Centralized API client factory for external services.
Provides pre-configured resilient clients for all external APIs.
"""

from typing import Optional, Dict, Any
from datetime import timedelta

from .http_client import ResilientHTTPClient, HTTPClientConfig

from .resilience import CircuitBreakerConfig, RetryConfig
try:
    from .logging import get_logger
    from . import get_settings
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    # Try direct import
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(parent_dir)
    from trading_common import get_settings

logger = get_logger(__name__)
settings = get_settings()


class APIClientFactory:
    """Factory for creating pre-configured API clients for external services."""
    
    @staticmethod
    def create_alpaca_client() -> ResilientHTTPClient:
        """Create Alpaca API client with optimized settings."""
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
                exponential_base=2.0,
                jitter=True
            ),
            rate_limit_per_minute=200,  # Alpaca has generous limits
            max_concurrent_requests=20,
            default_headers={
                'User-Agent': 'AI-Trading-System/1.0',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        )
        
        client = ResilientHTTPClient("alpaca_api", config)
        logger.info("Created Alpaca API client with resilience patterns")
        return client
    
    @staticmethod
    def create_polygon_client() -> ResilientHTTPClient:
        """Create Polygon.io API client with optimized settings."""
        config = HTTPClientConfig(
            timeout=10.0,
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=4,
                recovery_timeout=45,
                success_threshold=2
            ),
            retry_config=RetryConfig(
                max_attempts=3,
                initial_delay=1.5,
                max_delay=20.0,
                exponential_base=2.0,
                jitter=True
            ),
            rate_limit_per_minute=300,  # Polygon has good limits
            max_concurrent_requests=30,
            default_headers={
                'User-Agent': 'AI-Trading-System/1.0',
                'Accept': 'application/json'
            }
        )
        
        client = ResilientHTTPClient("polygon_api", config)
        logger.info("Created Polygon API client with resilience patterns")
        return client
    
    @staticmethod
    def create_alpha_vantage_client() -> ResilientHTTPClient:
        """Create Alpha Vantage API client with conservative settings."""
        config = HTTPClientConfig(
            timeout=30.0,
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=2,  # More conservative due to rate limits
                recovery_timeout=120,  # Longer recovery time
                success_threshold=1
            ),
            retry_config=RetryConfig(
                max_attempts=2,
                initial_delay=5.0,
                max_delay=60.0,
                exponential_base=2.0,
                jitter=True
            ),
            rate_limit_per_minute=5,  # Very strict rate limits
            max_concurrent_requests=2,
            default_headers={
                'User-Agent': 'AI-Trading-System/1.0',
                'Accept': 'application/json'
            }
        )
        
        client = ResilientHTTPClient("alpha_vantage_api", config)
        logger.info("Created Alpha Vantage API client with conservative resilience patterns")
        return client
    
    @staticmethod
    def create_news_api_client() -> ResilientHTTPClient:
        """Create NewsAPI client with appropriate settings."""
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
                exponential_base=2.0,
                jitter=True
            ),
            rate_limit_per_minute=60,  # Conservative for news APIs
            max_concurrent_requests=10,
            default_headers={
                'User-Agent': 'AI-Trading-System/1.0',
                'Accept': 'application/json'
            }
        )
        
        client = ResilientHTTPClient("news_api", config)
        logger.info("Created News API client with resilience patterns")
        return client
    
    @staticmethod 
    def create_reddit_client() -> ResilientHTTPClient:
        """Create Reddit API client with rate limit considerations."""
        config = HTTPClientConfig(
            timeout=15.0,
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=120,  # Reddit can have longer recovery times
                success_threshold=2
            ),
            retry_config=RetryConfig(
                max_attempts=2,
                initial_delay=3.0,
                max_delay=30.0,
                exponential_base=2.0,
                jitter=True
            ),
            rate_limit_per_minute=60,  # Reddit has strict rate limits
            max_concurrent_requests=5,
            default_headers={
                'User-Agent': 'AI-Trading-System/1.0 by /u/trading_bot',
                'Accept': 'application/json'
            }
        )
        
        client = ResilientHTTPClient("reddit_api", config)
        logger.info("Created Reddit API client with strict rate limiting")
        return client
    
    @staticmethod
    def create_twitter_client() -> ResilientHTTPClient:
        """Create Twitter/X API client with v2 API settings."""
        config = HTTPClientConfig(
            timeout=15.0,
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=2,  # Twitter can be strict
                recovery_timeout=180,  # Long recovery for rate limits
                success_threshold=2
            ),
            retry_config=RetryConfig(
                max_attempts=2,
                initial_delay=5.0,
                max_delay=60.0,
                exponential_base=2.0,
                jitter=True
            ),
            rate_limit_per_minute=300,  # Depends on API tier
            max_concurrent_requests=10,
            default_headers={
                'User-Agent': 'AI-Trading-System/1.0',
                'Accept': 'application/json'
            }
        )
        
        client = ResilientHTTPClient("twitter_api", config)
        logger.info("Created Twitter API client with rate limit protection")
        return client
    
    @staticmethod
    def create_openai_client() -> ResilientHTTPClient:
        """Create OpenAI API client for AI/LLM calls."""
        config = HTTPClientConfig(
            timeout=60.0,  # AI calls can take longer
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=60,
                success_threshold=2
            ),
            retry_config=RetryConfig(
                max_attempts=3,
                initial_delay=2.0,
                max_delay=30.0,
                exponential_base=2.0,
                jitter=True
            ),
            rate_limit_per_minute=60,  # Conservative for AI APIs
            max_concurrent_requests=5,
            default_headers={
                'User-Agent': 'AI-Trading-System/1.0',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        )
        
        client = ResilientHTTPClient("openai_api", config)
        logger.info("Created OpenAI API client with extended timeout")
        return client
    
    @staticmethod
    def create_anthropic_client() -> ResilientHTTPClient:
        """Create Anthropic Claude API client."""
        config = HTTPClientConfig(
            timeout=60.0,
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=60,
                success_threshold=2
            ),
            retry_config=RetryConfig(
                max_attempts=3,
                initial_delay=2.0,
                max_delay=30.0,
                exponential_base=2.0,
                jitter=True
            ),
            rate_limit_per_minute=50,  # Conservative for Anthropic
            max_concurrent_requests=3,
            default_headers={
                'User-Agent': 'AI-Trading-System/1.0',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'anthropic-version': '2023-06-01'
            }
        )
        
        client = ResilientHTTPClient("anthropic_api", config)
        logger.info("Created Anthropic API client with extended timeout")
        return client
    
    @staticmethod
    def create_yahoo_finance_client() -> ResilientHTTPClient:
        """Create Yahoo Finance API client (unofficial API)."""
        config = HTTPClientConfig(
            timeout=10.0,
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=5,  # Yahoo can be less reliable
                recovery_timeout=60,
                success_threshold=3
            ),
            retry_config=RetryConfig(
                max_attempts=3,
                initial_delay=1.0,
                max_delay=20.0,
                exponential_base=2.0,
                jitter=True
            ),
            rate_limit_per_minute=100,
            max_concurrent_requests=20,
            default_headers={
                'User-Agent': 'Mozilla/5.0 (compatible; AI-Trading-System/1.0)',
                'Accept': 'application/json'
            }
        )
        
        client = ResilientHTTPClient("yahoo_finance_api", config)
        logger.info("Created Yahoo Finance API client with higher fault tolerance")
        return client
    
    @staticmethod
    def create_sec_edgar_client() -> ResilientHTTPClient:
        """Create SEC EDGAR API client for filings."""
        config = HTTPClientConfig(
            timeout=30.0,
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=120,
                success_threshold=2
            ),
            retry_config=RetryConfig(
                max_attempts=2,
                initial_delay=5.0,
                max_delay=60.0,
                exponential_base=2.0,
                jitter=True
            ),
            rate_limit_per_minute=10,  # SEC has strict rate limits
            max_concurrent_requests=2,
            default_headers={
                'User-Agent': 'AI-Trading-System admin@example.com',  # SEC requires contact info
                'Accept': 'application/json',
                'Host': 'www.sec.gov'
            }
        )
        
        client = ResilientHTTPClient("sec_edgar_api", config)
        logger.info("Created SEC EDGAR API client with strict rate limiting")
        return client
    
    @staticmethod
    def create_generic_client(
        name: str,
        timeout: float = 30.0,
        rate_limit_per_minute: Optional[int] = None,
        max_concurrent: int = 20,
        failure_threshold: int = 3,
        recovery_timeout: int = 60
    ) -> ResilientHTTPClient:
        """Create a generic API client with custom settings."""
        config = HTTPClientConfig(
            timeout=timeout,
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                success_threshold=2
            ),
            retry_config=RetryConfig(
                max_attempts=3,
                initial_delay=1.0,
                max_delay=30.0,
                exponential_base=2.0,
                jitter=True
            ),
            rate_limit_per_minute=rate_limit_per_minute,
            max_concurrent_requests=max_concurrent,
            default_headers={
                'User-Agent': 'AI-Trading-System/1.0',
                'Accept': 'application/json'
            }
        )
        
        client = ResilientHTTPClient(name, config)
        logger.info(f"Created generic API client '{name}' with custom resilience patterns")
        return client


# Pre-configured client instances (lazy-loaded)
_client_cache: Dict[str, ResilientHTTPClient] = {}


def get_alpaca_client() -> ResilientHTTPClient:
    """Get cached Alpaca API client."""
    if 'alpaca' not in _client_cache:
        _client_cache['alpaca'] = APIClientFactory.create_alpaca_client()
    return _client_cache['alpaca']


def get_polygon_client() -> ResilientHTTPClient:
    """Get cached Polygon API client."""
    if 'polygon' not in _client_cache:
        _client_cache['polygon'] = APIClientFactory.create_polygon_client()
    return _client_cache['polygon']


def get_alpha_vantage_client() -> ResilientHTTPClient:
    """Get cached Alpha Vantage API client."""
    if 'alpha_vantage' not in _client_cache:
        _client_cache['alpha_vantage'] = APIClientFactory.create_alpha_vantage_client()
    return _client_cache['alpha_vantage']


def get_news_client() -> ResilientHTTPClient:
    """Get cached News API client."""
    if 'news' not in _client_cache:
        _client_cache['news'] = APIClientFactory.create_news_api_client()
    return _client_cache['news']


def get_reddit_client() -> ResilientHTTPClient:
    """Get cached Reddit API client."""
    if 'reddit' not in _client_cache:
        _client_cache['reddit'] = APIClientFactory.create_reddit_client()
    return _client_cache['reddit']


def get_openai_client() -> ResilientHTTPClient:
    """Get cached OpenAI API client."""
    if 'openai' not in _client_cache:
        _client_cache['openai'] = APIClientFactory.create_openai_client()
    return _client_cache['openai']


async def cleanup_all_clients():
    """Close all cached API clients."""
    for client in _client_cache.values():
        await client.close()
    _client_cache.clear()
    logger.info("All API clients closed and cache cleared")