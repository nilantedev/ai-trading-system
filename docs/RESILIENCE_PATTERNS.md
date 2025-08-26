# Centralized Resilience Patterns

This document explains the comprehensive resilience patterns implemented in the AI Trading System to ensure fault tolerance and reliability when calling external APIs.

## Overview

The system implements multiple resilience patterns to handle failures gracefully:

- **Circuit Breakers**: Prevent cascading failures by temporarily blocking calls to failing services
- **Retry Strategies**: Automatically retry failed requests with exponential backoff and jitter
- **Rate Limiting**: Control request rates to external APIs to avoid rate limit violations
- **Bulkhead Pattern**: Isolate resources to prevent one failing component from affecting others
- **Comprehensive Monitoring**: Track all resilience patterns and provide health insights

## Architecture

### Core Components

1. **Resilience Module** (`trading_common/resilience.py`)
   - Circuit breakers with configurable thresholds
   - Retry strategies with exponential backoff
   - Rate limiters using token bucket algorithm
   - Bulkhead pools for resource isolation

2. **HTTP Client** (`trading_common/http_client.py`)
   - Centralized resilient HTTP client
   - Automatic application of all resilience patterns
   - Comprehensive metrics and logging

3. **API Client Factory** (`trading_common/api_clients.py`)
   - Pre-configured clients for different external services
   - Service-specific resilience settings
   - Lazy loading and caching

4. **Resilience Monitor** (`trading_common/resilience_monitor.py`)
   - Centralized monitoring of all resilience components
   - Alerting system for degraded services
   - Health status reporting

## Usage Examples

### Basic Circuit Breaker

```python
from trading_common.resilience import CircuitBreaker, CircuitBreakerConfig

# Create circuit breaker with custom config
config = CircuitBreakerConfig(
    failure_threshold=5,     # Open after 5 failures
    recovery_timeout=60,     # Try recovery after 60 seconds
    success_threshold=2      # Close after 2 successes
)

circuit_breaker = CircuitBreaker("external_api", config)

# Use circuit breaker
async def call_external_api():
    return await circuit_breaker.call(make_api_request)
```

### Retry Strategy

```python
from trading_common.resilience import RetryStrategy, RetryConfig

# Create retry strategy
retry_config = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True
)

retry_strategy = RetryStrategy(retry_config)

# Use retry strategy
async def resilient_api_call():
    return await retry_strategy.execute(make_api_request)
```

### Using Decorators

```python
from trading_common.resilience import with_retry, with_circuit_breaker

@with_circuit_breaker("news_api", failure_threshold=3)
@with_retry(max_attempts=3)
async def get_news_data(symbol: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/news/{symbol}") as response:
            return await response.json()
```

### Centralized HTTP Client

```python
from trading_common.api_clients import get_alpaca_client, get_news_client

# Get pre-configured clients
alpaca_client = get_alpaca_client()
news_client = get_news_client()

# Make resilient requests
async def get_stock_data():
    # Automatic rate limiting, circuit breaking, retries
    response = await alpaca_client.get("/v2/stocks/AAPL/bars", params={
        "timeframe": "1Day",
        "start": "2023-01-01"
    })
    return response.body

async def get_news():
    response = await news_client.get("/everything", params={
        "q": "AAPL",
        "apiKey": "your-api-key"
    })
    return response.body
```

### Service Integration Example

Here's how the NewsAPI service integrates comprehensive resilience patterns:

```python
class NewsAPIProvider(NewsProviderAPI):
    """NewsAPI provider with comprehensive resilience patterns."""
    
    def __init__(self, config: Dict[str, Any]):
        config.update({
            'provider_name': 'newsapi',
            'rate_limit_per_minute': 100,
            'max_concurrent_requests': 3,
            'failure_threshold': 3,
            'recovery_timeout': 120,
            'max_attempts': 2,
            'initial_delay': 2.0,
            'max_delay': 30.0
        })
        super().__init__(config)  # Initializes all resilience patterns
    
    async def get_news(self, request: NewsRequest) -> List[NewsArticle]:
        try:
            # Make resilient API call (applies all patterns automatically)
            response_data = await self._make_resilient_request(
                'GET',
                f"{self.base_url}/everything",
                params=params
            )
            
            # Process response
            return self._process_articles(response_data)
            
        except Exception as e:
            logger.error(f"NewsAPI get_news failed: {e}")
            return []  # Graceful degradation
```

## Configuration

### Circuit Breaker Configuration

```python
CircuitBreakerConfig(
    failure_threshold=5,      # Number of failures before opening
    recovery_timeout=60,      # Seconds before trying half-open
    success_threshold=2,      # Successes needed to close from half-open
    expected_exception=Exception  # Exception types to track
)
```

### Retry Configuration

```python
RetryConfig(
    max_attempts=3,          # Maximum retry attempts
    initial_delay=1.0,       # Initial delay in seconds
    max_delay=60.0,          # Maximum delay cap
    exponential_base=2.0,    # Backoff multiplier
    jitter=True              # Add randomness to prevent thundering herd
)
```

### HTTP Client Configuration

```python
HTTPClientConfig(
    timeout=30.0,                    # Request timeout
    circuit_breaker=cb_config,       # Circuit breaker config
    retry_config=retry_config,       # Retry config
    rate_limit_per_minute=100,       # Rate limiting
    max_concurrent_requests=50,      # Bulkhead size
    verify_ssl=True,                 # SSL verification
    default_headers={}               # Default headers
)
```

## Monitoring and Alerting

### Health Check Endpoints

The system provides REST API endpoints for monitoring resilience health:

- `GET /api/v1/resilience/health` - Overall health status
- `GET /api/v1/resilience/metrics` - Comprehensive metrics
- `GET /api/v1/resilience/circuit-breakers` - Circuit breaker states
- `GET /api/v1/resilience/alerts` - Active and resolved alerts
- `GET /api/v1/resilience/dashboard` - Complete dashboard data

### Programmatic Monitoring

```python
from trading_common.resilience_monitor import get_resilience_monitor

# Get the global monitor
monitor = get_resilience_monitor()

# Register components for monitoring
monitor.register_component("news_api", news_provider)
monitor.register_component("alpaca_api", alpaca_client)

# Start monitoring
await monitor.start_monitoring()

# Get health status
health = monitor.get_health_status()
print(f"System health: {health.value}")

# Get metrics summary
summary = monitor.get_metrics_summary()
print(f"Success rate: {summary['requests']['success_rate']:.2%}")
```

### Metrics Available

- Circuit breaker states (open/closed/half-open counts)
- Request success/failure rates
- Average response times
- Rate limiting hits
- Bulkhead rejections
- Active alerts

### Alert Conditions

The system automatically generates alerts for:

1. **High Circuit Breaker Open Rate** (>30% open)
2. **High API Failure Rate** (>10% failures)
3. **High Response Times** (>5 seconds average)

## Pre-configured Clients

The system includes pre-configured clients for common external services:

### Trading APIs
- **Alpaca**: `get_alpaca_client()` - Trading and market data
- **Polygon**: `get_polygon_client()` - Real-time market data
- **Alpha Vantage**: `get_alpha_vantage_client()` - Conservative settings for rate limits

### News APIs
- **NewsAPI**: `get_news_client()` - General news
- **Reddit**: `get_reddit_client()` - Social media sentiment

### AI APIs
- **OpenAI**: `get_openai_client()` - Extended timeout for AI calls
- **Anthropic**: `get_anthropic_client()` - Claude API integration

### Other APIs
- **Yahoo Finance**: `get_yahoo_finance_client()` - Higher fault tolerance
- **SEC EDGAR**: `get_sec_edgar_client()` - Strict rate limiting

Each client is optimized with appropriate:
- Timeout settings
- Rate limits
- Circuit breaker thresholds
- Retry strategies
- Concurrent request limits

## Best Practices

### 1. Service-Specific Configuration
Configure resilience patterns based on the external service characteristics:

```python
# Conservative for rate-limited APIs
alpha_vantage_config = {
    'rate_limit_per_minute': 5,
    'max_concurrent_requests': 2,
    'failure_threshold': 2,
    'recovery_timeout': 120
}

# Aggressive for high-availability APIs
polygon_config = {
    'rate_limit_per_minute': 300,
    'max_concurrent_requests': 30,
    'failure_threshold': 5,
    'recovery_timeout': 30
}
```

### 2. Graceful Degradation
Always return safe defaults instead of propagating failures:

```python
async def get_news_sentiment(symbol: str) -> float:
    try:
        news_data = await resilient_news_client.get_news(symbol)
        return calculate_sentiment(news_data)
    except Exception:
        return 0.0  # Neutral sentiment as fallback
```

### 3. Monitor and Alert
Set up monitoring for your resilience patterns:

```python
# Register all your API clients with the monitor
monitor = get_resilience_monitor()
monitor.register_component("my_service", my_api_client)
await monitor.start_monitoring()
```

### 4. Test Failure Scenarios
Regularly test your resilience patterns:

```python
# Test circuit breaker opening
await simulate_api_failures(count=5)
assert circuit_breaker.state.state == CircuitState.OPEN

# Test recovery
await asyncio.sleep(recovery_timeout)
await successful_api_call()
assert circuit_breaker.state.state == CircuitState.CLOSED
```

## Integration with Existing Services

To add resilience patterns to an existing service:

1. **Import the resilience module**:
   ```python
   from trading_common.resilience import get_circuit_breaker, RetryStrategy
   ```

2. **Initialize resilience components** in your service constructor:
   ```python
   def __init__(self, config):
       self.circuit_breaker = get_circuit_breaker("my_service")
       self.retry_strategy = RetryStrategy()
   ```

3. **Wrap external API calls**:
   ```python
   async def call_external_api(self):
       return await self.circuit_breaker.call(
           lambda: self.retry_strategy.execute(self._make_http_call)
       )
   ```

4. **Register with the monitor**:
   ```python
   from trading_common.resilience_monitor import get_resilience_monitor
   get_resilience_monitor().register_component("my_service", self)
   ```

## Troubleshooting

### Circuit Breaker Stuck Open
- Check the error logs for the root cause of failures
- Verify external service availability
- Consider manual reset if needed (requires super admin role)

### High Failure Rates
- Check network connectivity
- Verify API credentials and quotas
- Review rate limiting settings

### Performance Issues
- Monitor response times in the dashboard
- Adjust timeout settings if needed
- Check bulkhead pool sizes

### Rate Limiting
- Review the rate limit configuration
- Check if external service limits have changed
- Consider distributing load across multiple API keys

For more detailed troubleshooting, check the resilience monitoring dashboard at `/api/v1/resilience/dashboard`.