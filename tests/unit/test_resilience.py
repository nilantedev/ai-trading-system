#!/usr/bin/env python3
"""
Comprehensive tests for resilience patterns.
Tests circuit breakers, retry strategies, and reliability patterns.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock
import os

# Add parent directories to path for imports  
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_common.resilience import (
    CircuitBreaker, CircuitState, CircuitBreakerConfig,
    RetryStrategy, RetryConfig, RateLimiter, BulkheadPool,
    with_retry, with_circuit_breaker, get_circuit_breaker
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1)
        self.breaker = CircuitBreaker("test_breaker", self.config)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state allows calls."""
        async def success_func():
            return "success"
        
        result = await self.breaker.call(success_func)
        assert result == "success"
        assert self.breaker.state.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after failure threshold."""
        async def failing_func():
            raise Exception("Test failure")
        
        # Trigger failures up to threshold
        for i in range(3):
            with pytest.raises(Exception):
                await self.breaker.call(failing_func)
        
        # Circuit should now be open
        assert self.breaker.state.state == CircuitState.OPEN
        
        # Next call should fail immediately without calling function
        with pytest.raises(Exception) as exc_info:
            await self.breaker.call(failing_func)
        
        assert "Circuit breaker" in str(exc_info.value)
        assert "OPEN" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        async def failing_func():
            raise Exception("Test failure")
        
        async def success_func():
            return "success"
        
        # Trigger failures to open circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await self.breaker.call(failing_func)
        
        assert self.breaker.state.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # First call after timeout should move to half-open
        result = await self.breaker.call(success_func)
        assert result == "success"
        
        # Another success should close the circuit
        result = await self.breaker.call(success_func)
        assert result == "success"
        assert self.breaker.state.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker reopens on failure in half-open state."""
        async def failing_func():
            raise Exception("Test failure")
        
        async def success_func():
            return "success"
        
        # Open circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await self.breaker.call(failing_func)
        
        # Wait for recovery
        await asyncio.sleep(1.1)
        
        # Fail in half-open state
        with pytest.raises(Exception):
            await self.breaker.call(failing_func)
        
        # Should be open again
        assert self.breaker.state.state == CircuitState.OPEN
    
    def test_get_circuit_breaker_state(self):
        """Test getting circuit breaker state information."""
        state = self.breaker.get_state()
        
        assert state["name"] == "test_breaker"
        assert state["state"] == "closed"
        assert state["failure_count"] == 0
        assert state["success_count"] == 0
    
    def test_global_circuit_breaker_registry(self):
        """Test global circuit breaker registry."""
        breaker1 = get_circuit_breaker("test_global")
        breaker2 = get_circuit_breaker("test_global")
        
        # Should return same instance
        assert breaker1 is breaker2


class TestRetryStrategy:
    """Test retry strategy functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.config = RetryConfig(max_attempts=3, initial_delay=0.1, max_delay=1.0)
        self.strategy = RetryStrategy(self.config)
    
    @pytest.mark.asyncio
    async def test_retry_success_on_first_attempt(self):
        """Test that successful functions don't retry."""
        call_count = 0
        
        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await self.strategy.execute(success_func)
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self):
        """Test retry succeeds after some failures."""
        call_count = 0
        
        async def intermittent_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Failure {call_count}")
            return "success"
        
        result = await self.strategy.execute(intermittent_func)
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        """Test retry gives up after max attempts."""
        call_count = 0
        
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise Exception(f"Failure {call_count}")
        
        with pytest.raises(Exception) as exc_info:
            await self.strategy.execute(always_fails)
        
        assert "Failure 3" in str(exc_info.value)
        assert call_count == 3
    
    def test_retry_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        delays = [self.strategy.get_delay(i) for i in range(5)]
        
        # Should increase exponentially (with jitter variation)
        assert delays[0] < delays[1] < delays[2]
        
        # Should not exceed max delay
        long_delay = self.strategy.get_delay(10)
        assert long_delay <= self.config.max_delay
    
    @pytest.mark.asyncio
    async def test_retry_decorator_async(self):
        """Test retry decorator on async functions."""
        call_count = 0
        
        @with_retry(max_attempts=3, backoff_base=2.0)
        async def decorated_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Retry needed")
            return "success"
        
        result = await decorated_func()
        
        assert result == "success"
        assert call_count == 2
    
    def test_retry_decorator_sync(self):
        """Test retry decorator on sync functions."""
        call_count = 0
        
        @with_retry(max_attempts=3, backoff_base=2.0)
        def decorated_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Retry needed")
            return "success"
        
        result = decorated_func()
        
        assert result == "success"
        assert call_count == 2


class TestRateLimiter:
    """Test token bucket rate limiter."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_within_rate(self):
        """Test rate limiter allows requests within rate."""
        limiter = RateLimiter(rate=10, per=1.0)  # 10 per second
        
        # Should allow first requests
        for _ in range(5):
            result = await limiter.acquire()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_over_rate(self):
        """Test rate limiter blocks requests over rate."""
        limiter = RateLimiter(rate=2, per=1.0)  # 2 per second
        
        # Consume all tokens
        assert await limiter.acquire() is True
        assert await limiter.acquire() is True
        
        # Should be blocked now
        assert await limiter.acquire() is False
    
    @pytest.mark.asyncio
    async def test_rate_limiter_refill(self):
        """Test rate limiter token refill over time."""
        limiter = RateLimiter(rate=2, per=0.1)  # 2 per 100ms = 20 per second
        
        # Consume all tokens
        assert await limiter.acquire() is True
        assert await limiter.acquire() is True
        
        # Should be blocked
        assert await limiter.acquire() is False
        
        # Wait for refill
        await asyncio.sleep(0.11)
        
        # Should be allowed again
        assert await limiter.acquire() is True
    
    @pytest.mark.asyncio
    async def test_rate_limiter_wait_and_acquire(self):
        """Test rate limiter wait and acquire functionality."""
        limiter = RateLimiter(rate=1, per=0.1)  # 1 per 100ms
        
        start_time = time.time()
        
        # First acquire should be immediate
        await limiter.wait_and_acquire()
        
        # Second acquire should wait
        await limiter.wait_and_acquire()
        
        elapsed = time.time() - start_time
        assert elapsed >= 0.1  # Should have waited at least 100ms


class TestBulkheadPool:
    """Test bulkhead pattern for resource isolation."""
    
    @pytest.mark.asyncio
    async def test_bulkhead_allows_concurrent_execution(self):
        """Test bulkhead allows concurrent execution up to limit."""
        pool = BulkheadPool(max_concurrent=2)
        
        execution_times = []
        
        async def slow_task(task_id):
            start = time.time()
            await asyncio.sleep(0.1)
            end = time.time()
            execution_times.append((task_id, start, end))
            return f"task_{task_id}"
        
        # Start 2 tasks concurrently
        tasks = [
            pool.execute(slow_task, 1),
            pool.execute(slow_task, 2)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 2
        assert "task_1" in results
        assert "task_2" in results
    
    @pytest.mark.asyncio
    async def test_bulkhead_blocks_over_limit(self):
        """Test bulkhead blocks execution over concurrency limit."""
        pool = BulkheadPool(max_concurrent=1)
        
        start_times = []
        
        async def timed_task(task_id):
            start_times.append((task_id, time.time()))
            await asyncio.sleep(0.1)
            return f"task_{task_id}"
        
        # Start 3 tasks - only 1 should run at a time
        tasks = [
            pool.execute(timed_task, 1),
            pool.execute(timed_task, 2),
            pool.execute(timed_task, 3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        
        # Tasks should have started sequentially, not concurrently
        assert len(start_times) == 3
        
        # Check that tasks started in sequence (with some tolerance for timing)
        time_diffs = [
            start_times[i+1][1] - start_times[i][1] 
            for i in range(len(start_times) - 1)
        ]
        
        # Each subsequent task should start significantly later
        for diff in time_diffs:
            assert diff >= 0.05  # At least 50ms gap (less than 100ms sleep for timing tolerance)
    
    def test_bulkhead_status(self):
        """Test bulkhead status reporting."""
        pool = BulkheadPool(max_concurrent=5)
        
        status = pool.get_status()
        
        assert status["active"] == 0
        assert status["available"] == 5


class TestCombinedPatterns:
    """Test combination of resilience patterns."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_with_retry(self):
        """Test circuit breaker combined with retry."""
        call_count = 0
        
        @with_circuit_breaker("combined_test", failure_threshold=2)
        @with_retry(max_attempts=3)
        async def unreliable_service():
            nonlocal call_count
            call_count += 1
            if call_count <= 4:  # Fail first 4 calls
                raise Exception("Service unavailable")
            return "success"
        
        # First set of calls should fail and open circuit
        with pytest.raises(Exception):
            await unreliable_service()
        
        # Circuit should be open, preventing further calls
        breaker = get_circuit_breaker("combined_test")
        assert breaker.state.failure_count >= 2


if __name__ == "__main__":
    # Run with: python -m pytest tests/unit/test_resilience.py -v
    pytest.main([__file__, "-v"])