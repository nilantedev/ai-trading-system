"""Resilience utilities (circuit breaker & adaptive token bucket)

These primitives provide standardized resilience patterns across services:
  * AsyncCircuitBreaker: protects fragile dependencies (e.g., Pulsar, external APIs)
  * AdaptiveTokenBucket: dynamic rate limiter adjusting to observed latency / error rate

Design Goals:
  - Non-blocking async operation
  - Minimal external dependencies
  - Introspection metrics hooks (integrate with prometheus-client externally)
  - Deterministic state transitions

Usage Example (Circuit Breaker):
    breaker = AsyncCircuitBreaker(name="pulsar_producer")
    async def send_event(evt):
        async with breaker.context():
            await producer.send(evt)

Usage Example (Adaptive Token Bucket):
    bucket = AdaptiveTokenBucket(rate=50, capacity=100)
    if await bucket.acquire():
        await call_api()

NOTE: Prometheus integration left to caller (expose breaker.state, breaker.metrics_snapshot())
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable, Optional, Dict, Any


class BreakerState(str, Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Short-circuit calls
    HALF_OPEN = "half_open" # Allow limited trial executions


@dataclass
class BreakerConfig:
    failure_threshold: int = 5            # Failures before opening
    recovery_timeout: float = 30.0        # Seconds before HALF_OPEN probing
    half_open_max_calls: int = 3          # Allowed trial calls in HALF_OPEN
    half_open_success_reset: int = 2      # Required consecutive successes to close
    rolling_window: float = 60.0          # Rolling window for failure counter trim


class AsyncCircuitBreaker:
    def __init__(self, name: str, config: Optional[BreakerConfig] = None, time_fn: Callable[[], float] = time.time):
        self.name = name
        self.config = config or BreakerConfig()
        self._state: BreakerState = BreakerState.CLOSED
        self._lock = asyncio.Lock()
        self._events: list[tuple[float, bool]] = []  # (timestamp, success)
        self._last_state_change = time_fn()
        self._half_open_calls = 0
        self._half_open_successes = 0
        self._time_fn = time_fn
        self._listeners: list[Callable[[BreakerState, BreakerState], None]] = []  # (old,new)

    @property
    def state(self) -> BreakerState:
        return self._state

    def _trim(self):
        cutoff = self._time_fn() - self.config.rolling_window
        while self._events and self._events[0][0] < cutoff:
            self._events.pop(0)

    def _failure_count(self) -> int:
        self._trim()
        return sum(1 for _, success in self._events if not success)

    def _record(self, success: bool):
        self._events.append((self._time_fn(), success))
        self._trim()

    def metrics_snapshot(self) -> Dict[str, Any]:
        fails = self._failure_count()
        return {
            "state": self._state.value,
            "failures_window": fails,
            "half_open_calls": self._half_open_calls,
            "half_open_successes": self._half_open_successes,
            "last_state_change_ts": self._last_state_change,
        }

    async def allow(self) -> bool:
        async with self._lock:
            now = self._time_fn()
            if self._state == BreakerState.OPEN:
                if now - self._last_state_change >= self.config.recovery_timeout:
                    self._transition(BreakerState.HALF_OPEN)
                else:
                    return False

            if self._state == BreakerState.HALF_OPEN and self._half_open_calls >= self.config.half_open_max_calls:
                # Pending outcome of in-flight half-open calls
                return False
            if self._state == BreakerState.HALF_OPEN:
                self._half_open_calls += 1
            return True

    def _transition(self, new_state: BreakerState):
        if new_state != self._state:
            old = self._state
            self._state = new_state
            self._last_state_change = self._time_fn()
            if new_state == BreakerState.HALF_OPEN:
                self._half_open_calls = 0
                self._half_open_successes = 0
            # Notify listeners (non-blocking; protect against exceptions)
            for cb in list(self._listeners):  # copy for safety
                try:
                    cb(old, new_state)
                except Exception:
                    pass

    async def on_result(self, success: bool):
        async with self._lock:
            self._record(success)
            if self._state == BreakerState.CLOSED:
                if not success and self._failure_count() >= self.config.failure_threshold:
                    self._transition(BreakerState.OPEN)
            elif self._state == BreakerState.HALF_OPEN:
                if success:
                    self._half_open_successes += 1
                    if self._half_open_successes >= self.config.half_open_success_reset:
                        self._transition(BreakerState.CLOSED)
                else:
                    self._transition(BreakerState.OPEN)

    def add_state_listener(self, callback: Callable[[BreakerState, BreakerState], None]):
        """Register a synchronous listener invoked on state transitions.

        Callback signature: (old_state, new_state)
        """
        self._listeners.append(callback)

    @asynccontextmanager
    async def context(self):
        allowed = await self.allow()
        if not allowed:
            raise RuntimeError(f"CircuitBreaker[{self.name}] open")
        try:
            yield
        except Exception:
            await self.on_result(False)
            raise
        else:
            await self.on_result(True)


@dataclass
class AdaptiveBucketConfig:
    rate: float = 50.0       # tokens per second baseline
    capacity: int = 100      # max bucket depth
    min_rate: float = 5.0
    max_rate: float = 500.0
    error_rate_high: float = 0.2   # raise throttle when errors >20%
    latency_high: float = 1.0      # seconds (p95 external tracking; injected)
    adjustment_interval: float = 10.0
    decrease_factor: float = 0.7
    increase_factor: float = 1.15


class AdaptiveTokenBucket:
    def __init__(self, rate: float, capacity: int, config: Optional[AdaptiveBucketConfig] = None, time_fn: Callable[[], float] = time.time):
        cfg = config or AdaptiveBucketConfig(rate=rate, capacity=capacity)
        self.cfg = cfg
        self._time_fn = time_fn
        self._capacity = capacity
        self._tokens = capacity * 1.0
        self._rate = rate
        self._last_refill = time_fn()
        self._lock = asyncio.Lock()
        self._stats: list[tuple[float, bool, float]] = []  # (ts, success, latency)
        self._last_adjust = time_fn()

    def _refill(self):
        now = self._time_fn()
        delta = now - self._last_refill
        if delta > 0:
            self._tokens = min(self._capacity, self._tokens + delta * self._rate)
            self._last_refill = now

    async def acquire(self, tokens: float = 1.0, wait: bool = False, timeout: Optional[float] = None) -> bool:
        start = self._time_fn()
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
            if not wait:
                return False
            if timeout and self._time_fn() - start > timeout:
                return False
            await asyncio.sleep(0.01)

    def record_result(self, success: bool, latency: float):
        ts = self._time_fn()
        self._stats.append((ts, success, latency))
        cutoff = ts - 60.0
        while self._stats and self._stats[0][0] < cutoff:
            self._stats.pop(0)
        if ts - self._last_adjust >= self.cfg.adjustment_interval:
            self._adjust()

    def _adjust(self):
        if not self._stats:
            return
        errors = sum(1 for _, success, _ in self._stats if not success)
        total = len(self._stats)
        err_rate = errors / total if total else 0.0
        # Use p95 latency
        latencies = sorted(l for _, _, l in self._stats)
        p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
        # Adjust downward if poor health
        if err_rate > self.cfg.error_rate_high or p95 > self.cfg.latency_high:
            self._rate = max(self.cfg.min_rate, self._rate * self.cfg.decrease_factor)
        else:
            self._rate = min(self.cfg.max_rate, self._rate * self.cfg.increase_factor)
        self._last_adjust = self._time_fn()

    def metrics_snapshot(self) -> Dict[str, Any]:
        return {
            "rate": self._rate,
            "tokens": self._tokens,
            "capacity": self._capacity,
            "observations": len(self._stats),
        }

__all__ = [
    "BreakerState",
    "BreakerConfig",
    "AsyncCircuitBreaker",
    "AdaptiveBucketConfig",
    "AdaptiveTokenBucket",
]
