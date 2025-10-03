#!/usr/bin/env python3
"""Lightweight resilience shims for risk-monitor service.

Provides AsyncCircuitBreaker, BreakerState, and AdaptiveTokenBucket with a
minimal interface expected by main.py, without introducing new external
dependencies. Designed to be stable and metrics-friendly.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Callable, List, Optional, Tuple


class BreakerState(Enum):
    CLOSED = "closed"
    HALF_OPEN = "half_open"
    OPEN = "open"


class AsyncCircuitBreaker:
    """Simplified async-friendly circuit breaker with listener hooks.

    This implementation focuses on state transitions and listener callbacks
    used for metrics. It exposes on_result(success: bool) to update state based
    on recent outcomes and recovery timeout heuristics.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
    ):
        self.name = name
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_timeout = float(recovery_timeout)
        self.success_threshold = max(1, int(success_threshold))

        self._state: BreakerState = BreakerState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._last_failure_ts: Optional[float] = None
        self._listeners: List[Callable[[BreakerState, BreakerState], None]] = []

    @property
    def state(self) -> BreakerState:
        # AUTO transition from OPEN to HALF_OPEN when timeout elapsed
        if self._state is BreakerState.OPEN and self._last_failure_ts is not None:
            if (time.time() - self._last_failure_ts) >= self.recovery_timeout:
                self._transition(BreakerState.HALF_OPEN)
        return self._state

    def add_state_listener(self, cb: Callable[[BreakerState, BreakerState], None]):
        self._listeners.append(cb)

    def _transition(self, new_state: BreakerState):
        if new_state is self._state:
            return
        old = self._state
        self._state = new_state
        # Reset counters appropriately
        if new_state is BreakerState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state is BreakerState.HALF_OPEN:
            self._success_count = 0
        for cb in list(self._listeners):
            try:
                cb(old, new_state)
            except Exception:
                # Listener failures must not affect flow
                pass

    async def on_result(self, success: bool, _: Optional[float] = None):
        """Update breaker state given an operation result.

        Args:
            success: True if the recent operation succeeded; False otherwise.
            _: Optional latency placeholder (ignored here; kept for signature
               compatibility with callers that pass latency values).
        """
        if success:
            # Success logic
            if self.state is BreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._transition(BreakerState.CLOSED)
            elif self.state is BreakerState.OPEN:
                # If we observed a success while OPEN and timeout elapsed,
                # move towards closing; otherwise ignore.
                if self._last_failure_ts is None or (time.time() - self._last_failure_ts) >= self.recovery_timeout:
                    self._transition(BreakerState.HALF_OPEN)
            else:
                # CLOSED: keep counters clean
                self._failure_count = 0
        else:
            # Failure logic
            self._failure_count += 1
            self._last_failure_ts = time.time()
            if self.state is BreakerState.HALF_OPEN:
                # Any failure in HALF_OPEN re-opens the circuit
                self._transition(BreakerState.OPEN)
            elif self.state is BreakerState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._transition(BreakerState.OPEN)
            # If already OPEN, keep it OPEN and update last failure time


class AdaptiveTokenBucket:
    """Simple adaptive token bucket for rate control KPIs.

    This is intentionally minimal for metrics. It tracks a current rate and
    token count, and exposes a record_result() hook to nudge values based on
    success/failure. No concurrency control is attempted here.
    """

    def __init__(self, rate: float, capacity: float):
        self._base_rate = float(rate)
        self._rate = float(rate)
        self._capacity = float(capacity)
        self._tokens = float(capacity)

    def metrics_snapshot(self) -> dict:
        return {
            "rate": float(self._rate),
            "tokens": max(0.0, float(self._tokens)),
            "capacity": float(self._capacity),
        }

    def record_result(self, success: bool, latency_seconds: float):
        # Light adaptation: decrease tokens on failure, slowly refill on success
        if success:
            # gentle refill and rate normalization towards base
            self._tokens = min(self._capacity, self._tokens + max(1.0, self._capacity * 0.05))
            # drift rate 5% towards base
            self._rate = self._rate + (self._base_rate - self._rate) * 0.05
        else:
            self._tokens = max(0.0, self._tokens - max(1.0, self._capacity * 0.1))
            # reduce rate up to 20% on consecutive failures
            self._rate = max(self._base_rate * 0.2, self._rate * 0.8)
