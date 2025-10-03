"""Base provider contracts for market data vendors.

Defines minimal protocol/ABC style interfaces for quote and historical daily
providers. Providers should:
  * Be side-effect free (no writes) – only fetch & normalize
  * Avoid embedding resilience (circuit breakers / retries / rate limiting) –
    those are applied at the service orchestration layer (MarketDataService)
  * Return normalized dict structures (NOT pydantic) for lightweight usage

Normalized field conventions (quotes & bars):
  symbol: str                     (root symbol without exchange suffix)
  timestamp: datetime (UTC)
  open/high/low/close: float
  volume: int (default 0 if unknown)
  data_source: str (short provider tag e.g. "alpaca", "polygon")
  timeframe: str (for bars; e.g. '1min', '5min', '1d', 'quote')

Daily bars may also include:
  adjusted_close: float | None

Capability Flags:
  Providers may expose a .capabilities set[str] (e.g. {'quotes','intraday','daily'}).

NOTE: Using typing.Protocol keeps this lightweight without enforcing inheritance.
"""
from __future__ import annotations

from typing import Protocol, List, Dict, Any, Optional
from datetime import datetime

class QuoteProvider(Protocol):
    """Protocol for quote-capable providers."""
    capabilities: set

    async def fetch_quote(self, symbol: str) -> Optional[Dict[str, Any]]:  # returns normalized dict or None
        ...

class DailyHistoryProvider(Protocol):
    """Protocol for daily history providers."""
    capabilities: set

    async def fetch_daily_bars(
        self,
        symbol: str,
        start_date: str,  # YYYY-MM-DD
        end_date: str,    # YYYY-MM-DD
        **kwargs
    ) -> List[Dict[str, Any]]:
        ...

class IntradayBarProvider(Protocol):
    """Protocol for intraday / timeframe-based bar providers."""
    capabilities: set

    async def fetch_timeframe_bars(
        self,
        symbol: str,
        timeframe: str,
        start_iso: str,
        end_iso: str,
        limit: int = 1000,
        **kwargs
    ) -> List[Dict[str, Any]]:
        ...

__all__ = [
    "QuoteProvider",
    "DailyHistoryProvider",
    "IntradayBarProvider",
]
