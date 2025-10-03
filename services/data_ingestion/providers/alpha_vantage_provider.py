"""Alpha Vantage Provider Abstraction (Phase 1)

Scope:
  * GLOBAL_QUOTE endpoint for latest quote snapshot
  * Normalization only; resilience externalized

Notes:
  * Alpha Vantage has strict rate limits; service layer must enforce.
  * Daily historical (TIME_SERIES_DAILY / ADJUSTED) intentionally deferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlphaVantageConfig:
    api_key: str
    base_url: str = "https://www.alphavantage.co/query"


class AlphaVantageProvider:
    capabilities = {"quotes"}

    def __init__(self, cfg: AlphaVantageConfig, session):  # session: aiohttp.ClientSession
        self.cfg = cfg
        self.session = session

    async def fetch_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.cfg.api_key:
            return None
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol.upper(),
            "apikey": self.cfg.api_key,
        }
        try:
            async with self.session.get(self.cfg.base_url, params=params, timeout=15) as resp:
                if resp.status == 200:
                    payload = await resp.json()
                    if 'Error Message' in payload:
                        logger.warning("Alpha Vantage error: %s", payload['Error Message'])
                        return None
                    if 'Note' in payload:
                        # Rate limit / throttling note
                        raise RuntimeError("Alpha Vantage rate limited")
                    quote = payload.get('Global Quote') or {}
                    if not quote:
                        return None
                    # AV does not provide a precise timestamp; use now UTC
                    now = datetime.now(tz=timezone.utc)
                    def _f(key: str, default: float = 0.0) -> float:
                        try:
                            return float(quote.get(key, default))
                        except Exception:  # noqa: BLE001
                            return default
                    return {
                        "symbol": symbol.upper(),
                        "timestamp": now,
                        "open": _f('02. open'),
                        "high": _f('03. high'),
                        "low": _f('04. low'),
                        "close": _f('05. price'),
                        "volume": int(float(quote.get('06. volume', 0) or 0)),
                        "timeframe": "quote",
                        "data_source": "alpha_vantage"
                    }
                elif resp.status == 429:
                    raise RuntimeError("Alpha Vantage rate limit (429)")
                elif resp.status >= 500:
                    raise RuntimeError(f"Alpha Vantage server error {resp.status}")
        except Exception as e:  # noqa: BLE001
            logger.debug("Alpha Vantage fetch_quote error %s", e)
        return None

__all__ = ["AlphaVantageProvider", "AlphaVantageConfig"]
