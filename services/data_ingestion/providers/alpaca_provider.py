"""Alpaca Provider Abstraction

Scope (Phase 1 extraction):
  * Real-time quote fetch (latest trade/quote consolidation simplified)
  * Intraday bars (timeframe) limited subset
  * Normalization only – resilience is handled by MarketDataService

Future Enhancements:
  * Corporate actions & dividend endpoints
  * Pagination for large intraday windows
  * Multi-symbol batching (if API supports efficiently)

Environment / Config dependency is passed in (no direct env reads here) to keep
provider deterministic & testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlpacaConfig:
    api_key: str
    secret_key: str
    data_url: str = "https://data.alpaca.markets"


class AlpacaProvider:
    capabilities = {"quotes", "intraday"}

    def __init__(self, cfg: AlpacaConfig, session):  # session: aiohttp.ClientSession
        self.cfg = cfg
        self.session = session

    # ---------------- Quotes ---------------- #
    async def fetch_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.cfg.api_key or not self.cfg.secret_key:
            return None
        url = f"{self.cfg.data_url}/v2/stocks/{symbol}/quotes/latest"
        headers = {
            'APCA-API-KEY-ID': self.cfg.api_key,
            'APCA-API-SECRET-KEY': self.cfg.secret_key,
        }
        try:
            async with self.session.get(url, headers=headers, timeout=10) as resp:
                if resp.status == 200:
                    payload = await resp.json()
                    q = payload.get('quote') or {}
                    if not q:
                        return None
                    # Normalization: Use 't' (ISO) if present else current time
                    ts_raw = q.get('t')
                    try:
                        ts = datetime.fromisoformat(ts_raw.replace('Z', '+00:00')) if ts_raw else datetime.utcnow()
                    except Exception:
                        ts = datetime.utcnow()
                    return {
                        'symbol': symbol.upper(),
                        'timestamp': ts,
                        'open': float(q.get('o', 0.0)),
                        'high': float(q.get('h', 0.0)),
                        'low': float(q.get('l', 0.0)),
                        'close': float(q.get('c', 0.0)),
                        'volume': int(q.get('v', 0)),
                        'timeframe': 'quote',
                        'data_source': 'alpaca'
                    }
                elif resp.status in (401, 403):
                    logger.warning("Alpaca auth issue (%s)", resp.status)
                elif resp.status == 429:
                    raise RuntimeError("Alpaca rate limit (429)")
                elif resp.status >= 500:
                    raise RuntimeError(f"Alpaca server error {resp.status}")
        except Exception as e:  # noqa: BLE001
            logger.debug("Alpaca fetch_quote error %s", e)
        return None

    # ---------------- Intraday Timeframe Bars ---------------- #
    async def fetch_timeframe_bars(
        self,
        symbol: str,
        timeframe: str,
        start_iso: str,
        end_iso: str,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        if not self.cfg.api_key or not self.cfg.secret_key:
            return []
        url = f"{self.cfg.data_url}/v2/stocks/{symbol}/bars"
        headers = {
            'APCA-API-KEY-ID': self.cfg.api_key,
            'APCA-API-SECRET-KEY': self.cfg.secret_key,
        }
        params = {
            'timeframe': timeframe,
            'start': start_iso,
            'end': end_iso,
            'limit': limit,
            'adjustment': 'raw'
        }
        try:
            async with self.session.get(url, headers=headers, params=params, timeout=15) as resp:
                if resp.status == 200:
                    payload = await resp.json()
                    bars = payload.get('bars', [])
                    out: List[Dict[str, Any]] = []
                    for b in bars:
                        try:
                            ts = datetime.fromisoformat(b['t'].replace('Z', '+00:00'))
                            out.append({
                                'symbol': symbol.upper(),
                                'timestamp': ts,
                                'open': float(b.get('o', 0.0)),
                                'high': float(b.get('h', 0.0)),
                                'low': float(b.get('l', 0.0)),
                                'close': float(b.get('c', 0.0)),
                                'volume': int(b.get('v', 0)),
                                'timeframe': timeframe,
                                'data_source': 'alpaca'
                            })
                        except Exception:
                            continue
                    return out
                elif resp.status == 429:
                    raise RuntimeError("Alpaca rate limit (429)")
        except Exception as e:  # noqa: BLE001
            logger.debug("Alpaca fetch_timeframe_bars error %s", e)
        return []

__all__ = ["AlpacaProvider", "AlpacaConfig"]
