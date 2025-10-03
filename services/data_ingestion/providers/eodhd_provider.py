"""EODHD Provider Abstraction

Lightweight, dependency-minimal adapter to the EODHD REST API for historical
daily (EOD) price data (20+ year coverage) and (future) options metadata.

Design Principles:
  * Pure async HTTP using existing aiohttp session (no extra SDK dependency)
  * Resilience (retry, rate limiting) delegated to caller (MarketDataService)
  * Normalization returns simple list[dict] with strictly typed fields
  * NO side effects (no writes) – ingestion/ persistence handled elsewhere
  * Safe in production: guarded by presence of EODHD_API_KEY

Supported Endpoint (Phase 1):
  GET https://eodhd.com/api/eod/{SYMBOL}.{EXCHANGE}?from=YYYY-MM-DD&to=YYYY-MM-DD&api_token=TOKEN&fmt=json

Notes:
  * If exchange suffix not provided by caller (e.g. "AAPL"), we attempt a
    minimal heuristic: default to US ("US") variant first (AAPL.US) then try
    raw symbol fallback.
  * Caller may pre-resolve correct exchange mapping from reference data.
  * EODHD sometimes returns adjusted_close; retain both close & adjusted if present.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EODHDConfig:
    api_key: str
    base_url: str = "https://eodhd.com/api"
    default_exchange: str = "US"  # Heuristic default; can be overridden per call


class EODHDProvider:
    """Async EODHD API wrapper (Phase 1: daily bars)."""

    def __init__(self, cfg: EODHDConfig, session):  # session: aiohttp.ClientSession
        self.cfg = cfg
        self.session = session

    async def fetch_daily_bars(
        self,
        symbol: str,
        start_date: str,  # YYYY-MM-DD
        end_date: str,    # YYYY-MM-DD
        *,
        exchange: Optional[str] = None,
        timeout: float = 20.0,
    ) -> List[Dict[str, Any]]:
        """Fetch daily (EOD) bars for symbol within inclusive date range.

        Returns list of normalized dicts:
          {
            'symbol': str,
            'timestamp': datetime (UTC, date at 00:00),
            'open': float, 'high': float, 'low': float, 'close': float,
            'adjusted_close': float | None,
            'volume': int,
            'data_source': 'eodhd'
          }
        """
        if not self.cfg.api_key:
            logger.debug("EODHDProvider called without API key configured; returning empty list")
            return []

        candidates = []
        # If symbol already contains a dot, assume user provided exchange suffix
        if "." in symbol:
            candidates.append(symbol)
        else:
            exch = exchange or self.cfg.default_exchange
            candidates.append(f"{symbol}.{exch}")
            candidates.append(symbol)  # raw fallback

        params_base = {
            "from": start_date,
            "to": end_date,
            "api_token": self.cfg.api_key,
            "fmt": "json",
        }

        last_error: Optional[Exception] = None
        for candidate in candidates:
            url = f"{self.cfg.base_url}/eod/{candidate}"
            try:
                async with self.session.get(url, params=params_base, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if not isinstance(data, list):  # Might be error structure
                            logger.warning("EODHD non-list response for %s: %s", candidate, data)
                            continue
                        return self._normalize_daily(candidate, data)
                    elif resp.status in (404, 400):
                        # Try next candidate
                        logger.debug("EODHD candidate %s not found (%s)", candidate, resp.status)
                        continue
                    elif resp.status == 429:
                        raise RuntimeError("EODHD rate limit (429)")
                    else:
                        text = await resp.text()
                        logger.warning("EODHD unexpected %s for %s: %s", resp.status, candidate, text[:180])
            except Exception as e:  # noqa: BLE001
                last_error = e
                await asyncio.sleep(0.5)  # small backoff between candidates
                continue

        if last_error:
            logger.debug("EODHD all candidates failed for %s: %s", symbol, last_error)
        return []

    # ------------------------- Internal Helpers ------------------------- #
    def _normalize_daily(self, resolved_symbol: str, raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in raw_rows:
            try:
                # EODHD returns date as 'YYYY-MM-DD'
                date_str = row.get("date") or row.get("Date")
                if not date_str:
                    continue
                ts = datetime.strptime(date_str, "%Y-%m-%d")
                out.append({
                    "symbol": resolved_symbol.split(".")[0],  # store base symbol
                    "timestamp": ts,
                    "open": float(row.get("open") or row.get("Open") or 0.0),
                    "high": float(row.get("high") or row.get("High") or 0.0),
                    "low": float(row.get("low") or row.get("Low") or 0.0),
                    "close": float(row.get("close") or row.get("Close") or 0.0),
                    "adjusted_close": float(row.get("adjusted_close") or row.get("Adj_Close") or row.get("adjClose") or 0.0) if row.get("adjusted_close") or row.get("Adj_Close") or row.get("adjClose") else None,
                    "volume": int(row.get("volume") or row.get("Volume") or 0),
                    "data_source": "eodhd",
                })
            except Exception:  # pragma: no cover - individual row tolerance
                continue
        return out

__all__ = ["EODHDProvider", "EODHDConfig"]
