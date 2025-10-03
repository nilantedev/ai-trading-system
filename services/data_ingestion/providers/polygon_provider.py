"""Polygon Provider Abstraction (Phase 1)

Scope:
  * Real-time last trade/quote simplified endpoint
  * Normalization only; no resilience here (handled by service layer)

Future Enhancements:
  * Aggregates (daily/intraday) with pagination
  * Multi-symbol batching (if supported)
  * Corporate actions / splits integration
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolygonConfig:
    api_key: str
    base_url: str = "https://api.polygon.io"


class PolygonProvider:
    capabilities = {"quotes", "option_aggs"}  # future: add 'intraday','daily'

    def __init__(self, cfg: PolygonConfig, session):  # session: aiohttp.ClientSession
        self.cfg = cfg
        self.session = session

    async def fetch_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch latest trade (used as quote proxy) and normalize.

        Endpoint chosen: /v2/last/trade/{symbol}
        This is a simplified representation (Polygon also offers /v3/trades, /v2/last/nbbo/{symbol}).
        """
        if not self.cfg.api_key:
            return None
        url = f"{self.cfg.base_url}/v2/last/trade/{symbol.upper()}"
        # Polygon expects 'apiKey' (capital K). For robustness, also send Authorization header.
        params = {"apiKey": self.cfg.api_key}
        try:
            headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
            async with self.session.get(url, params=params, headers=headers, timeout=10) as resp:
                if resp.status == 200:
                    payload = await resp.json()
                    trade = payload.get("results") or {}
                    if not trade:
                        return None
                    # Polygon timestamps in nanoseconds epoch; convert if present
                    ts_ns = trade.get("t")
                    if isinstance(ts_ns, (int, float)) and ts_ns > 0:
                        ts = datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc)
                    else:
                        ts = datetime.now(tz=timezone.utc)
                    price = float(trade.get("p", 0.0))
                    size = int(trade.get("s", 0))
                    return {
                        "symbol": symbol.upper(),
                        "timestamp": ts,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": size,
                        "timeframe": "quote",
                        "data_source": "polygon"
                    }
                elif resp.status == 429:
                    raise RuntimeError("Polygon rate limit (429)")
                elif resp.status in (401, 403):
                    logger.warning("Polygon auth issue (%s)", resp.status)
                elif resp.status >= 500:
                    raise RuntimeError(f"Polygon server error {resp.status}")
        except Exception as e:  # noqa: BLE001
            logger.debug("Polygon fetch_quote error %s", e)
        return None

    async def fetch_option_aggregates(
        self,
        option_ticker: str,
        start_date: datetime,
        end_date: datetime,
        *,
        adjusted: bool = True,
        limit: int = 50000,
    ) -> Optional[list[dict]]:
        """Fetch daily aggregates for a specific option contract ticker using Polygon v2.

        option_ticker: e.g., 'O:SPY250919C00450000'
        Returns a list of dicts with ts (datetime), o,h,l,c,v
        """
        if not self.cfg.api_key:
            return None
        url = (
            f"{self.cfg.base_url}/v2/aggs/ticker/{option_ticker}/range/1/day/"
            f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        )
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": limit,
            "apiKey": self.cfg.api_key,
        }
        try:
            async with self.session.get(url, params=params, timeout=20) as resp:
                if resp.status == 401 or resp.status == 403:
                    logger.warning("Polygon auth issue (%s) option_aggs", resp.status)
                    return None
                if resp.status == 429:
                    logger.debug("Polygon option aggs rate limit 429 for %s", option_ticker)
                    return None
                if resp.status >= 500:
                    logger.debug("Polygon option aggs server error %s for %s", resp.status, option_ticker)
                    return None
                if resp.status != 200:
                    logger.debug("Polygon option aggs HTTP %s for %s", resp.status, option_ticker)
                    return None
                data = await resp.json()
        except Exception as e:  # noqa: BLE001
            logger.debug("Polygon option aggs error for %s: %s", option_ticker, e)
            return None
        results = data.get("results") or []
        if not results:
            logger.debug(
                "polygon.option.aggs.empty ticker=%s window=%s->%s url=%s",
                option_ticker, start_date.date(), end_date.date(), url
            )
        out: list[dict] = []
        for r in results:
            try:
                ts = datetime.utcfromtimestamp(r.get("t", 0) / 1000)
                if ts < start_date or ts > end_date:
                    continue
                out.append({
                    "timestamp": ts,
                    "open": float(r.get("o", 0.0)),
                    "high": float(r.get("h", 0.0)),
                    "low": float(r.get("l", 0.0)),
                    "close": float(r.get("c", 0.0)),
                    "volume": int(r.get("v", 0)),
                })
            except Exception:
                continue
        return out

    async def list_option_contracts(
        self,
        underlying: str,
        start_expiry: Optional[datetime] = None,
        end_expiry: Optional[datetime] = None,
        *,
        expired: bool = True,
        limit: int = 1000,
    ) -> Optional[List[Dict[str, Any]]]:
        """List option contracts for an underlying via Polygon v3 reference API.

        Returns a list of dicts with keys: ticker, expiration_date, strike_price, contract_type (call/put)
        """
        if not self.cfg.api_key:
            return None
        base = f"{self.cfg.base_url}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying.upper(),
            "expired": str(expired).lower(),
            "limit": min(max(limit, 1), 1000),
            "sort": "expiration_date",
            "order": "asc",
            "apiKey": self.cfg.api_key,
        }
        if start_expiry:
            params["expiration_date.gte"] = start_expiry.strftime('%Y-%m-%d')
        if end_expiry:
            params["expiration_date.lte"] = end_expiry.strftime('%Y-%m-%d')
        out: List[Dict[str, Any]] = []
        url = base
        # Simple pagination by next_url if present
        try:
            while True:
                headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
                async with self.session.get(url, params=params, headers=headers, timeout=20) as resp:
                    if resp.status != 200:
                        logger.debug("Polygon list_option_contracts HTTP %s for %s", resp.status, underlying)
                        break
                    data = await resp.json()
                results = data.get("results") or []
                for c in results:
                    try:
                        out.append({
                            "ticker": c.get("ticker"),
                            "expiration_date": c.get("expiration_date"),
                            "strike_price": float(c.get("strike_price", 0.0)),
                            "contract_type": c.get("contract_type"),  # call/put
                        })
                    except Exception:
                        continue
                next_url = data.get("next_url")
                if next_url:
                    # next_url already includes apiKey
                    url = next_url
                    params = None
                else:
                    break
        except Exception as e:  # noqa: BLE001
            logger.debug("Polygon list_option_contracts error for %s: %s", underlying, e)
            return None
        return out

__all__ = ["PolygonProvider", "PolygonConfig"]
