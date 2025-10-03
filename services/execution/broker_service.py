"""Broker Service - Interface to trading brokers with paper Alpaca support.

This module selects a broker implementation at runtime based on environment:
  - If ALPACA_API_KEY_ID and ALPACA_SECRET_KEY are present and ALPACA_ENV=paper
    then an Alpaca Paper Trading client is used against the paper endpoint.
  - Otherwise, a no-op PaperBroker mock is used (simulated fills).

Only paper trading is supported; live trading is explicitly refused for safety.
"""

import logging
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class PaperBroker:
    """Fallback paper broker that simulates acceptance without execution."""

    connected: bool = False

    async def connect(self) -> bool:
        self.connected = True
        logger.info("Connected to mock PaperBroker (simulated)")
        return True

    async def disconnect(self) -> None:
        self.connected = False
        logger.info("Disconnected from mock PaperBroker")

    async def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("PaperBroker: submit_order %s", order)
        return {
            "order_id": order.get("client_order_id") or order.get("id") or "paper-mock",
            "status": "submitted",
            "message": "Paper order (mock) submitted successfully"
        }

    async def cancel_order(self, order_id: str) -> bool:
        logger.info("PaperBroker: cancel_order %s", order_id)
        return True

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        return {
            "order_id": order_id,
            "status": "filled",
            "filled_qty": 0,
            "message": "Paper mock status"
        }


class AlpacaPaperBroker:
    """Alpaca Paper Trading broker using REST API.

    Environment variables:
      - ALPACA_API_KEY_ID (required)
      - ALPACA_SECRET_KEY (required)
      - ALPACA_BASE_URL (default: https://paper-api.alpaca.markets)
      - ALPACA_ENV must be 'paper' (safety gate)
    """

    def __init__(self, api_key_id: str, secret_key: str, base_url: str):
        self.api_key_id = api_key_id
        self.secret_key = secret_key
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self.connected = False

    async def connect(self) -> bool:
        headers = {
            "APCA-API-KEY-ID": self.api_key_id,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json"
        }
        timeout = aiohttp.ClientTimeout(total=5)
        self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        # Probe account (paper) health
        try:
            async with self.session.get(f"{self.base_url}/v2/account") as resp:
                if resp.status == 200:
                    self.connected = True
                    logger.info("Connected to Alpaca Paper Trading API")
                    return True
                logger.warning("Alpaca connect probe failed: %s", resp.status)
        except Exception as e:  # noqa: BLE001
            logger.warning("Alpaca connect probe error: %s", e)
        return False

    async def disconnect(self) -> None:
        if self.session:
            try:
                await self.session.close()
            except Exception:  # noqa: BLE001
                pass
        self.session = None
        self.connected = False

    async def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        if not self.session:
            raise RuntimeError("Alpaca broker not connected")
        # Map generic order to Alpaca fields
        payload = {
            "symbol": order.get("symbol"),
            "qty": order.get("quantity") or order.get("qty"),
            "side": order.get("side"),  # 'buy' | 'sell'
            "type": order.get("order_type", "market"),  # 'market'|'limit'|'stop'|'stop_limit'
            "time_in_force": order.get("time_in_force", "day").upper(),  # DAY,GTC,IOC,FOK
            "client_order_id": order.get("client_order_id") or order.get("id")
        }
        if order.get("price") is not None:
            payload["limit_price"] = order.get("price")
        if order.get("stop_price") is not None:
            payload["stop_price"] = order.get("stop_price")
        # Force paper-only precaution: base URL must include 'paper'
        if "paper" not in self.base_url:
            raise RuntimeError("Alpaca live trading is disabled; use paper endpoint only")
        async with self.session.post(f"{self.base_url}/v2/orders", json=payload) as resp:
            data = await resp.json()
            if resp.status not in (200, 201):
                logger.error("Alpaca submit_order error %s: %s", resp.status, data)
                return {"status": "error", "code": resp.status, "data": data}
            return {
                "order_id": data.get("id"),
                "status": data.get("status"),
                "data": data
            }

    async def cancel_order(self, order_id: str) -> bool:
        if not self.session:
            return False
        async with self.session.delete(f"{self.base_url}/v2/orders/{order_id}") as resp:
            return resp.status in (200, 204)

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        if not self.session:
            return {"status": "error", "message": "not_connected"}
        async with self.session.get(f"{self.base_url}/v2/orders/{order_id}") as resp:
            data = await resp.json()
            return data | {"code": resp.status}


def get_broker_service():
    """Select and instantiate an appropriate broker service for execution."""
    env = (os.getenv("ALPACA_ENV", "paper").lower() or "paper").strip()
    key = os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("ALPACA_SECRET_KEY")
    base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    # Safety: only allow when explicit paper env and keys present
    if key and secret and env == "paper" and "paper" in base:
        logger.info("Using AlpacaPaperBroker for order execution")
        return AlpacaPaperBroker(key, secret, base)
    logger.info("Using fallback PaperBroker (no external trades)")
    return PaperBroker()