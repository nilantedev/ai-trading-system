#!/usr/bin/env python3
"""
WebSocket Router - WebSocket endpoints for real-time streaming
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional, Dict, Any
import json
import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.websocket_manager import connection_manager, websocket_streamer
from api.websocket_auth import authenticate_websocket
from trading_common import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Helper to DRY message handling loop
async def _ws_message_loop(websocket: WebSocket):
    while True:
        try:
            message = await websocket.receive_text()
            await connection_manager.handle_client_message(websocket, message)
        except WebSocketDisconnect:
            break
        except Exception as e:
            logger.error("WebSocket message error", error=str(e))
            break

@router.websocket("/ws/market-data")
async def websocket_market_data(
    websocket: WebSocket,
    symbols: Optional[str] = Query(None, description="Comma-separated symbols to subscribe to"),
    token: Optional[str] = Query(None, description="Authentication token")
):
    symbol_list = [s.strip().upper() for s in symbols.split(',')] if symbols else []
    try:
        user_info = await authenticate_websocket(websocket, "market_data", token)
        # connect (user_info may be None if anonymous allowed)
        await connection_manager.connect(websocket, "market_data", symbols=symbol_list, user_info=user_info)
        await _ws_message_loop(websocket)
    except WebSocketDisconnect:
        pass
    finally:
        await connection_manager.disconnect(websocket)

@router.websocket("/ws/signals")
async def websocket_signals(
    websocket: WebSocket,
    symbols: Optional[str] = Query(None),
    strategies: Optional[str] = Query(None, description="Comma-separated strategy names"),
    token: Optional[str] = Query(None)
):
    symbol_list = [s.strip().upper() for s in symbols.split(',')] if symbols else []
    strategy_list = [s.strip() for s in strategies.split(',')] if strategies else []
    try:
        user_info = await authenticate_websocket(websocket, "signals", token)
        if user_info:
            user_info["subscribed_strategies"] = strategy_list
        await connection_manager.connect(websocket, "signals", symbols=symbol_list, user_info=user_info)
        await _ws_message_loop(websocket)
    except WebSocketDisconnect:
        pass
    finally:
        await connection_manager.disconnect(websocket)

@router.websocket("/ws/orders")
async def websocket_orders(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="Authentication token (required)")
):
    try:
        user_info = await authenticate_websocket(websocket, "orders", token)
        if not user_info:
            return  # authenticate_websocket already closed if required
        await connection_manager.connect(websocket, "orders", user_info=user_info)
        await _ws_message_loop(websocket)
    except WebSocketDisconnect:
        pass
    finally:
        await connection_manager.disconnect(websocket)

@router.websocket("/ws/portfolio")
async def websocket_portfolio(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="Authentication token (required)")
):
    try:
        user_info = await authenticate_websocket(websocket, "portfolio", token)
        if not user_info:
            return
        await connection_manager.connect(websocket, "portfolio", user_info=user_info)
        await _ws_message_loop(websocket)
    except WebSocketDisconnect:
        pass
    finally:
        await connection_manager.disconnect(websocket)

@router.websocket("/ws/alerts")
async def websocket_alerts(
    websocket: WebSocket,
    severity: Optional[str] = Query(None, description="Minimum alert severity (low, medium, high, critical)"),
    token: Optional[str] = Query(None)
):
    try:
        user_info = await authenticate_websocket(websocket, "alerts", token)
        if user_info:
            user_info["alert_severity_filter"] = severity
        await connection_manager.connect(websocket, "alerts", user_info=user_info)
        await _ws_message_loop(websocket)
    except WebSocketDisconnect:
        pass
    finally:
        await connection_manager.disconnect(websocket)

@router.websocket("/ws/system")
async def websocket_system(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="Authentication token (required)")
):
    try:
        user_info = await authenticate_websocket(websocket, "system", token)
        if not user_info:
            return
        await connection_manager.connect(websocket, "system", user_info=user_info)
        await _ws_message_loop(websocket)
    except WebSocketDisconnect:
        pass
    finally:
        await connection_manager.disconnect(websocket)

@router.get("/ws/stats")
async def get_websocket_stats():
    try:
        stats = connection_manager.get_connection_stats()
        return {"success": True, "timestamp": datetime.utcnow().isoformat(), "data": stats, "message": "WebSocket statistics retrieved"}
    except Exception as e:
        logger.error("Error getting WebSocket stats", error=str(e))
        return {"success": False, "error": str(e)}

@router.get("/ws/test")
async def test_websocket_broadcast():
    try:
        test_message = {"type": "test_broadcast", "timestamp": datetime.utcnow().isoformat(), "data": {"message": "This is a test broadcast"}}
        for stream_type in connection_manager.connections.keys():
            await connection_manager.broadcast_to_stream(stream_type, test_message)
        return {"success": True, "timestamp": datetime.utcnow().isoformat(), "message": "Test broadcast sent", "connections": {s: len(c) for s, c in connection_manager.connections.items()}}
    except Exception as e:
        logger.error("Error in WebSocket test broadcast", error=str(e))
        return {"success": False, "error": str(e)}