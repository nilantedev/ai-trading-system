#!/usr/bin/env python3
"""
WebSocket Router - WebSocket endpoints for real-time streaming
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from typing import List, Optional, Dict, Any
import json
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.websocket_manager import connection_manager, websocket_streamer
from datetime import datetime
from trading_common import get_logger

# Import APIException after other imports to avoid circular import
class APIException(Exception):
    def __init__(self, status_code, detail, error_code=None, context=None):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code
        self.context = context

logger = get_logger(__name__)

router = APIRouter()


@router.websocket("/ws/market-data")
async def websocket_market_data(
    websocket: WebSocket,
    symbols: Optional[str] = Query(None, description="Comma-separated symbols to subscribe to"),
    token: Optional[str] = Query(None, description="Authentication token")
):
    """WebSocket endpoint for real-time market data."""
    symbol_list = []
    
    try:
        # Parse symbols
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        # Authenticate user (optional for market data)
        user_info = None
        if token:
            try:
                # Would validate token here
                user_info = {"token": token, "authenticated": True}
            except Exception as e:
                logger.warning(f"WebSocket authentication failed: {e}")
        
        # Connect to WebSocket
        await connection_manager.connect(
            websocket, 
            "market_data", 
            symbols=symbol_list,
            user_info=user_info
        )
        
        # Handle messages
        while True:
            try:
                # Wait for message from client
                message = await websocket.receive_text()
                await connection_manager.handle_client_message(websocket, message)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in market data WebSocket: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Market data WebSocket error: {e}")
    finally:
        await connection_manager.disconnect(websocket)


@router.websocket("/ws/signals")
async def websocket_signals(
    websocket: WebSocket,
    symbols: Optional[str] = Query(None, description="Comma-separated symbols to subscribe to"),
    strategies: Optional[str] = Query(None, description="Comma-separated strategy names"),
    token: Optional[str] = Query(None, description="Authentication token")
):
    """WebSocket endpoint for real-time trading signals."""
    symbol_list = []
    strategy_list = []
    
    try:
        # Parse parameters
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        if strategies:
            strategy_list = [s.strip() for s in strategies.split(',')]
        
        # Authenticate user (optional for basic signals)
        user_info = None
        if token:
            try:
                # Would validate token here
                user_info = {
                    "token": token, 
                    "authenticated": True,
                    "subscribed_strategies": strategy_list
                }
            except Exception as e:
                logger.warning(f"WebSocket authentication failed: {e}")
        
        # Connect to WebSocket
        await connection_manager.connect(
            websocket, 
            "signals", 
            symbols=symbol_list,
            user_info=user_info
        )
        
        # Handle messages
        while True:
            try:
                message = await websocket.receive_text()
                await connection_manager.handle_client_message(websocket, message)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in signals WebSocket: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Signals WebSocket error: {e}")
    finally:
        await connection_manager.disconnect(websocket)


@router.websocket("/ws/orders")
async def websocket_orders(
    websocket: WebSocket,
    token: str = Query(..., description="Authentication token (required)")
):
    """WebSocket endpoint for real-time order updates (requires authentication)."""
    try:
        # Authenticate user (required for order updates)
        user_info = None
        try:
            # Would validate token here and get user info
            if token.startswith("demo_") or token == "admin123":
                user_info = {"token": token, "authenticated": True, "user_id": "demo_user"}
            else:
                await websocket.close(code=1008, reason="Invalid authentication token")
                return
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            await websocket.close(code=1008, reason="Authentication failed")
            return
        
        # Connect to WebSocket
        await connection_manager.connect(
            websocket, 
            "orders", 
            user_info=user_info
        )
        
        # Handle messages
        while True:
            try:
                message = await websocket.receive_text()
                await connection_manager.handle_client_message(websocket, message)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in orders WebSocket: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Orders WebSocket error: {e}")
    finally:
        await connection_manager.disconnect(websocket)


@router.websocket("/ws/portfolio")
async def websocket_portfolio(
    websocket: WebSocket,
    token: str = Query(..., description="Authentication token (required)")
):
    """WebSocket endpoint for real-time portfolio updates (requires authentication)."""
    try:
        # Authenticate user (required for portfolio updates)
        user_info = None
        try:
            # Would validate token here
            if token.startswith("demo_") or token == "admin123":
                user_info = {"token": token, "authenticated": True, "user_id": "demo_user"}
            else:
                await websocket.close(code=1008, reason="Invalid authentication token")
                return
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            await websocket.close(code=1008, reason="Authentication failed")
            return
        
        # Connect to WebSocket
        await connection_manager.connect(
            websocket, 
            "portfolio", 
            user_info=user_info
        )
        
        # Handle messages
        while True:
            try:
                message = await websocket.receive_text()
                await connection_manager.handle_client_message(websocket, message)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in portfolio WebSocket: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Portfolio WebSocket error: {e}")
    finally:
        await connection_manager.disconnect(websocket)


@router.websocket("/ws/alerts")
async def websocket_alerts(
    websocket: WebSocket,
    severity: Optional[str] = Query(None, description="Minimum alert severity (low, medium, high, critical)"),
    token: Optional[str] = Query(None, description="Authentication token")
):
    """WebSocket endpoint for real-time system alerts."""
    try:
        # Authenticate user (optional for public alerts)
        user_info = None
        if token:
            try:
                # Would validate token here
                user_info = {
                    "token": token, 
                    "authenticated": True,
                    "alert_severity_filter": severity
                }
            except Exception as e:
                logger.warning(f"WebSocket authentication failed: {e}")
        
        # Connect to WebSocket
        await connection_manager.connect(
            websocket, 
            "alerts", 
            user_info=user_info
        )
        
        # Handle messages
        while True:
            try:
                message = await websocket.receive_text()
                await connection_manager.handle_client_message(websocket, message)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in alerts WebSocket: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Alerts WebSocket error: {e}")
    finally:
        await connection_manager.disconnect(websocket)


@router.websocket("/ws/system")
async def websocket_system(
    websocket: WebSocket,
    token: str = Query(..., description="Authentication token (required)")
):
    """WebSocket endpoint for real-time system status updates (requires authentication)."""
    try:
        # Authenticate user (required for system updates)
        user_info = None
        try:
            # Would validate token here
            if token.startswith("demo_") or token == "admin123":
                user_info = {"token": token, "authenticated": True, "admin": True}
            else:
                await websocket.close(code=1008, reason="Invalid authentication token")
                return
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            await websocket.close(code=1008, reason="Authentication failed")
            return
        
        # Connect to WebSocket
        await connection_manager.connect(
            websocket, 
            "system", 
            user_info=user_info
        )
        
        # Handle messages
        while True:
            try:
                message = await websocket.receive_text()
                await connection_manager.handle_client_message(websocket, message)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in system WebSocket: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"System WebSocket error: {e}")
    finally:
        await connection_manager.disconnect(websocket)


@router.get("/ws/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics."""
    try:
        stats = connection_manager.get_connection_stats()
        
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "data": stats,
            "message": "WebSocket statistics retrieved"
        }
        
    except Exception as e:
        logger.error(f"Error getting WebSocket stats: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to get WebSocket statistics",
            error_code="WEBSOCKET_STATS_ERROR",
            context={"error": str(e)}
        )


# Test endpoint for WebSocket functionality
@router.get("/ws/test")
async def test_websocket_broadcast():
    """Test endpoint to trigger WebSocket broadcasts."""
    try:
        # Broadcast test message to all streams
        test_message = {
            "type": "test_broadcast",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "message": "This is a test broadcast",
                "test_id": "TEST_001"
            }
        }
        
        # Broadcast to all stream types
        for stream_type in connection_manager.connections.keys():
            await connection_manager.broadcast_to_stream(stream_type, test_message)
        
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Test broadcast sent to all WebSocket connections",
            "connections": {
                stream: len(conns) 
                for stream, conns in connection_manager.connections.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Error in WebSocket test broadcast: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to send test broadcast",
            error_code="WEBSOCKET_TEST_ERROR",
            context={"error": str(e)}
        )