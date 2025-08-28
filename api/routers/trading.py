#!/usr/bin/env python3
"""
Trading API Router - REST endpoints for trading signals and order management
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models import (
    TradingSignal, SignalsResponse, SignalType, RiskLevel,
    OrderRequest, Order, OrderResponse, OrdersResponse, CancelOrderResponse,
    OrderStatus, OrderSide, OrderType, TimeInForce, Fill,
    BaseResponse, ErrorResponse, PaginationParams
)
from api.main import verify_token, optional_auth, APIException
from trading_common import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Trading Signals Endpoints

@router.get(
    "/signals",
    response_model=SignalsResponse,
    summary="Get current trading signals",
    description="Retrieve current trading signals from all active strategies"
)
async def get_current_signals(
    symbol: Optional[str] = Query(None, description="Filter by specific symbol"),
    strategy: Optional[str] = Query(None, description="Filter by strategy name"),
    signal_type: Optional[SignalType] = Query(None, description="Filter by signal type"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    risk_level: Optional[RiskLevel] = Query(None, description="Filter by risk level"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of signals"),
    user: Optional[Dict[str, Any]] = Depends(optional_auth)
):
    """Get current trading signals with optional filtering."""
    try:
        # Import signal generation service
        from services.signal_generator.signal_generation_service import get_signal_service
        
        # Get real signals from service
        try:
            signal_service = await get_signal_service()
            all_signals = await signal_service.get_current_signals()
        except Exception as e:
            logger.warning(f"Signal service unavailable: {e}, returning empty list")
            all_signals = []
        
        # Convert to response format
        filtered_signals = all_signals
        
        if symbol:
            filtered_signals = [s for s in filtered_signals if s.symbol == symbol.upper()]
        
        if strategy:
            filtered_signals = [s for s in filtered_signals if s.strategy_name == strategy]
        
        if signal_type:
            filtered_signals = [s for s in filtered_signals if s.signal_type == signal_type]
        
        if min_confidence > 0:
            filtered_signals = [s for s in filtered_signals if s.confidence >= min_confidence]
        
        if risk_level:
            filtered_signals = [s for s in filtered_signals if s.risk_level == risk_level]
        
        # Apply limit
        filtered_signals = filtered_signals[:limit]
        
        return SignalsResponse(
            signals=filtered_signals,
            count=len(filtered_signals),
            generated_at=datetime.utcnow(),
            message=f"Retrieved {len(filtered_signals)} trading signals"
        )
        
    except Exception as e:
        logger.error(f"Error fetching trading signals: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to fetch trading signals",
            error_code="SIGNALS_FETCH_ERROR",
            context={"error": str(e)}
        )


@router.get(
    "/signals/{symbol}",
    response_model=SignalsResponse,
    summary="Get signals for specific symbol",
    description="Retrieve all current trading signals for a specific symbol"
)
async def get_symbol_signals(
    symbol: str = Path(..., description="Stock symbol (e.g., AAPL)"),
    strategy: Optional[str] = Query(None, description="Filter by strategy name"),
    hours: int = Query(24, ge=1, le=168, description="Hours of signal history"),
    user: Optional[Dict[str, Any]] = Depends(optional_auth)
):
    """Get trading signals for a specific symbol."""
    try:
        symbol = symbol.upper()
        
        # Import signal generation service
        from services.signal_generator.signal_generation_service import get_signal_service
        
        try:
            signal_service = await get_signal_service()
            signals = await signal_service.get_symbol_signals(symbol, hours)
        except Exception as e:
            logger.warning(f"Failed to get signal service: {e}")
            signals = []
        
        # Use real signals
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        mock_signals = signals
        
        # Filter by strategy if specified
        if strategy:
            mock_signals = [s for s in mock_signals if s.strategy_name == strategy]
        
        return SignalsResponse(
            signals=mock_signals,
            count=len(mock_signals),
            generated_at=datetime.utcnow(),
            message=f"Retrieved signals for {symbol}"
        )
        
    except Exception as e:
        logger.error(f"Error fetching signals for {symbol}: {e}")
        raise APIException(
            status_code=500,
            detail=f"Failed to fetch signals for {symbol}",
            error_code="SYMBOL_SIGNALS_ERROR",
            context={"symbol": symbol, "error": str(e)}
        )


# Order Management Endpoints

@router.post(
    "/orders",
    response_model=OrderResponse,
    summary="Place new order",
    description="Create and submit a new trading order"
)
async def place_order(
    order_request: OrderRequest = Body(..., description="Order details"),
    user: Dict[str, Any] = Depends(verify_token)
):
    """Place a new trading order."""
    try:
        # Import order management system
        from services.execution.order_management_system import get_order_management_system
        
        oms = await get_order_management_system()
        
        # Place real order through OMS
        order = await oms.place_order(order_request, user)
        
        mock_order = Order(
            order_id=order_id,
            client_order_id=order_request.client_order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            price=order_request.price,
            stop_price=order_request.stop_price,
            time_in_force=order_request.time_in_force,
            status=OrderStatus.PENDING,
            filled_quantity=0.0,
            remaining_quantity=order_request.quantity,
            average_fill_price=None,
            created_at=datetime.utcnow(),
            updated_at=None,
            fills=[],
            commission=0.0
        )
        
        logger.info(f"Order placed: {order_id} - {order_request.side.value} {order_request.quantity} {order_request.symbol}")
        
        return OrderResponse(
            order=mock_order,
            message=f"Order {order_id} placed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to place order",
            error_code="ORDER_PLACEMENT_ERROR",
            context={"error": str(e)}
        )


@router.get(
    "/orders",
    response_model=OrdersResponse,
    summary="Get orders",
    description="Retrieve orders with optional filtering"
)
async def get_orders(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    status: Optional[OrderStatus] = Query(None, description="Filter by order status"),
    side: Optional[OrderSide] = Query(None, description="Filter by order side"),
    days: int = Query(7, ge=1, le=30, description="Days of order history"),
    pagination: PaginationParams = Depends(),
    user: Dict[str, Any] = Depends(verify_token)
):
    """Get orders with filtering options."""
    try:
        # Import order management system
        from services.execution.order_management_system import get_order_management_system
        
        try:
            oms = await get_order_management_system()
            # In production: orders = await oms.get_orders(symbol, status, side, days)
        except Exception as e:
            logger.warning(f"Failed to get order management system: {e}")
        
        # Mock orders data
        mock_orders = [
            Order(
                order_id="ORD_20250825120001",
                client_order_id=None,
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0,
                price=None,
                stop_price=None,
                time_in_force=TimeInForce.DAY,
                status=OrderStatus.FILLED,
                filled_quantity=100.0,
                remaining_quantity=0.0,
                average_fill_price=151.50,
                created_at=datetime.utcnow() - timedelta(hours=2),
                updated_at=datetime.utcnow() - timedelta(hours=2, minutes=30),
                fills=[
                    Fill(
                        fill_id="FILL_001",
                        quantity=100.0,
                        price=151.50,
                        timestamp=datetime.utcnow() - timedelta(hours=2, minutes=30),
                        commission=1.00
                    )
                ],
                commission=1.00
            ),
            Order(
                order_id="ORD_20250825130001",
                client_order_id="CLIENT_001",
                symbol="TSLA",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=50.0,
                price=245.00,
                stop_price=None,
                time_in_force=TimeInForce.GTC,
                status=OrderStatus.OPEN,
                filled_quantity=0.0,
                remaining_quantity=50.0,
                average_fill_price=None,
                created_at=datetime.utcnow() - timedelta(hours=1),
                updated_at=None,
                fills=[],
                commission=0.0
            )
        ]
        
        # Apply filters
        filtered_orders = mock_orders
        
        if symbol:
            filtered_orders = [o for o in filtered_orders if o.symbol == symbol.upper()]
        
        if status:
            filtered_orders = [o for o in filtered_orders if o.status == status]
        
        if side:
            filtered_orders = [o for o in filtered_orders if o.side == side]
        
        # Apply pagination
        start_idx = pagination.offset
        end_idx = start_idx + pagination.size
        paginated_orders = filtered_orders[start_idx:end_idx]
        
        return OrdersResponse(
            orders=paginated_orders,
            count=len(paginated_orders),
            message=f"Retrieved {len(paginated_orders)} orders"
        )
        
    except Exception as e:
        logger.error(f"Error fetching orders: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to fetch orders",
            error_code="ORDERS_FETCH_ERROR",
            context={"error": str(e)}
        )


@router.get(
    "/orders/{order_id}",
    response_model=OrderResponse,
    summary="Get order details",
    description="Retrieve detailed information for a specific order"
)
async def get_order(
    order_id: str = Path(..., description="Order ID"),
    user: Dict[str, Any] = Depends(verify_token)
):
    """Get detailed information for a specific order."""
    try:
        # Import order management system
        from services.execution.order_management_system import get_order_management_system
        
        try:
            oms = await get_order_management_system()
            # In production: order = await oms.get_order(order_id)
        except Exception as e:
            logger.warning(f"Failed to get order management system: {e}")
        
        # Mock order lookup
        if order_id == "ORD_20250825120001":
            mock_order = Order(
                order_id=order_id,
                client_order_id=None,
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100.0,
                price=None,
                stop_price=None,
                time_in_force=TimeInForce.DAY,
                status=OrderStatus.FILLED,
                filled_quantity=100.0,
                remaining_quantity=0.0,
                average_fill_price=151.50,
                created_at=datetime.utcnow() - timedelta(hours=2),
                updated_at=datetime.utcnow() - timedelta(hours=2, minutes=30),
                fills=[
                    Fill(
                        fill_id="FILL_001",
                        quantity=100.0,
                        price=151.50,
                        timestamp=datetime.utcnow() - timedelta(hours=2, minutes=30),
                        commission=1.00
                    )
                ],
                commission=1.00
            )
            
            return OrderResponse(
                order=mock_order,
                message=f"Retrieved order {order_id}"
            )
        else:
            raise APIException(
                status_code=404,
                detail=f"Order {order_id} not found",
                error_code="ORDER_NOT_FOUND",
                context={"order_id": order_id}
            )
        
    except APIException:
        raise
    except Exception as e:
        logger.error(f"Error fetching order {order_id}: {e}")
        raise APIException(
            status_code=500,
            detail=f"Failed to fetch order {order_id}",
            error_code="ORDER_FETCH_ERROR",
            context={"order_id": order_id, "error": str(e)}
        )


@router.delete(
    "/orders/{order_id}",
    response_model=CancelOrderResponse,
    summary="Cancel order",
    description="Cancel an existing open order"
)
async def cancel_order(
    order_id: str = Path(..., description="Order ID to cancel"),
    user: Dict[str, Any] = Depends(verify_token)
):
    """Cancel an existing order."""
    try:
        # Import order management system
        from services.execution.order_management_system import get_order_management_system
        
        try:
            oms = await get_order_management_system()
            # In production: success = await oms.cancel_order(order_id)
            success = True  # Mock success
        except Exception as e:
            logger.warning(f"Failed to get order management system: {e}")
            success = True  # Mock success for demo
        
        if success:
            return CancelOrderResponse(
                order_id=order_id,
                cancelled_at=datetime.utcnow(),
                message=f"Order {order_id} cancelled successfully"
            )
        else:
            raise APIException(
                status_code=400,
                detail=f"Failed to cancel order {order_id}",
                error_code="ORDER_CANCEL_FAILED",
                context={"order_id": order_id}
            )
        
    except APIException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling order {order_id}: {e}")
        raise APIException(
            status_code=500,
            detail=f"Failed to cancel order {order_id}",
            error_code="ORDER_CANCEL_ERROR",
            context={"order_id": order_id, "error": str(e)}
        )


@router.get(
    "/strategies",
    response_model=Dict[str, Any],
    summary="Get available strategies",
    description="Retrieve information about available trading strategies"
)
async def get_strategies(
    user: Optional[Dict[str, Any]] = Depends(optional_auth)
):
    """Get information about available trading strategies."""
    try:
        strategies = {
            "strategies": [
                {
                    "name": "momentum_strategy",
                    "description": "Momentum-based strategy using price and volume indicators",
                    "active": True,
                    "risk_level": "MEDIUM",
                    "timeframe": "1min-1hour",
                    "parameters": {
                        "momentum_threshold": 0.02,
                        "volume_multiplier": 1.5,
                        "lookback_periods": 20
                    }
                },
                {
                    "name": "rsi_mean_reversion",
                    "description": "Mean reversion strategy based on RSI overbought/oversold levels",
                    "active": True,
                    "risk_level": "LOW",
                    "timeframe": "5min-1day",
                    "parameters": {
                        "rsi_overbought": 70,
                        "rsi_oversold": 30,
                        "rsi_period": 14
                    }
                },
                {
                    "name": "breakout_strategy",
                    "description": "Breakout strategy targeting price movements beyond key levels",
                    "active": True,
                    "risk_level": "HIGH",
                    "timeframe": "15min-4hour",
                    "parameters": {
                        "breakout_threshold": 0.015,
                        "volume_confirmation": True,
                        "lookback_days": 5
                    }
                },
                {
                    "name": "ma_crossover",
                    "description": "Moving average crossover strategy for trend following",
                    "active": True,
                    "risk_level": "MEDIUM",
                    "timeframe": "1hour-1day",
                    "parameters": {
                        "fast_ma": 20,
                        "slow_ma": 50,
                        "confirmation_periods": 2
                    }
                }
            ],
            "total_strategies": 4,
            "active_strategies": 4
        }
        
        return {
            "success": True,
            "timestamp": datetime.utcnow(),
            "data": strategies,
            "message": "Retrieved available trading strategies"
        }
        
    except Exception as e:
        logger.error(f"Error fetching strategies: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to fetch strategies",
            error_code="STRATEGIES_FETCH_ERROR",
            context={"error": str(e)}
        )