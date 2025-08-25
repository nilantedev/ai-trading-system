#!/usr/bin/env python3
"""
Portfolio API Router - REST endpoints for portfolio and position management
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.models import (
    PortfolioResponse, Position, Account, PerformanceMetrics, PerformanceResponse,
    BaseResponse, ErrorResponse, PaginationParams
)
from api.main import verify_token, optional_auth, APIException
from trading_common import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/portfolio",
    response_model=PortfolioResponse,
    summary="Get portfolio overview",
    description="Retrieve complete portfolio overview including account info and positions"
)
async def get_portfolio(
    user: Dict[str, Any] = Depends(verify_token)
):
    """Get complete portfolio overview."""
    try:
        # Import broker service for account data
        from services.broker_integration.broker_service import get_broker_service
        
        try:
            broker_service = await get_broker_service()
            # In production: account = await broker_service.get_account()
            # In production: positions = await broker_service.get_positions()
        except Exception as e:
            logger.warning(f"Failed to get broker service: {e}")
        
        # Mock account data
        mock_account = Account(
            account_id="DEMO_ACCOUNT_001",
            equity=125000.00,
            cash=25000.00,
            buying_power=100000.00,
            portfolio_value=125000.00,
            day_trade_buying_power=100000.00,
            pattern_day_trader=False
        )
        
        # Mock positions
        mock_positions = [
            Position(
                symbol="AAPL",
                quantity=100.0,
                market_value=15150.00,
                cost_basis=14800.00,
                unrealized_pl=350.00,
                unrealized_pl_percent=2.36,
                current_price=151.50,
                last_updated=datetime.utcnow()
            ),
            Position(
                symbol="GOOGL",
                quantity=50.0,
                market_value=7125.00,
                cost_basis=7250.00,
                unrealized_pl=-125.00,
                unrealized_pl_percent=-1.72,
                current_price=142.50,
                last_updated=datetime.utcnow()
            ),
            Position(
                symbol="TSLA",
                quantity=-25.0,  # Short position
                market_value=-6125.00,
                cost_basis=-6000.00,
                unrealized_pl=-125.00,
                unrealized_pl_percent=-2.08,
                current_price=245.00,
                last_updated=datetime.utcnow()
            ),
            Position(
                symbol="SPY",
                quantity=200.0,
                market_value=87000.00,
                cost_basis=86400.00,
                unrealized_pl=600.00,
                unrealized_pl_percent=0.69,
                current_price=435.00,
                last_updated=datetime.utcnow()
            )
        ]
        
        return PortfolioResponse(
            account=mock_account,
            positions=mock_positions,
            position_count=len(mock_positions),
            last_updated=datetime.utcnow(),
            message="Retrieved portfolio overview"
        )
        
    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to fetch portfolio",
            error_code="PORTFOLIO_FETCH_ERROR",
            context={"error": str(e)}
        )


@router.get(
    "/positions",
    response_model=Dict[str, Any],
    summary="Get current positions",
    description="Retrieve all current positions with optional filtering"
)
async def get_positions(
    symbol: Optional[str] = Query(None, description="Filter by specific symbol"),
    min_value: Optional[float] = Query(None, description="Minimum position value filter"),
    sort_by: str = Query("market_value", description="Sort field (market_value, unrealized_pl, symbol)"),
    sort_order: str = Query("desc", description="Sort order (asc, desc)"),
    user: Dict[str, Any] = Depends(verify_token)
):
    """Get current positions with filtering and sorting."""
    try:
        # Import broker service
        from services.broker_integration.broker_service import get_broker_service
        
        try:
            broker_service = await get_broker_service()
            # In production: positions = await broker_service.get_positions()
        except Exception as e:
            logger.warning(f"Failed to get broker service: {e}")
        
        # Mock positions data
        mock_positions = [
            Position(
                symbol="AAPL",
                quantity=100.0,
                market_value=15150.00,
                cost_basis=14800.00,
                unrealized_pl=350.00,
                unrealized_pl_percent=2.36,
                current_price=151.50,
                last_updated=datetime.utcnow()
            ),
            Position(
                symbol="GOOGL",
                quantity=50.0,
                market_value=7125.00,
                cost_basis=7250.00,
                unrealized_pl=-125.00,
                unrealized_pl_percent=-1.72,
                current_price=142.50,
                last_updated=datetime.utcnow()
            ),
            Position(
                symbol="TSLA",
                quantity=-25.0,  # Short position
                market_value=-6125.00,
                cost_basis=-6000.00,
                unrealized_pl=-125.00,
                unrealized_pl_percent=-2.08,
                current_price=245.00,
                last_updated=datetime.utcnow()
            ),
            Position(
                symbol="SPY",
                quantity=200.0,
                market_value=87000.00,
                cost_basis=86400.00,
                unrealized_pl=600.00,
                unrealized_pl_percent=0.69,
                current_price=435.00,
                last_updated=datetime.utcnow()
            )
        ]
        
        # Apply filters
        filtered_positions = mock_positions
        
        if symbol:
            filtered_positions = [p for p in filtered_positions if p.symbol == symbol.upper()]
        
        if min_value is not None:
            filtered_positions = [p for p in filtered_positions if abs(p.market_value) >= min_value]
        
        # Apply sorting
        reverse_sort = sort_order.lower() == "desc"
        
        if sort_by == "market_value":
            filtered_positions.sort(key=lambda x: abs(x.market_value), reverse=reverse_sort)
        elif sort_by == "unrealized_pl":
            filtered_positions.sort(key=lambda x: x.unrealized_pl, reverse=reverse_sort)
        elif sort_by == "symbol":
            filtered_positions.sort(key=lambda x: x.symbol, reverse=reverse_sort)
        
        # Calculate summary statistics
        total_market_value = sum(p.market_value for p in filtered_positions)
        total_unrealized_pl = sum(p.unrealized_pl for p in filtered_positions)
        total_cost_basis = sum(p.cost_basis for p in filtered_positions)
        
        winning_positions = len([p for p in filtered_positions if p.unrealized_pl > 0])
        losing_positions = len([p for p in filtered_positions if p.unrealized_pl < 0])
        
        return {
            "success": True,
            "timestamp": datetime.utcnow(),
            "positions": [p.dict() for p in filtered_positions],
            "count": len(filtered_positions),
            "summary": {
                "total_market_value": total_market_value,
                "total_unrealized_pl": total_unrealized_pl,
                "total_cost_basis": total_cost_basis,
                "unrealized_pl_percent": (total_unrealized_pl / abs(total_cost_basis)) * 100 if total_cost_basis != 0 else 0,
                "winning_positions": winning_positions,
                "losing_positions": losing_positions,
                "win_rate": (winning_positions / max(len(filtered_positions), 1)) * 100
            },
            "message": f"Retrieved {len(filtered_positions)} positions"
        }
        
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to fetch positions",
            error_code="POSITIONS_FETCH_ERROR",
            context={"error": str(e)}
        )


@router.get(
    "/positions/{symbol}",
    response_model=Dict[str, Any],
    summary="Get position details",
    description="Get detailed information for a specific position"
)
async def get_position(
    symbol: str = Path(..., description="Stock symbol (e.g., AAPL)"),
    user: Dict[str, Any] = Depends(verify_token)
):
    """Get detailed information for a specific position."""
    try:
        symbol = symbol.upper()
        
        # Mock position lookup
        if symbol == "AAPL":
            position = Position(
                symbol=symbol,
                quantity=100.0,
                market_value=15150.00,
                cost_basis=14800.00,
                unrealized_pl=350.00,
                unrealized_pl_percent=2.36,
                current_price=151.50,
                last_updated=datetime.utcnow()
            )
            
            # Additional position details
            position_details = {
                "position": position.dict(),
                "trade_history": [
                    {
                        "trade_id": "TRD_001",
                        "date": (datetime.utcnow() - timedelta(days=5)).isoformat(),
                        "side": "BUY",
                        "quantity": 50.0,
                        "price": 148.00,
                        "value": 7400.00,
                        "commission": 1.00
                    },
                    {
                        "trade_id": "TRD_002", 
                        "date": (datetime.utcnow() - timedelta(days=3)).isoformat(),
                        "side": "BUY",
                        "quantity": 50.0,
                        "price": 148.00,
                        "value": 7400.00,
                        "commission": 1.00
                    }
                ],
                "risk_metrics": {
                    "position_weight": 12.12,  # % of portfolio
                    "beta": 1.15,
                    "volatility": 0.28,
                    "var_1d": -456.50,  # 1-day VaR at 95% confidence
                    "max_loss_1d": -681.75  # Maximum potential 1-day loss
                },
                "performance": {
                    "days_held": 5,
                    "return_since_purchase": 2.36,
                    "annualized_return": 172.34,
                    "dividend_yield": 0.44
                }
            }
            
            return {
                "success": True,
                "timestamp": datetime.utcnow(),
                "data": position_details,
                "message": f"Retrieved position details for {symbol}"
            }
        else:
            raise APIException(
                status_code=404,
                detail=f"Position for {symbol} not found",
                error_code="POSITION_NOT_FOUND",
                context={"symbol": symbol}
            )
        
    except APIException:
        raise
    except Exception as e:
        logger.error(f"Error fetching position for {symbol}: {e}")
        raise APIException(
            status_code=500,
            detail=f"Failed to fetch position for {symbol}",
            error_code="POSITION_FETCH_ERROR",
            context={"symbol": symbol, "error": str(e)}
        )


@router.get(
    "/performance",
    response_model=PerformanceResponse,
    summary="Get portfolio performance metrics",
    description="Retrieve comprehensive portfolio performance analytics"
)
async def get_performance(
    period: str = Query("1M", description="Performance period (1D, 1W, 1M, 3M, 6M, 1Y, YTD, ALL)"),
    benchmark: str = Query("SPY", description="Benchmark symbol for comparison"),
    user: Dict[str, Any] = Depends(verify_token)
):
    """Get portfolio performance metrics."""
    try:
        # Parse period
        period_days = {
            "1D": 1,
            "1W": 7,
            "1M": 30,
            "3M": 90,
            "6M": 180,
            "1Y": 365,
            "YTD": (datetime.utcnow() - datetime(datetime.utcnow().year, 1, 1)).days,
            "ALL": 365 * 10  # 10 years max
        }
        
        days = period_days.get(period, 30)
        period_start = datetime.utcnow() - timedelta(days=days)
        period_end = datetime.utcnow()
        
        # Mock performance metrics
        mock_metrics = PerformanceMetrics(
            total_return=7250.00,
            total_return_percent=6.15,
            daily_return=125.50,
            daily_return_percent=0.10,
            max_drawdown=-2.85,
            sharpe_ratio=1.42,
            win_rate=68.5,
            profit_factor=1.85,
            avg_win=425.30,
            avg_loss=-235.75
        )
        
        return PerformanceResponse(
            metrics=mock_metrics,
            period_start=period_start,
            period_end=period_end,
            message=f"Retrieved performance metrics for {period} period"
        )
        
    except Exception as e:
        logger.error(f"Error fetching performance metrics: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to fetch performance metrics",
            error_code="PERFORMANCE_FETCH_ERROR",
            context={"error": str(e)}
        )


@router.get(
    "/performance/history",
    response_model=Dict[str, Any],
    summary="Get performance history",
    description="Get historical performance data points for charting"
)
async def get_performance_history(
    period: str = Query("1M", description="History period"),
    interval: str = Query("1D", description="Data interval (1H, 4H, 1D)"),
    user: Dict[str, Any] = Depends(verify_token)
):
    """Get historical performance data for charting."""
    try:
        # Parse parameters
        period_days = {
            "1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365
        }
        
        days = period_days.get(period, 30)
        
        # Generate mock historical performance data
        performance_history = []
        base_value = 100000.0
        
        for i in range(days):
            date = datetime.utcnow() - timedelta(days=days-i)
            # Simulate portfolio growth with some volatility
            daily_return = 0.001 + (hash(str(date.date())) % 100 - 50) * 0.0001
            base_value *= (1 + daily_return)
            
            performance_history.append({
                "date": date.isoformat(),
                "portfolio_value": round(base_value, 2),
                "daily_return": round(daily_return * 100, 3),
                "cumulative_return": round(((base_value - 100000) / 100000) * 100, 2)
            })
        
        # Calculate additional metrics
        total_return = performance_history[-1]["cumulative_return"]
        volatility = 1.2  # Mock volatility
        
        return {
            "success": True,
            "timestamp": datetime.utcnow(),
            "data": {
                "history": performance_history,
                "summary": {
                    "period": period,
                    "total_return": total_return,
                    "volatility": volatility,
                    "data_points": len(performance_history),
                    "start_value": 100000.0,
                    "end_value": performance_history[-1]["portfolio_value"]
                }
            },
            "message": f"Retrieved {len(performance_history)} performance data points"
        }
        
    except Exception as e:
        logger.error(f"Error fetching performance history: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to fetch performance history",
            error_code="PERFORMANCE_HISTORY_ERROR",
            context={"error": str(e)}
        )


@router.get(
    "/allocation",
    response_model=Dict[str, Any],
    summary="Get portfolio allocation",
    description="Get portfolio allocation breakdown by various dimensions"
)
async def get_portfolio_allocation(
    breakdown: str = Query("symbol", description="Breakdown type (symbol, sector, asset_type)"),
    user: Dict[str, Any] = Depends(verify_token)
):
    """Get portfolio allocation breakdown."""
    try:
        if breakdown == "symbol":
            # Symbol-level allocation
            allocations = [
                {"symbol": "SPY", "weight": 69.6, "value": 87000.00, "sector": "ETF"},
                {"symbol": "AAPL", "weight": 12.1, "value": 15150.00, "sector": "Technology"},
                {"symbol": "GOOGL", "weight": 5.7, "value": 7125.00, "sector": "Technology"},
                {"symbol": "TSLA", "weight": -4.9, "value": -6125.00, "sector": "Consumer Discretionary"},  # Short
                {"symbol": "CASH", "weight": 20.0, "value": 25000.00, "sector": "Cash"}
            ]
        
        elif breakdown == "sector":
            # Sector-level allocation
            allocations = [
                {"sector": "ETF", "weight": 69.6, "value": 87000.00},
                {"sector": "Technology", "weight": 17.8, "value": 22275.00},
                {"sector": "Cash", "weight": 20.0, "value": 25000.00},
                {"sector": "Consumer Discretionary", "weight": -4.9, "value": -6125.00}  # Short
            ]
        
        elif breakdown == "asset_type":
            # Asset type allocation
            allocations = [
                {"asset_type": "Equities", "weight": 83.1, "value": 103250.00},
                {"asset_type": "ETFs", "weight": 69.6, "value": 87000.00},
                {"asset_type": "Cash", "weight": 20.0, "value": 25000.00},
                {"asset_type": "Short Positions", "weight": -4.9, "value": -6125.00}
            ]
        
        else:
            raise APIException(
                status_code=400,
                detail="Invalid breakdown type",
                error_code="INVALID_BREAKDOWN"
            )
        
        return {
            "success": True,
            "timestamp": datetime.utcnow(),
            "data": {
                "breakdown_type": breakdown,
                "allocations": allocations,
                "total_count": len(allocations),
                "portfolio_value": 125000.00
            },
            "message": f"Retrieved portfolio allocation by {breakdown}"
        }
        
    except APIException:
        raise
    except Exception as e:
        logger.error(f"Error fetching portfolio allocation: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to fetch portfolio allocation",
            error_code="ALLOCATION_FETCH_ERROR",
            context={"error": str(e)}
        )


@router.get(
    "/account",
    response_model=Dict[str, Any],
    summary="Get account information",
    description="Get detailed account information and trading permissions"
)
async def get_account_info(
    user: Dict[str, Any] = Depends(verify_token)
):
    """Get detailed account information."""
    try:
        # Import broker service
        from services.broker_integration.broker_service import get_broker_service
        
        try:
            broker_service = await get_broker_service()
            # In production: account = await broker_service.get_account()
        except Exception as e:
            logger.warning(f"Failed to get broker service: {e}")
        
        # Mock account data
        account_info = {
            "account": {
                "account_id": "DEMO_ACCOUNT_001",
                "account_type": "MARGIN",
                "status": "ACTIVE",
                "created_date": "2024-01-15T00:00:00Z",
                "equity": 125000.00,
                "cash": 25000.00,
                "buying_power": 100000.00,
                "portfolio_value": 125000.00,
                "day_trade_buying_power": 100000.00,
                "pattern_day_trader": False,
                "trade_suspended": False,
                "account_blocked": False
            },
            "trading_permissions": {
                "stocks": True,
                "options": False,
                "crypto": False,
                "international": False,
                "penny_stocks": False
            },
            "limits": {
                "daily_trades": 3,
                "max_position_size": 0.25,  # 25% of portfolio
                "max_order_value": 50000.00,
                "overnight_buying_power": 50000.00
            },
            "fees": {
                "stock_commission": 0.00,
                "options_commission": 0.65,
                "options_contract_fee": 0.65,
                "regulatory_fees": True
            }
        }
        
        return {
            "success": True,
            "timestamp": datetime.utcnow(),
            "data": account_info,
            "message": "Retrieved account information"
        }
        
    except Exception as e:
        logger.error(f"Error fetching account info: {e}")
        raise APIException(
            status_code=500,
            detail="Failed to fetch account information",
            error_code="ACCOUNT_FETCH_ERROR",
            context={"error": str(e)}
        )