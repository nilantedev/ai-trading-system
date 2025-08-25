#!/usr/bin/env python3
"""
API Models - Pydantic models for request/response validation and serialization
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
import re


# Base response model
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = Field(True, description="Request success status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    message: Optional[str] = Field(None, description="Optional response message")


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = Field(False, description="Request success status")
    error: Dict[str, Any] = Field(..., description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


# Market Data Models
class MarketDataRequest(BaseModel):
    """Market data request parameters."""
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    timeframe: str = Field("1min", description="Data timeframe")
    start_date: Optional[datetime] = Field(None, description="Start date for historical data")
    end_date: Optional[datetime] = Field(None, description="End date for historical data")
    limit: int = Field(100, ge=1, le=5000, description="Maximum number of records")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not re.match(r'^[A-Z]{1,5}$', v.upper()):
            raise ValueError('Invalid symbol format')
        return v.upper()
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        valid_timeframes = ['1min', '5min', '15min', '30min', '1hour', '1day', '1week', '1month']
        if v not in valid_timeframes:
            raise ValueError(f'Invalid timeframe. Must be one of: {valid_timeframes}')
        return v


class MarketDataPoint(BaseModel):
    """Single market data point."""
    symbol: str = Field(..., description="Stock symbol")
    timestamp: datetime = Field(..., description="Data timestamp")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Closing price")
    volume: int = Field(..., description="Trading volume")
    timeframe: str = Field(..., description="Data timeframe")
    data_source: str = Field(..., description="Data provider source")


class MarketDataResponse(BaseResponse):
    """Market data response."""
    data: List[MarketDataPoint] = Field(..., description="Market data points")
    count: int = Field(..., description="Number of data points returned")
    symbol: str = Field(..., description="Requested symbol")


class SymbolInfo(BaseModel):
    """Symbol information."""
    symbol: str = Field(..., description="Stock symbol")
    name: str = Field(..., description="Company name")
    exchange: str = Field(..., description="Exchange")
    sector: Optional[str] = Field(None, description="Sector")
    industry: Optional[str] = Field(None, description="Industry")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    active: bool = Field(True, description="Whether symbol is actively traded")


class SymbolsResponse(BaseResponse):
    """Available symbols response."""
    symbols: List[SymbolInfo] = Field(..., description="Available trading symbols")
    count: int = Field(..., description="Number of symbols")


# Trading Signal Models
class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TradingSignal(BaseModel):
    """Trading signal data."""
    symbol: str = Field(..., description="Stock symbol")
    signal_type: SignalType = Field(..., description="Signal type")
    confidence: float = Field(..., ge=0, le=1, description="Signal confidence (0-1)")
    strength: float = Field(..., ge=0, le=1, description="Signal strength (0-1)")
    price_target: Optional[float] = Field(None, description="Target price")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    strategy_name: str = Field(..., description="Strategy that generated signal")
    reasoning: str = Field(..., description="Signal reasoning")
    timestamp: datetime = Field(..., description="Signal generation time")
    risk_level: RiskLevel = Field(..., description="Risk assessment")


class SignalsResponse(BaseResponse):
    """Trading signals response."""
    signals: List[TradingSignal] = Field(..., description="Trading signals")
    count: int = Field(..., description="Number of signals")
    generated_at: datetime = Field(..., description="Signals generation timestamp")


# Order Models
class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class OrderRequest(BaseModel):
    """Order creation request."""
    symbol: str = Field(..., description="Stock symbol")
    side: OrderSide = Field(..., description="Order side (buy/sell)")
    order_type: OrderType = Field(OrderType.MARKET, description="Order type")
    quantity: float = Field(..., gt=0, description="Order quantity")
    price: Optional[float] = Field(None, gt=0, description="Limit price (required for limit orders)")
    stop_price: Optional[float] = Field(None, gt=0, description="Stop price (for stop orders)")
    time_in_force: TimeInForce = Field(TimeInForce.DAY, description="Time in force")
    client_order_id: Optional[str] = Field(None, description="Client-provided order ID")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return v.upper()
    
    @validator('price')
    def validate_price_for_limit_orders(cls, v, values):
        if values.get('order_type') == OrderType.LIMIT and v is None:
            raise ValueError('Price is required for limit orders')
        return v


class Fill(BaseModel):
    """Order fill information."""
    fill_id: str = Field(..., description="Fill ID")
    quantity: float = Field(..., description="Fill quantity")
    price: float = Field(..., description="Fill price")
    timestamp: datetime = Field(..., description="Fill timestamp")
    commission: float = Field(0.0, description="Commission paid")


class Order(BaseModel):
    """Order information."""
    order_id: str = Field(..., description="Order ID")
    client_order_id: Optional[str] = Field(None, description="Client order ID")
    symbol: str = Field(..., description="Stock symbol")
    side: OrderSide = Field(..., description="Order side")
    order_type: OrderType = Field(..., description="Order type")
    quantity: float = Field(..., description="Order quantity")
    price: Optional[float] = Field(None, description="Order price")
    stop_price: Optional[float] = Field(None, description="Stop price")
    time_in_force: TimeInForce = Field(..., description="Time in force")
    status: OrderStatus = Field(..., description="Order status")
    filled_quantity: float = Field(0.0, description="Filled quantity")
    remaining_quantity: float = Field(..., description="Remaining quantity")
    average_fill_price: Optional[float] = Field(None, description="Average fill price")
    created_at: datetime = Field(..., description="Order creation time")
    updated_at: Optional[datetime] = Field(None, description="Last update time")
    fills: List[Fill] = Field(default_factory=list, description="Order fills")
    commission: float = Field(0.0, description="Total commission")


class OrderResponse(BaseResponse):
    """Single order response."""
    order: Order = Field(..., description="Order details")


class OrdersResponse(BaseResponse):
    """Multiple orders response."""
    orders: List[Order] = Field(..., description="Orders list")
    count: int = Field(..., description="Number of orders")


class CancelOrderResponse(BaseResponse):
    """Cancel order response."""
    order_id: str = Field(..., description="Cancelled order ID")
    cancelled_at: datetime = Field(..., description="Cancellation timestamp")


# Portfolio Models
class Position(BaseModel):
    """Portfolio position."""
    symbol: str = Field(..., description="Stock symbol")
    quantity: float = Field(..., description="Position quantity")
    market_value: float = Field(..., description="Current market value")
    cost_basis: float = Field(..., description="Cost basis")
    unrealized_pl: float = Field(..., description="Unrealized P&L")
    unrealized_pl_percent: float = Field(..., description="Unrealized P&L percentage")
    current_price: float = Field(..., description="Current market price")
    last_updated: datetime = Field(..., description="Last update timestamp")


class Account(BaseModel):
    """Account information."""
    account_id: str = Field(..., description="Account ID")
    equity: float = Field(..., description="Total equity")
    cash: float = Field(..., description="Available cash")
    buying_power: float = Field(..., description="Buying power")
    portfolio_value: float = Field(..., description="Total portfolio value")
    day_trade_buying_power: float = Field(..., description="Day trading buying power")
    pattern_day_trader: bool = Field(..., description="Pattern day trader status")


class PortfolioResponse(BaseResponse):
    """Portfolio overview response."""
    account: Account = Field(..., description="Account information")
    positions: List[Position] = Field(..., description="Current positions")
    position_count: int = Field(..., description="Number of positions")
    last_updated: datetime = Field(..., description="Last update timestamp")


class PerformanceMetrics(BaseModel):
    """Portfolio performance metrics."""
    total_return: float = Field(..., description="Total return")
    total_return_percent: float = Field(..., description="Total return percentage")
    daily_return: float = Field(..., description="Daily return")
    daily_return_percent: float = Field(..., description="Daily return percentage")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    win_rate: float = Field(..., description="Win rate percentage")
    profit_factor: float = Field(..., description="Profit factor")
    avg_win: float = Field(..., description="Average winning trade")
    avg_loss: float = Field(..., description="Average losing trade")


class PerformanceResponse(BaseResponse):
    """Performance metrics response."""
    metrics: PerformanceMetrics = Field(..., description="Performance metrics")
    period_start: datetime = Field(..., description="Performance period start")
    period_end: datetime = Field(..., description="Performance period end")


# Risk Models
class RiskMetrics(BaseModel):
    """Risk assessment metrics."""
    symbol: str = Field(..., description="Stock symbol")
    risk_score: float = Field(..., ge=0, le=100, description="Risk score (0-100)")
    risk_level: RiskLevel = Field(..., description="Risk level")
    volatility: float = Field(..., description="Price volatility")
    volume_ratio: float = Field(..., description="Volume vs average ratio")
    price_change_1h: float = Field(..., description="1-hour price change")
    price_change_24h: float = Field(..., description="24-hour price change")
    liquidity_score: float = Field(..., ge=0, le=1, description="Liquidity score")
    timestamp: datetime = Field(..., description="Analysis timestamp")


class RiskAlert(BaseModel):
    """Risk alert notification."""
    alert_id: str = Field(..., description="Alert ID")
    symbol: str = Field(..., description="Stock symbol")
    alert_type: str = Field(..., description="Alert type")
    risk_level: RiskLevel = Field(..., description="Risk level")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")
    severity_score: float = Field(..., ge=0, le=100, description="Severity score")
    recommended_action: str = Field(..., description="Recommended action")
    timestamp: datetime = Field(..., description="Alert timestamp")


class RiskResponse(BaseResponse):
    """Risk assessment response."""
    risk_metrics: List[RiskMetrics] = Field(..., description="Risk metrics for symbols")
    alerts: List[RiskAlert] = Field(..., description="Active risk alerts")
    portfolio_risk_score: float = Field(..., description="Overall portfolio risk score")


# System Models
class ServiceHealth(BaseModel):
    """Service health status."""
    service: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Service metrics")
    connections: Dict[str, bool] = Field(default_factory=dict, description="Connection status")


class SystemHealth(BaseModel):
    """Overall system health."""
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="System version")
    uptime: float = Field(..., description="System uptime in seconds")
    services: Dict[str, ServiceHealth] = Field(..., description="Individual service health")
    summary: Dict[str, int] = Field(..., description="Health summary statistics")


class SystemHealthResponse(BaseResponse):
    """System health response."""
    health: SystemHealth = Field(..., description="System health information")


# WebSocket Models
class WebSocketMessage(BaseModel):
    """Base WebSocket message."""
    type: str = Field(..., description="Message type")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    data: Dict[str, Any] = Field(..., description="Message data")


class MarketDataUpdate(WebSocketMessage):
    """Market data WebSocket update."""
    type: str = Field("market_data", description="Message type")
    data: MarketDataPoint = Field(..., description="Market data point")


class SignalUpdate(WebSocketMessage):
    """Trading signal WebSocket update."""
    type: str = Field("signal", description="Message type")
    data: TradingSignal = Field(..., description="Trading signal")


class OrderUpdate(WebSocketMessage):
    """Order status WebSocket update."""
    type: str = Field("order", description="Message type")
    data: Order = Field(..., description="Order information")


class AlertUpdate(WebSocketMessage):
    """Alert WebSocket update."""
    type: str = Field("alert", description="Message type")
    data: RiskAlert = Field(..., description="Risk alert")


# News Models
class NewsArticle(BaseModel):
    """News article information."""
    article_id: str = Field(..., description="Article ID")
    title: str = Field(..., description="Article title")
    description: str = Field(..., description="Article description")
    url: str = Field(..., description="Article URL")
    source: str = Field(..., description="News source")
    published_at: datetime = Field(..., description="Publication timestamp")
    symbols: List[str] = Field(..., description="Related stock symbols")
    sentiment_score: float = Field(..., ge=-1, le=1, description="Sentiment score (-1 to 1)")
    sentiment_label: str = Field(..., description="Sentiment label")
    impact_score: float = Field(..., ge=0, le=1, description="Predicted market impact")


class NewsResponse(BaseResponse):
    """News articles response."""
    articles: List[NewsArticle] = Field(..., description="News articles")
    count: int = Field(..., description="Number of articles")


class SentimentAnalysis(BaseModel):
    """Sentiment analysis for a symbol."""
    symbol: str = Field(..., description="Stock symbol")
    overall_sentiment: float = Field(..., ge=-1, le=1, description="Overall sentiment score")
    sentiment_label: str = Field(..., description="Sentiment label")
    confidence: float = Field(..., ge=0, le=1, description="Analysis confidence")
    article_count: int = Field(..., description="Number of articles analyzed")
    positive_count: int = Field(..., description="Positive articles count")
    negative_count: int = Field(..., description="Negative articles count")
    neutral_count: int = Field(..., description="Neutral articles count")
    trend: str = Field(..., description="Sentiment trend")
    timestamp: datetime = Field(..., description="Analysis timestamp")


class SentimentResponse(BaseResponse):
    """Sentiment analysis response."""
    sentiment: SentimentAnalysis = Field(..., description="Sentiment analysis")


# Pagination Models
class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(50, ge=1, le=1000, description="Page size")
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


class PaginatedResponse(BaseResponse):
    """Paginated response base."""
    page: int = Field(..., description="Current page")
    size: int = Field(..., description="Page size")
    total: int = Field(..., description="Total items")
    pages: int = Field(..., description="Total pages")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")