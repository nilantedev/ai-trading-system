#!/usr/bin/env python3
"""
API Models - Pydantic models for request/response validation and serialization
"""

from pydantic import BaseModel, Field
try:  # Pydantic v2
    from pydantic import field_validator
except ImportError:  # pragma: no cover - fallback if still v1
    from pydantic import validator as field_validator  # type: ignore
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
    
    @field_validator('symbol')
    @classmethod
    def _validate_symbol(cls, v: str):
        if not re.match(r'^[A-Z]{1,5}$', v.upper()):
            raise ValueError('Invalid symbol format')
        return v.upper()

    @field_validator('timeframe')
    @classmethod
    def _validate_timeframe(cls, v: str):
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
    
    @field_validator('symbol')
    @classmethod
    def _order_symbol(cls, v: str):
        return v.upper()

    @field_validator('price')
    @classmethod
    def _order_price(cls, v, values):  # type: ignore[override]
        try:
            order_type = values.get('order_type')
        except Exception:  # pragma: no cover - defensive
            order_type = None
        if order_type == OrderType.LIMIT and v is None:
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


class PortfolioPosition(BaseModel):
    """Portfolio position tracking model."""
    symbol: str = Field(..., description="Stock symbol")
    quantity: int = Field(..., description="Number of shares")
    average_price: float = Field(..., description="Average entry price")
    current_price: float = Field(..., description="Current market price")
    market_value: float = Field(..., description="Current market value")
    cost_basis: float = Field(..., description="Total cost basis")
    unrealized_pnl: float = Field(..., description="Unrealized P&L")
    realized_pnl: float = Field(..., description="Realized P&L")
    position_type: str = Field(..., description="LONG or SHORT")
    opened_at: datetime = Field(..., description="Position open timestamp")
    last_updated: datetime = Field(..., description="Last update timestamp")


class Trade(BaseModel):
    """Trade execution model."""
    trade_id: str = Field(..., description="Unique trade identifier")
    order_id: str = Field(..., description="Associated order ID")
    symbol: str = Field(..., description="Stock symbol")
    side: str = Field(..., description="BUY or SELL")
    quantity: int = Field(..., description="Number of shares")
    price: float = Field(..., description="Execution price")
    value: float = Field(..., description="Trade value")
    commission: float = Field(0.0, description="Trade commission")
    executed_at: datetime = Field(..., description="Execution timestamp")
    venue: Optional[str] = Field(None, description="Execution venue")


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
    # Advanced risk metrics
    var_95: Optional[float] = Field(None, description="95% Value at Risk")
    cvar_95: Optional[float] = Field(None, description="95% Conditional VaR")
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    sortino_ratio: Optional[float] = Field(None, description="Sortino ratio")
    max_drawdown: Optional[float] = Field(None, description="Maximum drawdown")


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


# Market Regime Models
class MarketRegime(str, Enum):
    BULL_MARKET = "BULL_MARKET"
    BEAR_MARKET = "BEAR_MARKET"
    SIDEWAYS_MARKET = "SIDEWAYS_MARKET"
    VOLATILE_MARKET = "VOLATILE_MARKET"
    UNKNOWN = "UNKNOWN"


class RegimeDetection(BaseModel):
    """Market regime detection results."""
    symbol: str = Field(..., description="Stock symbol")
    regime: MarketRegime = Field(..., description="Current market regime")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    regime_duration: int = Field(..., description="Duration in current regime (periods)")
    transition_probabilities: Dict[str, float] = Field(..., description="Probability of transitioning to other regimes")
    regime_adjustment: float = Field(..., description="Position size adjustment factor")
    timestamp: datetime = Field(..., description="Detection timestamp")


class MarketRegimeDetection(BaseModel):
    """Market regime detection request/response model."""
    symbol: str = Field(..., description="Stock symbol to analyze")
    lookback_periods: int = Field(default=100, description="Number of periods to analyze")
    regime: Optional[MarketRegime] = Field(None, description="Detected regime")
    confidence: Optional[float] = Field(None, description="Detection confidence")
    indicators: Optional[Dict[str, float]] = Field(None, description="Regime indicators")
    timestamp: Optional[datetime] = Field(None, description="Analysis timestamp")


# Options Models
class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class Greeks(BaseModel):
    """Option Greeks."""
    delta: float = Field(..., description="Rate of change of option price with stock price")
    gamma: float = Field(..., description="Rate of change of delta with stock price")
    theta: float = Field(..., description="Rate of change of option price with time")
    vega: float = Field(..., description="Rate of change of option price with volatility")
    rho: float = Field(..., description="Rate of change of option price with interest rate")


class OptionPricing(BaseModel):
    """Option pricing information."""
    symbol: str = Field(..., description="Underlying symbol")
    strike_price: float = Field(..., description="Strike price")
    expiration: datetime = Field(..., description="Expiration date")
    option_type: OptionType = Field(..., description="Option type (call/put)")
    price: float = Field(..., description="Option price (Black-Scholes)")
    implied_volatility: float = Field(..., description="Implied volatility")
    greeks: Greeks = Field(..., description="Option Greeks")
    underlying_price: float = Field(..., description="Current underlying price")
    time_to_expiry: float = Field(..., description="Time to expiry in years")


class HedgeRecommendation(BaseModel):
    """Options hedge recommendation."""
    symbol: str = Field(..., description="Symbol to hedge")
    hedge_type: str = Field(..., description="Type of hedge (protective_put, covered_call, etc.)")
    strike_price: float = Field(..., description="Recommended strike price")
    expiration: datetime = Field(..., description="Recommended expiration")


class OptionsAnalysis(BaseModel):
    """Options analysis request/response model."""
    symbol: str = Field(..., description="Underlying symbol")
    analysis_type: str = Field(..., description="Type of analysis: pricing, greeks, strategy")
    option_chain: Optional[List[OptionPricing]] = Field(None, description="Option chain data")
    recommended_strategy: Optional[str] = Field(None, description="Recommended options strategy")
    hedge_recommendation: Optional[HedgeRecommendation] = Field(None, description="Hedge recommendation")
    max_profit: Optional[float] = Field(None, description="Maximum profit potential")
    max_loss: Optional[float] = Field(None, description="Maximum loss potential")
    breakeven_points: Optional[List[float]] = Field(None, description="Breakeven price points")
    timestamp: Optional[datetime] = Field(None, description="Analysis timestamp")
    contracts: Optional[int] = Field(None, description="Number of contracts needed")
    premium_cost: Optional[float] = Field(None, description="Total premium cost")
    hedge_effectiveness: Optional[float] = Field(None, ge=0, le=1, description="Hedge effectiveness ratio")
    greeks: Greeks = Field(..., description="Combined position Greeks")


# Portfolio Optimization Models
class OptimizationMethod(str, Enum):
    MARKOWITZ = "markowitz"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    HIERARCHICAL_RISK_PARITY = "hrp"
    KELLY_CRITERION = "kelly"


class PortfolioOptimization(BaseModel):
    """Portfolio optimization results."""
    method: OptimizationMethod = Field(..., description="Optimization method used")
    weights: Dict[str, float] = Field(..., description="Optimal portfolio weights")
    expected_return: float = Field(..., description="Expected portfolio return")
    expected_volatility: float = Field(..., description="Expected portfolio volatility")
    sharpe_ratio: float = Field(..., description="Expected Sharpe ratio")
    efficient_frontier: Optional[List[Dict]] = Field(None, description="Efficient frontier points")
    timestamp: datetime = Field(..., description="Optimization timestamp")


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