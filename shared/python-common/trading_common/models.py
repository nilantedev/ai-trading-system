"""Data models for AI Trading System."""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator, model_validator
import json


class TimeFrame(str, Enum):
    """Supported timeframes for market data."""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"


class SignalType(str, Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class OptionType(str, Enum):
    """Option types."""
    CALL = "call"
    PUT = "put"


class OrderStatus(str, Enum):
    """Trading signal/order status."""
    ACTIVE = "active"
    EXECUTED = "executed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class Severity(str, Enum):
    """Risk event severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MarketData(BaseModel):
    """Market data model for OHLCV data."""
    symbol: str = Field(..., description="Stock symbol (e.g., 'AAPL')")
    timestamp: datetime = Field(..., description="Data timestamp")
    open: float = Field(..., ge=0, description="Opening price")
    high: float = Field(..., ge=0, description="High price")
    low: float = Field(..., ge=0, description="Low price")
    close: float = Field(..., ge=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    vwap: Optional[float] = Field(None, ge=0, description="Volume-weighted average price")
    trade_count: Optional[int] = Field(None, ge=0, description="Number of trades")
    data_source: str = Field(..., description="Data provider (e.g., 'polygon', 'alpaca')")

    @validator('high')
    def high_gte_low(cls, v, values):
        """Validate high >= low."""
        if 'low' in values and v < values['low']:
            raise ValueError('High price must be >= low price')
        return v

    @validator('low')
    def low_lte_high(cls, v, values):
        """Validate low <= high."""
        if 'high' in values and v > values['high']:
            raise ValueError('Low price must be <= high price')
        return v

    class Config:
        validate_assignment = True


class OptionsData(BaseModel):
    """Options pricing and Greeks data."""
    underlying_symbol: str = Field(..., description="Underlying stock symbol")
    option_symbol: str = Field(..., description="Option contract symbol")
    timestamp: datetime = Field(..., description="Data timestamp")
    expiration_date: date = Field(..., description="Option expiration date")
    strike_price: float = Field(..., gt=0, description="Strike price")
    option_type: OptionType = Field(..., description="Option type (call/put)")
    bid: Optional[float] = Field(None, ge=0, description="Bid price")
    ask: Optional[float] = Field(None, ge=0, description="Ask price")
    last_price: Optional[float] = Field(None, ge=0, description="Last traded price")
    volume: Optional[int] = Field(None, ge=0, description="Trading volume")
    open_interest: Optional[int] = Field(None, ge=0, description="Open interest")
    implied_volatility: Optional[float] = Field(None, ge=0, le=10, description="Implied volatility")
    delta: Optional[float] = Field(None, ge=-1, le=1, description="Delta Greek")
    gamma: Optional[float] = Field(None, ge=0, description="Gamma Greek")
    theta: Optional[float] = Field(None, description="Theta Greek")
    vega: Optional[float] = Field(None, ge=0, description="Vega Greek")
    rho: Optional[float] = Field(None, description="Rho Greek")
    data_source: str = Field(..., description="Data provider")

    @validator('ask')
    def ask_gte_bid(cls, v, values):
        """Validate ask >= bid when both present."""
        if v is not None and 'bid' in values and values['bid'] is not None:
            if v < values['bid']:
                raise ValueError('Ask price must be >= bid price')
        return v

    class Config:
        validate_assignment = True


class TechnicalIndicator(BaseModel):
    """Technical indicator calculation result."""
    symbol: str = Field(..., description="Stock symbol")
    timestamp: datetime = Field(..., description="Calculation timestamp")
    indicator_name: str = Field(..., description="Indicator name (e.g., 'sma_20', 'rsi')")
    timeframe: TimeFrame = Field(..., description="Timeframe for calculation")
    value: float = Field(..., description="Indicator value")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")
    calculation_time: datetime = Field(default_factory=datetime.utcnow, description="When calculation was performed")

    @validator('metadata', pre=True)
    def parse_metadata(cls, v):
        """Parse metadata from JSON string if needed."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v or {}

    class Config:
        validate_assignment = True


class NewsEvent(BaseModel):
    """Financial news event with sentiment analysis."""
    id: str = Field(..., description="Unique news event ID")
    timestamp: datetime = Field(..., description="News publication timestamp")
    headline: str = Field(..., min_length=5, max_length=500, description="News headline")
    content: str = Field(..., min_length=10, description="Full news content")
    source: str = Field(..., description="News source (e.g., 'reuters', 'bloomberg')")
    symbols: List[str] = Field(default_factory=list, description="Related stock symbols")
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1, description="Sentiment score (-1 to 1)")
    relevance_score: Optional[float] = Field(None, ge=0, le=1, description="Relevance score (0 to 1)")
    news_type: Optional[str] = Field(None, description="News category")
    language: str = Field(default="en", description="Content language")
    url: Optional[str] = Field(None, description="Source URL")
    processed_at: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")

    @validator('symbols', pre=True)
    def parse_symbols(cls, v):
        """Parse symbols from comma-separated string if needed."""
        if isinstance(v, str):
            return [s.strip().upper() for s in v.split(',') if s.strip()]
        return [s.upper() for s in v] if v else []

    class Config:
        validate_assignment = True


class NewsItem(BaseModel):
    """News article with sentiment analysis."""
    title: str = Field(..., description="Article title")
    content: str = Field(..., description="Article content")
    source: str = Field(..., description="News source")
    published_at: datetime = Field(..., description="Publication timestamp")
    url: str = Field(..., description="Article URL")
    sentiment_score: float = Field(..., description="Sentiment score (-1 to 1)")
    relevance_score: float = Field(..., description="Relevance score (0 to 1)")
    symbols: List[str] = Field(default_factory=list, description="Related symbols")
    
    class Config:
        validate_assignment = True


class SocialSentiment(BaseModel):
    """Social media sentiment data."""
    platform: str = Field(..., description="Social media platform")
    content: str = Field(..., description="Post content")
    author: str = Field(..., description="Author username")
    timestamp: datetime = Field(..., description="Post timestamp")
    sentiment_score: float = Field(..., description="Sentiment score (-1 to 1)")
    engagement_score: float = Field(..., description="Engagement metrics")
    symbols: List[str] = Field(default_factory=list, description="Mentioned symbols")
    
    class Config:
        validate_assignment = True


class TradingSignal(BaseModel):
    """AI-generated trading signal."""
    id: str = Field(..., description="Unique signal ID")
    timestamp: datetime = Field(..., description="Signal generation timestamp")
    symbol: str = Field(..., description="Target stock symbol")
    signal_type: SignalType = Field(..., description="Signal type (buy/sell/hold)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level (0 to 1)")
    target_price: Optional[float] = Field(None, gt=0, description="Target price")
    stop_loss: Optional[float] = Field(None, gt=0, description="Stop loss price")
    take_profit: Optional[float] = Field(None, gt=0, description="Take profit price")
    timeframe: TimeFrame = Field(..., description="Signal timeframe")
    strategy_name: str = Field(..., description="Strategy that generated signal")
    agent_id: str = Field(..., description="AI agent ID")
    reasoning: Optional[str] = Field(None, description="AI reasoning for signal")
    market_conditions: Optional[Dict[str, Any]] = Field(None, description="Market state data")
    risk_assessment: Optional[Dict[str, Any]] = Field(None, description="Risk metrics")
    expires_at: Optional[datetime] = Field(None, description="Signal expiration time")
    status: OrderStatus = Field(default=OrderStatus.ACTIVE, description="Signal status")

    @model_validator(mode='after')
    def validate_prices(self):
        """Validate price relationships."""
        signal_type = self.signal_type
        target = self.target_price
        stop_loss = self.stop_loss
        take_profit = self.take_profit

        if signal_type == SignalType.BUY:
            if stop_loss and target and stop_loss >= target:
                raise ValueError('For buy signals, stop loss must be < target price')
            if take_profit and target and take_profit <= target:
                raise ValueError('For buy signals, take profit must be > target price')
        elif signal_type == SignalType.SELL:
            if stop_loss and target and stop_loss <= target:
                raise ValueError('For sell signals, stop loss must be > target price')
            if take_profit and target and take_profit >= target:
                raise ValueError('For sell signals, take profit must be < target price')

        return self

    class Config:
        validate_assignment = True


class Position(BaseModel):
    """Portfolio position."""
    symbol: str = Field(..., description="Stock symbol")
    quantity: int = Field(..., description="Position size (negative for short)")
    average_price: float = Field(..., gt=0, description="Average entry price")
    current_price: float = Field(..., gt=0, description="Current market price")
    market_value: float = Field(..., description="Current market value")
    unrealized_pnl: float = Field(..., description="Unrealized P&L")
    day_pnl: float = Field(..., description="Day P&L")


class PortfolioSnapshot(BaseModel):
    """Point-in-time portfolio state."""
    timestamp: datetime = Field(..., description="Snapshot timestamp")
    portfolio_id: str = Field(..., description="Portfolio identifier")
    total_value: float = Field(..., ge=0, description="Total portfolio value")
    cash_balance: float = Field(..., ge=0, description="Cash balance")
    positions_value: float = Field(..., description="Total positions value")
    unrealized_pnl: float = Field(..., description="Unrealized P&L")
    realized_pnl_today: float = Field(..., description="Today's realized P&L")
    buying_power: float = Field(..., ge=0, description="Available buying power")
    margin_used: Optional[float] = Field(None, ge=0, description="Margin used")
    risk_metrics: Optional[Dict[str, float]] = Field(None, description="Risk calculations")
    positions: List[Position] = Field(default_factory=list, description="Current positions")

    @validator('positions_value')
    def positions_value_matches(cls, v, values):
        """Validate positions value matches sum of position values."""
        positions = values.get('positions', [])
        if positions:
            calculated_value = sum(pos.market_value for pos in positions)
            if abs(calculated_value - v) > 0.01:  # Allow for small rounding differences
                raise ValueError(f'Positions value {v} does not match calculated {calculated_value}')
        return v

    class Config:
        validate_assignment = True


class RiskEvent(BaseModel):
    """Risk management event/alert."""
    id: str = Field(..., description="Unique event ID")
    timestamp: datetime = Field(..., description="Event timestamp")
    event_type: str = Field(..., description="Risk event type")
    severity: Severity = Field(..., description="Event severity")
    symbol: Optional[str] = Field(None, description="Related symbol")
    portfolio_id: Optional[str] = Field(None, description="Related portfolio")
    current_value: float = Field(..., description="Current metric value")
    threshold_value: float = Field(..., description="Threshold that was breached")
    action_taken: Optional[str] = Field(None, description="Automated action taken")
    description: str = Field(..., description="Event description")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    resolved_by: Optional[str] = Field(None, description="Who/what resolved the event")

    class Config:
        validate_assignment = True


class ModelPerformance(BaseModel):
    """AI model performance tracking."""
    timestamp: datetime = Field(..., description="Measurement timestamp")
    model_id: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version")
    metric_name: str = Field(..., description="Performance metric name")
    metric_value: float = Field(..., description="Metric value")
    evaluation_period: str = Field(..., description="Evaluation time period")
    data_points: int = Field(..., gt=0, description="Number of data points used")
    symbol: Optional[str] = Field(None, description="Symbol-specific metric")
    strategy: Optional[str] = Field(None, description="Strategy name")

    class Config:
        validate_assignment = True


# Response models for API endpoints
class ValidationResponse(BaseModel):
    """Standard validation response."""
    success: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    data: Optional[Dict[str, Any]] = None


class DatabaseOperationResponse(BaseModel):
    """Database operation result."""
    success: bool
    affected_rows: int
    execution_time_ms: float
    query_id: Optional[str] = None
    errors: List[str] = Field(default_factory=list)