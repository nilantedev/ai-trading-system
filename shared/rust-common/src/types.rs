//! Common data types for the trading system

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Trading symbol representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Symbol {
    /// Base symbol (e.g., "AAPL", "SPY")
    pub ticker: String,
    /// Exchange where the symbol is traded
    pub exchange: Option<String>,
    /// Asset class
    pub asset_class: AssetClass,
}

/// Asset class enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssetClass {
    /// Equity securities
    Equity,
    /// Options contracts
    Option,
    /// Futures contracts
    Future,
    /// Exchange-traded funds
    Etf,
    /// Cryptocurrencies
    Crypto,
    /// Foreign exchange
    Forex,
    /// Fixed income
    Bond,
}

/// Market data quote
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quote {
    /// Symbol
    pub symbol: Symbol,
    /// Bid price
    pub bid: Decimal,
    /// Ask price
    pub ask: Decimal,
    /// Bid size
    pub bid_size: i64,
    /// Ask size
    pub ask_size: i64,
    /// Last trade price
    pub last: Option<Decimal>,
    /// Last trade size
    pub last_size: Option<i64>,
    /// Quote timestamp
    pub timestamp: DateTime<Utc>,
    /// Exchange timestamp
    pub exchange_timestamp: Option<DateTime<Utc>>,
}

/// Trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Symbol
    pub symbol: Symbol,
    /// Trade price
    pub price: Decimal,
    /// Trade size/volume
    pub size: i64,
    /// Trade timestamp
    pub timestamp: DateTime<Utc>,
    /// Exchange timestamp
    pub exchange_timestamp: Option<DateTime<Utc>>,
    /// Trade conditions
    pub conditions: Vec<String>,
}

/// Order side enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    /// Buy order
    Buy,
    /// Sell order
    Sell,
}

/// Order type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    /// Market order
    Market,
    /// Limit order
    Limit,
    /// Stop order
    Stop,
    /// Stop limit order
    StopLimit,
    /// Trailing stop order
    TrailingStop,
}

/// Order time in force
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeInForce {
    /// Good for day
    Day,
    /// Good till canceled
    Gtc,
    /// Immediate or cancel
    Ioc,
    /// Fill or kill
    Fok,
}

/// Order status enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    /// Order pending submission
    Pending,
    /// Order submitted to exchange
    Submitted,
    /// Order accepted by exchange
    Accepted,
    /// Order partially filled
    PartiallyFilled,
    /// Order completely filled
    Filled,
    /// Order canceled
    Canceled,
    /// Order rejected
    Rejected,
    /// Order expired
    Expired,
}

/// Trading order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Unique order ID
    pub id: Uuid,
    /// Client order ID
    pub client_order_id: Option<String>,
    /// Symbol to trade
    pub symbol: Symbol,
    /// Order side (buy/sell)
    pub side: OrderSide,
    /// Order type
    pub order_type: OrderType,
    /// Order quantity
    pub quantity: i64,
    /// Order price (for limit orders)
    pub price: Option<Decimal>,
    /// Stop price (for stop orders)
    pub stop_price: Option<Decimal>,
    /// Time in force
    pub time_in_force: TimeInForce,
    /// Order status
    pub status: OrderStatus,
    /// Filled quantity
    pub filled_quantity: i64,
    /// Average fill price
    pub average_fill_price: Option<Decimal>,
    /// Order creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// Order metadata
    pub metadata: HashMap<String, String>,
}

/// Position in a security
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Symbol
    pub symbol: Symbol,
    /// Position size (positive = long, negative = short)
    pub quantity: i64,
    /// Average entry price
    pub average_price: Decimal,
    /// Current market value
    pub market_value: Decimal,
    /// Unrealized profit/loss
    pub unrealized_pnl: Decimal,
    /// Realized profit/loss for the day
    pub realized_pnl_today: Decimal,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

/// Account information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    /// Account ID
    pub id: String,
    /// Total equity
    pub equity: Decimal,
    /// Available buying power
    pub buying_power: Decimal,
    /// Cash balance
    pub cash: Decimal,
    /// Total market value of positions
    pub market_value: Decimal,
    /// Day trading buying power
    pub day_trading_buying_power: Option<Decimal>,
    /// Pattern day trader status
    pub is_pattern_day_trader: bool,
    /// Account status
    pub status: AccountStatus,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

/// Account status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccountStatus {
    /// Account is active
    Active,
    /// Account is restricted
    Restricted,
    /// Account is closed
    Closed,
    /// Account is suspended
    Suspended,
}

/// Trading signal from AI system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Unique signal ID
    pub id: Uuid,
    /// Symbol
    pub symbol: Symbol,
    /// Signal action
    pub action: SignalAction,
    /// Signal strength (0.0 to 1.0)
    pub strength: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Suggested position size
    pub suggested_size: Option<i64>,
    /// Price target
    pub price_target: Option<Decimal>,
    /// Stop loss level
    pub stop_loss: Option<Decimal>,
    /// Signal reasoning
    pub reasoning: String,
    /// Signal source/model
    pub source: String,
    /// Signal timestamp
    pub timestamp: DateTime<Utc>,
    /// Signal expiration
    pub expires_at: Option<DateTime<Utc>>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Signal action enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalAction {
    /// Buy/long signal
    Buy,
    /// Sell/short signal
    Sell,
    /// Hold current position
    Hold,
    /// Close position
    Close,
}

/// Risk metrics for a position or portfolio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    /// Value at Risk (VaR) 
    pub var_95: Decimal,
    /// Expected Shortfall (Conditional VaR)
    pub expected_shortfall: Decimal,
    /// Beta relative to market
    pub beta: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: Decimal,
    /// Volatility (annualized)
    pub volatility: f64,
    /// Calculation timestamp
    pub calculated_at: DateTime<Utc>,
}

/// Market data subscription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataSubscription {
    /// Subscription ID
    pub id: Uuid,
    /// Symbols to subscribe to
    pub symbols: Vec<Symbol>,
    /// Data types to receive
    pub data_types: Vec<MarketDataType>,
    /// Subscription status
    pub status: SubscriptionStatus,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
}

/// Market data type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketDataType {
    /// Real-time quotes
    Quotes,
    /// Real-time trades
    Trades,
    /// Level 2 order book
    Level2,
    /// Daily bars/candles
    DailyBars,
    /// Intraday bars/candles
    IntradayBars,
    /// Options chains
    OptionsChain,
}

/// Subscription status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubscriptionStatus {
    /// Subscription is active
    Active,
    /// Subscription is paused
    Paused,
    /// Subscription is canceled
    Canceled,
    /// Subscription failed
    Failed,
}

impl Symbol {
    /// Create a new equity symbol
    pub fn equity<S: Into<String>>(ticker: S) -> Self {
        Self {
            ticker: ticker.into(),
            exchange: None,
            asset_class: AssetClass::Equity,
        }
    }

    /// Create a new option symbol
    pub fn option<S: Into<String>>(ticker: S) -> Self {
        Self {
            ticker: ticker.into(),
            exchange: None,
            asset_class: AssetClass::Option,
        }
    }

    /// Create a new ETF symbol
    pub fn etf<S: Into<String>>(ticker: S) -> Self {
        Self {
            ticker: ticker.into(),
            exchange: None,
            asset_class: AssetClass::Etf,
        }
    }

    /// Set the exchange for this symbol
    pub fn with_exchange<S: Into<String>>(mut self, exchange: S) -> Self {
        self.exchange = Some(exchange.into());
        self
    }
}

impl Order {
    /// Create a new market buy order
    pub fn market_buy(symbol: Symbol, quantity: i64) -> Self {
        Self {
            id: Uuid::new_v4(),
            client_order_id: None,
            symbol,
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            quantity,
            price: None,
            stop_price: None,
            time_in_force: TimeInForce::Day,
            status: OrderStatus::Pending,
            filled_quantity: 0,
            average_fill_price: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Create a new limit buy order
    pub fn limit_buy(symbol: Symbol, quantity: i64, price: Decimal) -> Self {
        Self {
            id: Uuid::new_v4(),
            client_order_id: None,
            symbol,
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            quantity,
            price: Some(price),
            stop_price: None,
            time_in_force: TimeInForce::Day,
            status: OrderStatus::Pending,
            filled_quantity: 0,
            average_fill_price: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Check if the order is fully filled
    pub fn is_filled(&self) -> bool {
        self.status == OrderStatus::Filled
    }

    /// Check if the order is still active
    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Pending | OrderStatus::Submitted | OrderStatus::Accepted | OrderStatus::PartiallyFilled
        )
    }

    /// Get remaining quantity to be filled
    pub fn remaining_quantity(&self) -> i64 {
        self.quantity - self.filled_quantity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_creation() {
        let symbol = Symbol::equity("AAPL");
        assert_eq!(symbol.ticker, "AAPL");
        assert_eq!(symbol.asset_class, AssetClass::Equity);
        assert_eq!(symbol.exchange, None);

        let symbol_with_exchange = symbol.with_exchange("NASDAQ");
        assert_eq!(symbol_with_exchange.exchange, Some("NASDAQ".to_string()));
    }

    #[test]
    fn test_order_creation() {
        let symbol = Symbol::equity("AAPL");
        let order = Order::market_buy(symbol, 100);
        
        assert_eq!(order.side, OrderSide::Buy);
        assert_eq!(order.order_type, OrderType::Market);
        assert_eq!(order.quantity, 100);
        assert!(order.is_active());
        assert!(!order.is_filled());
        assert_eq!(order.remaining_quantity(), 100);
    }
}