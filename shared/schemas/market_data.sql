-- Market Data Tables for AI Trading System
-- QuestDB Time-Series Optimized Schema

-- 1. Market Data - High-frequency price and volume data
CREATE TABLE IF NOT EXISTS market_data (
    symbol STRING,
    timestamp TIMESTAMP,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume LONG,
    vwap DOUBLE,
    trade_count INT,
    data_source STRING
) TIMESTAMP(timestamp) PARTITION BY DAY;

-- Create index for fast symbol lookups
CREATE INDEX ON market_data (symbol);

-- 2. Options Data - Options chain and pricing
CREATE TABLE IF NOT EXISTS options_data (
    underlying_symbol STRING,
    option_symbol STRING,
    timestamp TIMESTAMP,
    expiration_date DATE,
    strike_price DOUBLE,
    option_type STRING, -- 'call' or 'put'
    bid DOUBLE,
    ask DOUBLE,
    last_price DOUBLE,
    volume LONG,
    open_interest LONG,
    implied_volatility DOUBLE,
    delta DOUBLE,
    gamma DOUBLE,
    theta DOUBLE,
    vega DOUBLE,
    rho DOUBLE,
    data_source STRING
) TIMESTAMP(timestamp) PARTITION BY DAY;

CREATE INDEX ON options_data (underlying_symbol);
CREATE INDEX ON options_data (option_symbol);

-- 3. Technical Indicators - Computed technical analysis values
CREATE TABLE IF NOT EXISTS technical_indicators (
    symbol STRING,
    timestamp TIMESTAMP,
    indicator_name STRING,
    timeframe STRING, -- '1m', '5m', '1h', '1d', etc.
    value DOUBLE,
    metadata STRING, -- JSON string for additional parameters
    calculation_time TIMESTAMP
) TIMESTAMP(timestamp) PARTITION BY DAY;

CREATE INDEX ON technical_indicators (symbol);
CREATE INDEX ON technical_indicators (indicator_name);

-- 4. News Events - Financial news and sentiment
CREATE TABLE IF NOT EXISTS news_events (
    id STRING,
    timestamp TIMESTAMP,
    headline STRING,
    content STRING,
    source STRING,
    symbols STRING, -- Comma-separated list of related symbols
    sentiment_score DOUBLE, -- -1 to 1, where -1 is very negative, 1 is very positive
    relevance_score DOUBLE, -- 0 to 1, where 1 is highly relevant
    news_type STRING, -- 'earnings', 'merger', 'economic', 'company', etc.
    language STRING,
    url STRING,
    processed_at TIMESTAMP
) TIMESTAMP(timestamp) PARTITION BY DAY;

CREATE INDEX ON news_events (symbols);
CREATE INDEX ON news_events (news_type);

-- 5. Trading Signals - AI-generated trading signals
CREATE TABLE IF NOT EXISTS trading_signals (
    id STRING,
    timestamp TIMESTAMP,
    symbol STRING,
    signal_type STRING, -- 'buy', 'sell', 'hold'
    confidence DOUBLE, -- 0 to 1
    target_price DOUBLE,
    stop_loss DOUBLE,
    take_profit DOUBLE,
    timeframe STRING,
    strategy_name STRING,
    agent_id STRING,
    reasoning STRING, -- AI's reasoning for the signal
    market_conditions STRING, -- JSON string describing market state
    risk_assessment STRING, -- JSON string with risk metrics
    expires_at TIMESTAMP,
    status STRING -- 'active', 'executed', 'expired', 'cancelled'
) TIMESTAMP(timestamp) PARTITION BY DAY;

CREATE INDEX ON trading_signals (symbol);
CREATE INDEX ON trading_signals (status);
CREATE INDEX ON trading_signals (strategy_name);

-- 6. Portfolio Snapshots - Point-in-time portfolio state
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    timestamp TIMESTAMP,
    portfolio_id STRING,
    total_value DOUBLE,
    cash_balance DOUBLE,
    positions_value DOUBLE,
    unrealized_pnl DOUBLE,
    realized_pnl_today DOUBLE,
    buying_power DOUBLE,
    margin_used DOUBLE,
    risk_metrics STRING, -- JSON with VaR, Sharpe ratio, etc.
    positions STRING -- JSON array of current positions
) TIMESTAMP(timestamp) PARTITION BY HOUR;

CREATE INDEX ON portfolio_snapshots (portfolio_id);

-- 7. Order Book Data - Level 2 market data (when available)
CREATE TABLE IF NOT EXISTS order_book (
    symbol STRING,
    timestamp TIMESTAMP,
    bid_price DOUBLE,
    bid_size LONG,
    ask_price DOUBLE,
    ask_size LONG,
    level INT, -- Level 1, 2, 3, etc.
    exchange STRING,
    data_source STRING
) TIMESTAMP(timestamp) PARTITION BY DAY;

CREATE INDEX ON order_book (symbol);

-- 8. Economic Data - Macro economic indicators
CREATE TABLE IF NOT EXISTS economic_data (
    timestamp TIMESTAMP,
    indicator_name STRING, -- 'GDP', 'CPI', 'unemployment_rate', etc.
    value DOUBLE,
    previous_value DOUBLE,
    forecast_value DOUBLE,
    country STRING,
    release_date TIMESTAMP,
    importance STRING, -- 'high', 'medium', 'low'
    source STRING,
    unit STRING -- '%', 'billions', 'index', etc.
) TIMESTAMP(timestamp) PARTITION BY MONTH;

CREATE INDEX ON economic_data (indicator_name);
CREATE INDEX ON economic_data (country);

-- 9. Model Performance Tracking - AI model accuracy metrics
CREATE TABLE IF NOT EXISTS model_performance (
    timestamp TIMESTAMP,
    model_id STRING,
    model_version STRING,
    metric_name STRING, -- 'accuracy', 'precision', 'recall', 'sharpe', 'max_drawdown'
    metric_value DOUBLE,
    evaluation_period STRING, -- '1d', '7d', '30d'
    data_points INT,
    symbol STRING, -- NULL for portfolio-level metrics
    strategy STRING
) TIMESTAMP(timestamp) PARTITION BY DAY;

CREATE INDEX ON model_performance (model_id);
CREATE INDEX ON model_performance (metric_name);

-- 10. Risk Events - Risk management triggers and alerts
CREATE TABLE IF NOT EXISTS risk_events (
    id STRING,
    timestamp TIMESTAMP,
    event_type STRING, -- 'position_limit', 'var_breach', 'drawdown', 'correlation'
    severity STRING, -- 'low', 'medium', 'high', 'critical'
    symbol STRING,
    portfolio_id STRING,
    current_value DOUBLE,
    threshold_value DOUBLE,
    action_taken STRING,
    description STRING,
    resolved_at TIMESTAMP,
    resolved_by STRING
) TIMESTAMP(timestamp) PARTITION BY DAY;

CREATE INDEX ON risk_events (event_type);
CREATE INDEX ON risk_events (severity);
CREATE INDEX ON risk_events (portfolio_id);