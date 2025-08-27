# Multi-Frequency Data Ingestion System

## Overview

The enhanced AI Trading System now supports **multi-frequency data ingestion** with intelligent routing based on trading strategies and market conditions. The system is robust enough to support any trading strategy - from high-frequency scalping to long-term position trading - as long as the data supports the decision.

## Key Features

### 1. **Tiered Data Velocity**
- **REALTIME** (WebSocket streaming): Millisecond latency for critical market data
- **URGENT** (30 seconds): Breaking news, major price moves, earnings
- **FAST** (2 minutes): Active trading signals, momentum changes
- **STANDARD** (15 minutes): Regular market updates, technical analysis
- **BACKGROUND** (1 hour): Historical analysis, fundamental data

### 2. **Strategy-Aware Routing**
The system automatically routes data to appropriate strategies based on:
- Data type and urgency
- Market conditions (volatility, trend, volume)
- Active trading strategies
- Resource availability

### 3. **Real-Time WebSocket Streaming**
- Polygon.io WebSocket for real-time quotes, trades, and bars
- Alpaca WebSocket for SIP data feed
- Automatic reconnection and failover
- Smart buffering with configurable size

### 4. **Intelligent Priority Queuing**
- CRITICAL: Immediate action required
- HIGH: Process within seconds
- MEDIUM: Process within minutes
- LOW: Background processing
- ARCHIVE: Historical data

## Configuration

### Environment Variables

```bash
# Multi-frequency polling intervals (seconds)
DATA_POLLING_URGENT=30           # Breaking news, major events
DATA_POLLING_FAST=120            # Fast market updates (2 min)
DATA_POLLING_STANDARD=900        # Standard updates (15 min)
DATA_POLLING_BACKGROUND=3600     # Background analysis (1 hour)

# News polling configuration
NEWS_POLLING_BREAKING=120        # Breaking news check (2 min)
NEWS_POLLING_STANDARD=900        # Regular news (15 min)
NEWS_MAX_REQUESTS_PER_DAY=900    # Stay within NewsAPI free tier

# Market data streaming
MARKET_DATA_STREAMING=true       # Enable real-time streaming
MARKET_DATA_WEBSOCKET_ENABLED=true
POLYGON_WEBSOCKET_ENABLED=true   # Enable Polygon WebSocket

# Data velocity thresholds
PRICE_CHANGE_URGENT_THRESHOLD=2.0   # % change to trigger urgent processing
VOLUME_SURGE_THRESHOLD=3.0          # Volume multiplier for urgent
NEWS_SENTIMENT_URGENT_SCORE=0.8     # Sentiment score for urgent

# Strategy routing
ENABLE_STRATEGY_ROUTING=true        # Enable intelligent data routing
FEATURE_DAY_TRADING_ENABLED=true    # Enable day trading strategies
FEATURE_MULTI_STRATEGY_SUPPORT=true # Enable multi-strategy support
```

## Architecture Components

### 1. Data Velocity Manager (`data_velocity_manager.py`)
Orchestrates multi-frequency data collection based on market conditions.

**Key Features:**
- Dynamic interval adjustment during market hours
- API quota management with automatic tracking
- Urgent event detection and prioritization
- Strategy-specific optimizations

**Usage:**
```python
from data_velocity_manager import DataVelocityManager, DataTier, TradingStrategy

manager = DataVelocityManager()
manager.register_data_source('polygon', DataTier.URGENT, fetch_polygon_data)
manager.set_active_strategies([TradingStrategy.DAY_TRADING])
await manager.start()
```

### 2. Real-Time Market Stream (`realtime_market_stream.py`)
Manages WebSocket connections for ultra-low latency market data.

**Key Features:**
- Multi-provider WebSocket support (Polygon, Alpaca)
- High-performance data buffering
- Volume surge and spread monitoring
- Automatic reconnection logic

**Usage:**
```python
from realtime_market_stream import RealTimeMarketStream

stream = RealTimeMarketStream()
stream.register_handler('trade', my_trade_handler)
await stream.start(['AAPL', 'GOOGL', 'TSLA'])
```

### 3. Strategy Data Router (`strategy_data_router.py`)
Routes data intelligently based on trading strategies and priorities.

**Key Features:**
- Strategy-specific data requirements
- Signal generation for multiple strategies
- Priority-based queue processing
- Real-time performance tracking

**Supported Strategies:**
- **Scalping**: 1-second data, tight spreads, order flow
- **Day Trading**: 30-second data, technical indicators
- **Swing Trading**: 2-minute data, sentiment analysis
- **Position Trading**: 15-minute data, fundamentals

**Usage:**
```python
from strategy_data_router import StrategyDataRouter, DataPacket

router = StrategyDataRouter()
router.activate_strategy('day_trading')
await router.route_data(packet)
```

### 4. Monitoring Dashboard (`monitoring_dashboard.py`)
Real-time monitoring of system health and performance.

**Key Features:**
- Web-based dashboard (port 8080)
- Prometheus metrics export
- Alert generation and tracking
- API health monitoring
- Resource usage tracking

**Access:**
- Dashboard: http://localhost:8080/dashboard
- Metrics: http://localhost:8080/metrics
- Health: http://localhost:8080/health

## Trading Strategy Support

### Day Trading
- **Data Requirements**: 30-second to 2-minute updates
- **Indicators**: RSI, MACD, VWAP, Volume
- **Risk**: 2% per trade
- **Enabled**: When `FEATURE_DAY_TRADING_ENABLED=true`

### Scalping
- **Data Requirements**: Real-time WebSocket streaming
- **Indicators**: Spread, Order flow, Volume profile
- **Risk**: 0.1% per trade
- **Enabled**: Automatically with day trading

### Swing Trading
- **Data Requirements**: 2-15 minute updates
- **Indicators**: SMA, Bollinger Bands, ADX
- **Risk**: 5% per trade
- **Enabled**: Always available

### Position Trading
- **Data Requirements**: 15 minute to hourly updates
- **Indicators**: SMA 50/200, Support/Resistance
- **Risk**: 10% per trade
- **Enabled**: Always available

## API Usage & Costs

### Current Configuration (Optimized)
- **Polygon.io**: $99/month (Primary - includes WebSocket)
- **NewsAPI**: $0 (Free tier - 1000 requests/day)
- **IEX Cloud**: $9/month (Backup)
- **Total**: $108/month

### Data Quotas
With multi-frequency polling:
- **NewsAPI**: ~900 requests/day (within 1000 limit)
- **Polygon**: Unlimited with subscription
- **Reddit**: 60 requests/minute (free)

## Performance Metrics

### Latency Targets
- **WebSocket**: < 10ms
- **Urgent**: < 100ms
- **Fast**: < 500ms
- **Standard**: < 2000ms

### Throughput
- **Real-time streams**: 10,000+ messages/second
- **Priority routing**: 1,000+ packets/second
- **Signal generation**: 100+ signals/minute

## Testing

Run the comprehensive test suite:
```bash
cd /home/nilante/main-nilante-server/ai-trading-system
python -m pytest tests/test_multi_frequency_system.py -v
```

## Deployment

1. **Set environment variables** in `.env.production`
2. **Start monitoring dashboard**:
   ```bash
   python services/data-ingestion/monitoring_dashboard.py
   ```
3. **Initialize data services**:
   ```bash
   python services/data-ingestion/main.py
   ```
4. **Access dashboard**: http://your-server:8080/dashboard

## Monitoring & Alerts

### Key Metrics to Watch
- **API Quota Usage**: Stay under limits
- **Data Latency**: Should match tier expectations
- **Error Rate**: Should be < 5%
- **Signal Quality**: Confidence scores > 0.6

### Alert Types
- **Critical**: API failures, system crashes
- **Warning**: High latency, quota warnings
- **Info**: Strategy changes, market state updates

## System Requirements

### Minimum
- **CPU**: 4 cores
- **RAM**: 8GB
- **Network**: 100Mbps
- **Storage**: 100GB SSD

### Recommended (Your Server)
- **CPU**: 64 cores ✅
- **RAM**: 988GB ✅
- **Network**: 1Gbps ✅
- **Storage**: 3.6TB NVMe + 15TB HDD ✅

## Conclusion

The enhanced system now provides:

1. **Full Strategy Support**: From scalping to position trading
2. **Intelligent Data Routing**: Based on urgency and strategy needs
3. **Real-Time Capabilities**: WebSocket streaming for instant data
4. **Cost Optimization**: $108/month with NewsAPI free tier
5. **Comprehensive Monitoring**: Web dashboard and Prometheus metrics
6. **Production Ready**: Tested, monitored, and scalable

The system is **robust enough for any trading strategy** and will only execute trades when the data and analysis support the decision. It dynamically adjusts data collection frequencies based on market conditions and active strategies, ensuring optimal performance while staying within API limits.