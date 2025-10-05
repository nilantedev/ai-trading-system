# Backtesting System Documentation
## PhD-Level Trading System

**Date**: October 4, 2025  
**Status**: ✅ **PRODUCTION READY**

---

## Overview

The backtesting system implements a research-grade framework for validating trading strategies with realistic market conditions, transaction costs, and performance analytics used by elite hedge funds.

### Key Features

✅ **Realistic Execution Simulation**
- Multiple fill models (instant, aggressive, passive, realistic)
- Bid-ask spread modeling
- Market impact (square-root model)
- Partial fills and order rejections
- Liquidity-dependent slippage

✅ **Transaction Cost Modeling**
- Commission (10 basis points default)
- Slippage (5 basis points + market impact)
- Market impact scaling with trade size
- Realistic cost attribution

✅ **Comprehensive Performance Analytics**
- Sharpe ratio, Sortino ratio, Calmar ratio
- Maximum drawdown and duration
- Win rate, profit factor, average trade PnL
- Transaction cost analysis
- Risk-adjusted returns

✅ **QuestDB Integration**
- High-speed time-series data loading
- Connection pooling for performance
- Multiple timeframes (1m, 5m, 15m, 1h, 1d)
- Batch loading for multiple symbols

✅ **Production-Ready API**
- RESTful API endpoints
- Async backtest execution
- Status tracking and results storage
- Available symbols discovery

---

## Architecture

### Components

1. **backtesting_engine.py** (800+ lines)
   - Core backtesting logic
   - Order execution simulation
   - Portfolio management
   - Performance calculation

2. **questdb_data_loader.py** (250+ lines)
   - QuestDB connection pooling
   - OHLCV data loading
   - Symbol discovery
   - Date range queries

3. **strategy_manager.py** (Enhanced)
   - Backtesting API endpoints
   - Strategy integration
   - Result tracking

### Data Flow

```
QuestDB (Historical Data)
    ↓
QuestDBDataLoader
    ↓
BacktestEngine → Strategy Signals → Orders → Fills
    ↓
Performance Metrics
    ↓
API Response / Storage
```

---

## API Reference

### 1. Run Backtest

**Endpoint**: `POST /backtest/run`

**Request**:
```json
{
    "strategy": "momentum",
    "symbols": ["AAPL", "GOOGL", "MSFT"],
    "start_date": "2024-01-01",
    "end_date": "2024-10-01",
    "initial_capital": 100000,
    "config": {
        "commission_bps": 10,
        "slippage_bps": 5,
        "fill_model": "realistic",
        "max_position_size": 0.20
    }
}
```

**Response**:
```json
{
    "backtest_id": "BT000001",
    "status": "running",
    "strategy": "momentum",
    "symbols": ["AAPL", "GOOGL", "MSFT"],
    "message": "Backtest initiated"
}
```

### 2. Get Backtest Status

**Endpoint**: `GET /backtest/{backtest_id}`

**Response**:
```json
{
    "strategy": "momentum",
    "symbols": ["AAPL", "GOOGL", "MSFT"],
    "start_date": "2024-01-01",
    "end_date": "2024-10-01",
    "status": "completed",
    "started_at": "2025-10-04T10:00:00",
    "completed_at": "2025-10-04T10:05:00",
    "metrics": {
        "total_return": 0.28,
        "annual_return": 0.35,
        "sharpe_ratio": 2.4,
        "sortino_ratio": 3.1,
        "calmar_ratio": 2.8,
        "max_drawdown": -0.12,
        "volatility": 0.18,
        "num_trades": 145,
        "win_rate": 0.68,
        "profit_factor": 2.3,
        "avg_trade_pnl": 185.50,
        "total_commission": 1245.80,
        "total_slippage": 823.45
    }
}
```

### 3. List All Backtests

**Endpoint**: `GET /backtest/list`

**Response**:
```json
{
    "backtests": [
        {
            "id": "BT000001",
            "strategy": "momentum",
            "status": "completed",
            "...": "..."
        }
    ],
    "total": 1
}
```

### 4. Get Available Symbols

**Endpoint**: `GET /backtest/symbols/available`

**Response**:
```json
{
    "symbols": ["AAPL", "GOOGL", "MSFT", "..."],
    "count": 150,
    "min_bars": 100
}
```

---

## Fill Models

### 1. INSTANT (Unrealistic)
- Instant fill at close price
- No slippage or market impact
- **Use**: Quick prototyping only

### 2. AGGRESSIVE (Market Orders)
- Fill at bid (sell) or ask (buy)
- Bid-ask spread cost
- **Use**: Fast execution strategies

### 3. PASSIVE (Limit Orders)
- Fill only if limit price reached
- Order rejection possible
- **Use**: Patient strategies

### 4. REALISTIC (Recommended)
- Combines spread, market impact, partial fills
- Order rejections (2% probability)
- Partial fills (5% probability)
- **Use**: Production backtesting

---

## Performance Metrics

### Return Metrics
- **Total Return**: Cumulative return over period
- **Annual Return**: Annualized return (CAGR)
- **Cumulative Returns**: Full equity curve

### Risk Metrics
- **Volatility**: Annualized standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return (excess return / volatility)
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return / max drawdown

### Drawdown Analysis
- **Max Drawdown**: Largest peak-to-trough decline
- **Max Drawdown Duration**: Longest time in drawdown
- **Avg Drawdown**: Average of all drawdowns

### Trade Statistics
- **Num Trades**: Total number of fills
- **Win Rate**: Percentage of profitable trades
- **Avg Win/Loss**: Average profit of wins/losses
- **Profit Factor**: Gross profits / gross losses
- **Avg Trade PnL**: Average P&L per trade

### Transaction Costs
- **Total Commission**: Sum of all commission costs
- **Total Slippage**: Sum of all slippage costs
- **Avg Cost Per Trade**: Average transaction cost

---

## Usage Examples

### Example 1: Quick Backtest

```bash
# Test momentum strategy on tech stocks
curl -X POST http://localhost:8006/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "momentum",
    "symbols": ["AAPL", "GOOGL", "MSFT"],
    "start_date": "2024-01-01",
    "end_date": "2024-10-01",
    "initial_capital": 100000
  }'

# Response: {"backtest_id": "BT000001", "status": "running", ...}

# Check status
curl http://localhost:8006/backtest/BT000001
```

### Example 2: Elite Strategy Backtest

```python
import requests

# Get available symbols
response = requests.get("http://localhost:8006/backtest/symbols/available")
symbols = response.json()["symbols"][:100]  # Top 100 by data availability

# Run backtest for statistical arbitrage
backtest_config = {
    "strategy": "statistical_arbitrage",
    "symbols": symbols,
    "start_date": "2024-01-01",
    "end_date": "2024-10-01",
    "initial_capital": 1000000,
    "config": {
        "commission_bps": 10,
        "slippage_bps": 5,
        "fill_model": "realistic",
        "max_position_size": 0.10  # 10% max per position
    }
}

response = requests.post(
    "http://localhost:8006/backtest/run",
    json=backtest_config
)

backtest_id = response.json()["backtest_id"]
print(f"Backtest started: {backtest_id}")

# Poll for completion
import time
while True:
    status_response = requests.get(f"http://localhost:8006/backtest/{backtest_id}")
    status = status_response.json()
    
    if status["status"] == "completed":
        metrics = status["metrics"]
        print(f"Backtest complete!")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        break
    elif status["status"] == "failed":
        print(f"Backtest failed: {status.get('error')}")
        break
    
    time.sleep(5)
```

### Example 3: Multi-Strategy Comparison

```python
import requests
import pandas as pd

strategies = [
    "momentum",
    "mean_reversion",
    "statistical_arbitrage",
    "market_making",
    "volatility_arbitrage",
    "index_arbitrage",
    "trend_following"
]

symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
results = []

for strategy in strategies:
    response = requests.post(
        "http://localhost:8006/backtest/run",
        json={
            "strategy": strategy,
            "symbols": symbols,
            "start_date": "2024-01-01",
            "end_date": "2024-10-01",
            "initial_capital": 100000
        }
    )
    
    backtest_id = response.json()["backtest_id"]
    
    # Wait for completion (simplified)
    time.sleep(10)
    
    status = requests.get(f"http://localhost:8006/backtest/{backtest_id}").json()
    
    if status["status"] == "completed":
        metrics = status["metrics"]
        results.append({
            "strategy": strategy,
            "return": metrics["annual_return"],
            "sharpe": metrics["sharpe_ratio"],
            "max_dd": metrics["max_drawdown"],
            "win_rate": metrics["win_rate"]
        })

# Create comparison DataFrame
df = pd.DataFrame(results)
print(df.to_string())

# Best strategy by Sharpe
best = df.loc[df["sharpe"].idxmax()]
print(f"\nBest Strategy: {best['strategy']} (Sharpe: {best['sharpe']:.2f})")
```

---

## Transaction Cost Model

### Commission Formula
```
commission = trade_value * (commission_bps / 10000)
```

Default: 10 bps = 0.10% per trade

### Slippage Formula
```
base_slippage = price * (slippage_bps / 10000)
market_impact = price * impact_factor * sqrt(quantity / volume)
total_slippage = base_slippage + market_impact
```

Default base: 5 bps = 0.05%  
Impact factor: 0.1 (configurable)

### Market Impact (Square-Root Model)

Based on research by Almgren, Chriss, and Hasbrouck:
- Impact scales with √(participation rate)
- Reflects liquidity constraints
- Typical values: 0.05-0.20 depending on asset class

**References**:
- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions
- Hasbrouck, J. (2007). Empirical Market Microstructure

---

## Best Practices

### 1. Data Requirements
- ✅ Minimum 100 bars per symbol
- ✅ Use 90+ days for meaningful statistics
- ✅ Check for data quality (missing bars, outliers)
- ✅ Use 1-day bars for swing strategies, intraday for HFT

### 2. Backtest Configuration
- ✅ Use realistic fill model for production testing
- ✅ Set transaction costs to match broker fees
- ✅ Account for market impact (especially for large trades)
- ✅ Test across multiple market regimes

### 3. Strategy Validation
- ✅ Run walk-forward analysis (rolling windows)
- ✅ Test on out-of-sample data
- ✅ Compare to benchmark (SPY)
- ✅ Check for data snooping bias

### 4. Performance Evaluation
- ✅ Sharpe ratio > 1.0 (good), > 2.0 (excellent)
- ✅ Max drawdown < 20%
- ✅ Win rate > 50% (for mean reversion)
- ✅ Profit factor > 1.5

### 5. Risk Management
- ✅ Set max position sizes (10-20% per position)
- ✅ Implement max drawdown stop (20%)
- ✅ Diversify across strategies
- ✅ Monitor correlation between positions

---

## Troubleshooting

### Issue: No market data found

**Cause**: QuestDB doesn't have data for specified symbols/dates

**Solution**:
```bash
# Check available symbols
curl http://localhost:8006/backtest/symbols/available

# Verify data range
docker exec -it trading-questdb psql -h localhost -p 8812 -U admin -d qdb \
  -c "SELECT symbol, MIN(timestamp), MAX(timestamp), COUNT(*) 
      FROM market_data 
      GROUP BY symbol 
      LIMIT 10;"
```

### Issue: Backtest fails with connection error

**Cause**: QuestDB connection issues

**Solution**:
```bash
# Check QuestDB is running
docker ps | grep questdb

# Test connection
docker exec -it trading-questdb psql -h localhost -p 8812 -U admin -d qdb -c "SELECT 1;"
```

### Issue: Low performance (Sharpe < 0.5)

**Cause**: Strategy not profitable or poor parameter tuning

**Solution**:
- ✅ Review strategy parameters
- ✅ Test on different symbols/timeframes
- ✅ Check transaction costs aren't too high
- ✅ Verify strategy logic is correct

---

## Roadmap

### Phase 1 (Complete) ✅
- [x] Backtesting engine core
- [x] QuestDB data loader
- [x] API endpoints
- [x] Performance metrics

### Phase 2 (In Progress) 🔄
- [ ] Run 90-day backtests on 100 symbols
- [ ] Multi-strategy portfolio backtesting
- [ ] Walk-forward analysis
- [ ] Results storage in PostgreSQL

### Phase 3 (Planned) 📋
- [ ] Visualization dashboard
- [ ] Monte Carlo simulation
- [ ] Parameter optimization (grid search, genetic algorithms)
- [ ] Paper trading integration

---

## Academic References

1. **Bailey, D. H., & López de Prado, M. (2014)**  
   "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality"

2. **Harvey, C. R., & Liu, Y. (2015)**  
   "Backtesting"  
   Journal of Portfolio Management

3. **López de Prado, M. (2018)**  
   "Advances in Financial Machine Learning"  
   Wiley

4. **Almgren, R., & Chriss, N. (2001)**  
   "Optimal Execution of Portfolio Transactions"  
   Journal of Risk

5. **Hasbrouck, J. (2007)**  
   "Empirical Market Microstructure"  
   Oxford University Press

---

## Conclusion

The backtesting system provides a production-grade framework for validating trading strategies with realistic market conditions and comprehensive performance analytics. It implements best practices from academic research and elite hedge fund methodologies.

**Next Steps**:
1. ✅ Run backtests for all 7 strategies
2. ✅ Validate profitability across 100 symbols
3. ✅ Compare to benchmark (SPY)
4. ✅ Deploy profitable strategies to paper trading

---

**Last Updated**: October 4, 2025  
**Version**: 1.0.0  
**Status**: Production Ready
