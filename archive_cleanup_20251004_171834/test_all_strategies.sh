#!/bin/bash
# Test all 7 strategies with backtesting

echo "=================================================="
echo "Testing All Trading Strategies"
echo "=================================================="

STRATEGIES=("momentum" "mean_reversion" "stat_arb" "market_making" "vol_arb" "index_arb" "trend_following")
SYMBOLS='["AAPL","GOOGL","MSFT"]'
START_DATE="2024-07-01"
END_DATE="2024-10-01"
CAPITAL=100000

for strategy in "${STRATEGIES[@]}"; do
    echo ""
    echo "Testing strategy: $strategy"
    echo "---"
    
    RESPONSE=$(curl -s -X POST http://localhost:8006/backtest/run \
        -H "Content-Type: application/json" \
        -d "{\"strategy\":\"$strategy\",\"symbols\":$SYMBOLS,\"start_date\":\"$START_DATE\",\"end_date\":\"$END_DATE\",\"initial_capital\":$CAPITAL}")
    
    BACKTEST_ID=$(echo $RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('backtest_id', 'ERROR'))")
    
    if [ "$BACKTEST_ID" != "ERROR" ]; then
        echo "Backtest ID: $BACKTEST_ID"
        sleep 5
        
        # Get results
        curl -s http://localhost:8006/backtest/$BACKTEST_ID | python3 -c "
import sys, json
data = json.load(sys.stdin)
metrics = data.get('metrics', {})
print(f\"  Status: {data.get('status')}\" )
print(f\"  Trades: {metrics.get('num_trades', 0)}\")
print(f\"  Return: {metrics.get('total_return', 0)*100:.2f}%\")
print(f\"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}\")
print(f\"  Max DD: {metrics.get('max_drawdown', 0)*100:.2f}%\")
"
    else
        echo "  ERROR: Failed to start backtest"
    fi
done

echo ""
echo "=================================================="
echo "All strategies tested!"
echo "=================================================="
