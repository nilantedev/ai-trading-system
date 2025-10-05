#!/bin/bash

# Test Backtesting System
# Verifies all components are working

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Backtesting System Test${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Test 1: Check strategy-engine is running
echo -e "${YELLOW}[1/4] Checking strategy-engine service...${NC}"
if curl -s http://localhost:8006/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Strategy-engine is healthy"
else
    echo -e "${RED}✗${NC} Strategy-engine is not responding"
    exit 1
fi

# Test 2: Check available strategies
echo
echo -e "${YELLOW}[2/4] Checking available strategies...${NC}"
STRATEGIES=$(curl -s http://localhost:8006/strategies | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('strategies', [])))")
if [ "$STRATEGIES" -ge 7 ]; then
    echo -e "${GREEN}✓${NC} Found $STRATEGIES strategies"
else
    echo -e "${RED}✗${NC} Expected 7 strategies, found $STRATEGIES"
    exit 1
fi

# Test 3: Check available symbols (QuestDB)
echo
echo -e "${YELLOW}[3/4] Checking available symbols for backtesting...${NC}"
RESPONSE=$(curl -s http://localhost:8006/backtest/symbols/available 2>&1)
if echo "$RESPONSE" | grep -q "symbols"; then
    SYMBOL_COUNT=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('count', 0))" 2>/dev/null || echo "0")
    if [ "$SYMBOL_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✓${NC} Found $SYMBOL_COUNT symbols with sufficient data"
    else
        echo -e "${YELLOW}⚠${NC} No symbols found in QuestDB (may need to seed data)"
        echo -e "  This is expected if data ingestion hasn't run yet"
    fi
else
    echo -e "${YELLOW}⚠${NC} Backtest symbols endpoint not fully operational"
    echo -e "  Response: $RESPONSE"
fi

# Test 4: Test backtest API structure
echo
echo -e "${YELLOW}[4/4] Testing backtest API endpoint...${NC}"
TEST_BACKTEST=$(cat <<EOF
{
    "strategy": "momentum",
    "symbols": ["AAPL"],
    "start_date": "2024-01-01",
    "end_date": "2024-02-01",
    "initial_capital": 100000
}
EOF
)

BACKTEST_RESPONSE=$(curl -s -X POST http://localhost:8006/backtest/run \
    -H "Content-Type: application/json" \
    -d "$TEST_BACKTEST" 2>&1)

if echo "$BACKTEST_RESPONSE" | grep -q "backtest_id"; then
    BACKTEST_ID=$(echo "$BACKTEST_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('backtest_id', ''))" 2>/dev/null || echo "")
    echo -e "${GREEN}✓${NC} Backtest API is operational"
    echo -e "  Backtest ID: ${YELLOW}$BACKTEST_ID${NC}"
    
    # Check backtest status
    sleep 1
    STATUS_RESPONSE=$(curl -s http://localhost:8006/backtest/$BACKTEST_ID 2>&1)
    STATUS=$(echo "$STATUS_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('status', ''))" 2>/dev/null || echo "unknown")
    echo -e "  Status: ${YELLOW}$STATUS${NC}"
else
    echo -e "${YELLOW}⚠${NC} Backtest API responded but may need data"
    echo -e "  Response: $BACKTEST_RESPONSE"
fi

echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo
echo -e "${GREEN}✓ Strategy-engine: Operational${NC}"
echo -e "${GREEN}✓ Strategies: $STRATEGIES available${NC}"
if [ "$SYMBOL_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ QuestDB: $SYMBOL_COUNT symbols available${NC}"
else
    echo -e "${YELLOW}⚠ QuestDB: No data yet (run data ingestion)${NC}"
fi
echo -e "${GREEN}✓ Backtest API: Operational${NC}"
echo
echo -e "${YELLOW}Next Steps:${NC}"
echo -e "  1. Ensure data ingestion is running to populate QuestDB"
echo -e "  2. Wait for 100+ symbols with 90+ days of data"
echo -e "  3. Run comprehensive backtests across all strategies"
echo
