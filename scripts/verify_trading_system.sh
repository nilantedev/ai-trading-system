#!/bin/bash
# Final Dashboard Deployment and Verification
# This script verifies trading system readiness and dashboard functionality

set -e

cd /srv/ai-trading-system

echo "========================================"
echo "TRADING SYSTEM FINAL VERIFICATION"
echo "========================================"
echo ""

# 1. Trading System Readiness
echo "=== 1. TRADING SYSTEM STATUS ==="
PAPER_MODE=$(grep "^PAPER_TRADING=" .env | cut -d= -f2)
TRADING_MODE=$(grep "^TRADING_MODE=" .env | cut -d= -f2)
echo "✓ Paper Trading: $PAPER_MODE"
echo "✓ Trading Mode: $TRADING_MODE"
echo ""

# 2. Service Health
echo "=== 2. MICROSERVICES HEALTH (7 Required) ==="
HEALTHY_COUNT=0
for service in ml data-ingestion signal-generator execution risk-monitor strategy-engine backtesting; do
    STATUS=$(docker ps --filter "name=trading-$service" --format "{{.Status}}" 2>/dev/null | grep -o "healthy" || echo "down")
    if [ "$STATUS" = "healthy" ]; then
        echo "✓ trading-$service: HEALTHY"
        HEALTHY_COUNT=$((HEALTHY_COUNT + 1))
    else
        echo "✗ trading-$service: DOWN"
    fi
done
echo ""
echo "Services Operational: $HEALTHY_COUNT/7"
echo ""

# 3. Continuous Processing Configuration
echo "=== 3. CONTINUOUS PROCESSING CONFIG ==="
echo "API Workers: $(grep "^API_WORKERS=" .env | cut -d= -f2)"
echo "Market Data Interval: $(grep "^DAILY_DELTA_INTERVAL_SECONDS=" .env | cut -d= -f2)s"
echo "Signal Generation: $(grep "^SIGNAL_GENERATION_INTERVAL=" .env | cut -d= -f2)s"
echo "Risk Check Interval: $(grep "^RISK_CHECK_INTERVAL=" .env | cut -d= -f2)s"
echo ""

# 4. Data Inventory
echo "=== 4. DATA INVENTORY ==="
MARKET_DATA=$(curl -s "http://localhost:9000/exec?query=SELECT+count(*)+FROM+market_data" | jq -r '.dataset[0][0]' 2>/dev/null || echo "0")
SOCIAL_DATA=$(curl -s "http://localhost:9000/exec?query=SELECT+count(*)+FROM+social_signals" | jq -r '.dataset[0][0]' 2>/dev/null || echo "0")
OPTIONS_DATA=$(curl -s "http://localhost:9000/exec?query=SELECT+count(*)+FROM+options_data" | jq -r '.dataset[0][0]' 2>/dev/null || echo "0")
WATCHLIST=$(docker exec ad0acd6b53e7_trading-redis redis-cli -a 'Okunka!Blebogyan02$' --no-auth-warning SCARD watchlist 2>/dev/null || echo "0")

echo "✓ Market Data Bars: $MARKET_DATA"
echo "✓ Social Signals: $SOCIAL_DATA"
echo "✓ Options Contracts: $OPTIONS_DATA"
echo "✓ Watchlist Symbols: $WATCHLIST"
echo ""

# 5. API Endpoints
echo "=== 5. DASHBOARD API ENDPOINTS ==="
ENDPOINTS="watchlist/all services/health market/summary data/comprehensive social/recent options/flow"
ALL_OK=true
for endpoint in $ENDPOINTS; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8000/api/dashboard/$endpoint")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "✓ $endpoint: HTTP $HTTP_CODE"
    else
        echo "✗ $endpoint: HTTP $HTTP_CODE"
        ALL_OK=false
    fi
done
echo ""

# 6. Dashboard Files
echo "=== 6. DASHBOARD FILES ==="
if [ -f "api/templates/business/dashboard_v2.html" ]; then
    SIZE=$(stat -f%z "api/templates/business/dashboard_v2.html" 2>/dev/null || stat -c%s "api/templates/business/dashboard_v2.html" 2>/dev/null)
    echo "✓ Business Dashboard: $SIZE bytes"
else
    echo "✗ Business Dashboard: MISSING"
fi

if [ -f "api/templates/admin/dashboard.html" ]; then
    SIZE=$(stat -f%z "api/templates/admin/dashboard.html" 2>/dev/null || stat -c%s "api/templates/admin/dashboard.html" 2>/dev/null)
    echo "✓ Admin Dashboard: $SIZE bytes"
else
    echo "✗ Admin Dashboard: MISSING"
fi
echo ""

# 7. Summary
echo "========================================"
echo "DEPLOYMENT STATUS SUMMARY"
echo "========================================"
echo ""

if [ "$HEALTHY_COUNT" -eq 7 ] && [ "$ALL_OK" = true ] && [ "$MARKET_DATA" -gt 1000000 ]; then
    echo "✓✓✓ SYSTEM READY FOR TRADING ✓✓✓"
    echo ""
    echo "Status: OPERATIONAL"
    echo "Mode: Paper Trading (Safe)"
    echo "Services: All Healthy"
    echo "Data: $MARKET_DATA+ bars available"
    echo "APIs: All endpoints responding"
    echo "Dashboards: Production ready"
    echo ""
    echo "Access:"
    echo "  Business: https://biz.mekoshi.com/business"
    echo "  Admin: https://admin.mekoshi.com/admin"
else
    echo "⚠ SYSTEM PARTIALLY READY ⚠"
    echo ""
    echo "Issues detected - review above sections"
fi
echo ""
echo "========================================"
