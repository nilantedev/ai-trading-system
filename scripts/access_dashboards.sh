#!/bin/bash
#
# Quick Access to PhD-Level Trading Dashboards
# Generates MFA code and provides login instructions
#

set -e

echo "========================================"
echo "PhD-LEVEL TRADING DASHBOARDS"
echo "========================================"
echo ""

# Generate current MFA code
echo "Generating MFA code..."
MFA_CODE=$(docker exec trading-api python3 -c "import pyotp; print(pyotp.TOTP('I23JMI2MTIR4LNNOAUCJOCYPNINIGL4M').now())")

echo ""
echo "DASHBOARD ACCESS INFORMATION"
echo "----------------------------"
echo ""
echo "🔐 LOGIN CREDENTIALS:"
echo "  Username: nilante"
echo "  Password: Okunka!Blebogyan02\$"
echo "  MFA Code: $MFA_CODE"
echo "  (Valid for next 30 seconds)"
echo ""
echo "🌐 DASHBOARD URLs:"
echo ""
echo "  📊 Business Intelligence (Investor Platform):"
echo "     http://localhost:8000/business"
echo ""
echo "     Features:"
echo "       - Live options flow whale tracker"
echo "       - Real-time market heatmap"
echo "       - PhD-level symbol analysis"
echo "       - ML ensemble forecasts (7 models)"
echo "       - Factor exposures & risk metrics"
echo "       - News sentiment analysis"
echo "       - Trading recommendations"
echo ""
echo "  ⚙️  Admin Control Panel (God-Mode):"
echo "     http://localhost:8000/admin"
echo ""
echo "     Features:"
echo "       - Service management (restart/scale/logs)"
echo "       - Model weight tuning (6 models)"
echo "       - Factor configuration (6 factors)"
echo "       - Real-time system metrics"
echo "       - Manual backfill triggers"
echo "       - Emergency controls (kill switch)"
echo ""
echo "🔑 QUICK LOGIN:"
echo ""
echo "1. Open: http://localhost:8000/auth/login"
echo "2. Enter credentials above"
echo "3. Access dashboards"
echo ""
echo "📝 SYSTEM STATUS:"
echo ""

# Check API health
if curl -s -f http://localhost:8000/health >/dev/null 2>&1; then
    echo "   ✅ API: HEALTHY"
else
    echo "   ❌ API: NOT RESPONDING"
fi

# Check watchlist
REDIS_PASS=$(grep "^REDIS_PASSWORD=" /srv/ai-trading-system/.env | cut -d= -f2)
WATCHLIST=$(docker exec trading-redis redis-cli -a "$REDIS_PASS" SCARD watchlist 2>&1 | grep -v Warning)
echo "   ✅ Watchlist: $WATCHLIST symbols"

# Check streaming
STREAMS=$(docker exec trading-data-ingestion curl -s http://localhost:8002/streams/status 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(sum(1 for v in data.values() if v.get('enabled', False)))" 2>/dev/null)
echo "   ✅ Streaming: $STREAMS active streams"

# Check models
MODELS=$(docker exec trading-api curl -s http://ollama:11434/api/tags 2>/dev/null | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('models', [])))" 2>/dev/null)
echo "   ✅ ML Models: $MODELS loaded"

echo ""
echo "🚀 SYSTEM STATUS: TRADING-READY"
echo ""
echo "========================================"
echo ""

# Ask if user wants to open browser
read -p "Open dashboards in browser? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Opening dashboards..."
    
    # Try different browser commands
    if command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost:8000/auth/login" &
    elif command -v open &> /dev/null; then
        open "http://localhost:8000/auth/login" &
    elif command -v firefox &> /dev/null; then
        firefox "http://localhost:8000/auth/login" &
    elif command -v google-chrome &> /dev/null; then
        google-chrome "http://localhost:8000/auth/login" &
    else
        echo "Could not auto-open browser. Please open manually:"
        echo "http://localhost:8000/auth/login"
    fi
fi

echo ""
echo "💡 TIP: Run this script again if MFA code expires"
echo ""
