#!/bin/bash

# Comprehensive Live Trading System Verification
# Checks: Data Processing, Backfills, Intelligence, Trading Services
# Date: October 2, 2025

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     LIVE TRADING SYSTEM COMPREHENSIVE VERIFICATION              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

PASS=0
FAIL=0
WARN=0

check_pass() {
    echo -e "  ${GREEN}✓${NC} $1"
    PASS=$((PASS + 1))
}

check_fail() {
    echo -e "  ${RED}✗${NC} $1"
    FAIL=$((FAIL + 1))
}

check_warn() {
    echo -e "  ${YELLOW}⚠${NC} $1"
    WARN=$((WARN + 1))
}

# ========================================
# 1. DATA POPULATION CHECK
# ========================================
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${BLUE}1. DATA POPULATION STATUS${NC}"
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# QuestDB (Time-Series)
MARKET_COUNT=$(curl -s "http://localhost:9000/exec?query=SELECT%20count()%20FROM%20market_data" 2>/dev/null | grep -o 'dataset":\[\[[0-9]*\]\]' | grep -o '[0-9]*' || echo "0")
OPTIONS_COUNT=$(curl -s "http://localhost:9000/exec?query=SELECT%20count()%20FROM%20options_data" 2>/dev/null | grep -o 'dataset":\[\[[0-9]*\]\]' | grep -o '[0-9]*' || echo "0")
# News is in PostgreSQL, not QuestDB
NEWS_COUNT=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM news_events;" 2>/dev/null | tr -d ' ' || echo "0")
SOCIAL_COUNT=$(curl -s "http://localhost:9000/exec?query=SELECT%20count()%20FROM%20social_signals" 2>/dev/null | grep -o 'dataset":\[\[[0-9]*\]\]' | grep -o '[0-9]*' || echo "0")

echo "  QuestDB (Time-Series Database):"
if [ "$MARKET_COUNT" -gt 1000000 ]; then
    check_pass "Market Data: $(printf "%'d" $MARKET_COUNT) bars (EXCELLENT)"
elif [ "$MARKET_COUNT" -gt 100000 ]; then
    check_warn "Market Data: $(printf "%'d" $MARKET_COUNT) bars (GOOD, needs more)"
else
    check_fail "Market Data: $(printf "%'d" $MARKET_COUNT) bars (INSUFFICIENT)"
fi

if [ "$OPTIONS_COUNT" -gt 100000 ]; then
    check_pass "Options Data: $(printf "%'d" $OPTIONS_COUNT) bars (EXCELLENT)"
elif [ "$OPTIONS_COUNT" -gt 10000 ]; then
    check_warn "Options Data: $(printf "%'d" $OPTIONS_COUNT) bars (GOOD, needs more)"
else
    check_fail "Options Data: $(printf "%'d" $OPTIONS_COUNT) bars (INSUFFICIENT)"
fi

if [ "$NEWS_COUNT" -gt 10000 ]; then
    check_pass "News Events: $(printf "%'d" $NEWS_COUNT) articles"
elif [ "$NEWS_COUNT" -gt 1000 ]; then
    check_warn "News Events: $(printf "%'d" $NEWS_COUNT) articles (needs more)"
else
    check_fail "News Events: $(printf "%'d" $NEWS_COUNT) articles (INSUFFICIENT)"
fi

if [ "$SOCIAL_COUNT" -gt 10000 ]; then
    check_pass "Social Signals: $(printf "%'d" $SOCIAL_COUNT) signals"
else
    check_warn "Social Signals: $(printf "%'d" $SOCIAL_COUNT) signals"
fi

# Weaviate Vector DB
WEAVIATE_NEWS=$(curl -s -X POST http://localhost:8080/v1/graphql -H "Content-Type: application/json" -d '{"query":"{Aggregate{NewsArticle{meta{count}}}}"}' 2>/dev/null | grep -o '"count":[0-9]*' | grep -o '[0-9]*' || echo "0")
WEAVIATE_SOCIAL=$(curl -s -X POST http://localhost:8080/v1/graphql -H "Content-Type: application/json" -d '{"query":"{Aggregate{SocialSentiment{meta{count}}}}"}' 2>/dev/null | grep -o '"count":[0-9]*' | grep -o '[0-9]*' || echo "0")

echo ""
echo "  Weaviate (Vector Database - AI Embeddings):"
if [ "$WEAVIATE_NEWS" -gt 10000 ]; then
    check_pass "News Embeddings: $(printf "%'d" $WEAVIATE_NEWS) vectors"
else
    check_warn "News Embeddings: $(printf "%'d" $WEAVIATE_NEWS) vectors (indexing in progress)"
fi

if [ "$WEAVIATE_SOCIAL" -gt 5000 ]; then
    check_pass "Social Embeddings: $(printf "%'d" $WEAVIATE_SOCIAL) vectors"
else
    check_warn "Social Embeddings: $(printf "%'d" $WEAVIATE_SOCIAL) vectors (indexing in progress)"
fi

# ========================================
# 2. BACKFILL STATUS
# ========================================
echo ""
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${BLUE}2. HISTORICAL BACKFILL STATUS${NC}"
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

BACKFILL_ENABLED=$(grep "ENABLE_HISTORICAL_BACKFILL" /srv/ai-trading-system/.env 2>/dev/null | cut -d= -f2)
if [ "$BACKFILL_ENABLED" = "true" ]; then
    check_pass "Historical Backfill: ENABLED"
else
    check_warn "Historical Backfill: DISABLED (enable for comprehensive data)"
fi

# Check backfill progress
BACKFILL_SYMBOLS=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM historical_backfill_progress;" 2>/dev/null | tr -d ' ' || echo "0")
if [ "$BACKFILL_SYMBOLS" -gt 50 ]; then
    check_pass "Backfill Progress: $BACKFILL_SYMBOLS symbols tracked"
else
    check_warn "Backfill Progress: $BACKFILL_SYMBOLS symbols (backfill in progress)"
fi

# Check latest backfill activity
LATEST_BACKFILL=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "SELECT last_date FROM historical_backfill_progress ORDER BY last_date DESC LIMIT 1;" 2>/dev/null | tr -d ' ')
if [ -n "$LATEST_BACKFILL" ]; then
    check_pass "Latest Backfill Date: $LATEST_BACKFILL"
else
    check_warn "Latest Backfill: No data yet"
fi

# ========================================
# 3. REAL-TIME DATA STREAMS
# ========================================
echo ""
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${BLUE}3. REAL-TIME DATA STREAMING${NC}"
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

STREAMS=$(curl -s http://localhost:8002/streams/status 2>/dev/null)

# Check key streams
for stream in "quote_stream" "news_stream" "social_stream" "daily_delta" "daily_options"; do
    ENABLED=$(echo "$STREAMS" | grep -o "\"$stream\":{\"enabled\":[^,]*" | grep -o "true\|false")
    LAST_RUN=$(echo "$STREAMS" | grep -o "\"$stream\":{[^}]*\"last_run\":\"[^\"]*\"" | grep -o '"last_run":"[^"]*"' | cut -d'"' -f4)
    
    if [ "$ENABLED" = "true" ]; then
        # Calculate how recent the last run was
        if [ -n "$LAST_RUN" ]; then
            NOW=$(date +%s)
            # Parse ISO timestamp with microseconds
            LAST_CLEAN=$(echo "$LAST_RUN" | sed 's/\.[0-9]*Z*$//')
            LAST=$(date -d "$LAST_CLEAN" +%s 2>/dev/null || echo "0")
            if [ "$LAST" -eq 0 ]; then
                AGE=999999
            else
                AGE=$((NOW - LAST))
            fi
            
            if [ $AGE -lt 300 ]; then  # Less than 5 minutes
                check_pass "$stream: ACTIVE (last run: ${AGE}s ago)"
            elif [ $AGE -lt 3600 ]; then  # Less than 1 hour
                check_warn "$stream: ENABLED (last run: $((AGE/60))m ago)"
            else
                check_warn "$stream: ENABLED (last run: $((AGE/3600))h ago)"
            fi
        else
            check_warn "$stream: ENABLED (no recent activity)"
        fi
    else
        check_fail "$stream: DISABLED"
    fi
done

# ========================================
# 4. INTELLIGENCE SYSTEM (ML)
# ========================================
echo ""
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${BLUE}4. AI/ML INTELLIGENCE SYSTEM${NC}"
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check ML service health
ML_HEALTH=$(curl -s http://localhost:8001/health 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
if [ "$ML_HEALTH" = "healthy" ]; then
    check_pass "ML Service: HEALTHY"
else
    check_fail "ML Service: $ML_HEALTH"
fi

# Check if ML is processing data
ML_METRICS=$(curl -s http://localhost:8001/metrics 2>/dev/null)
INFERENCE_REQUESTS=$(echo "$ML_METRICS" | grep "app_inference_requests_total" | grep -v "^#" | head -1 | awk '{print $2}' || echo "0")
if [ "$(echo "$INFERENCE_REQUESTS > 0" | bc -l 2>/dev/null || echo "0")" -eq 1 ]; then
    check_pass "ML Inference: ACTIVE (processed requests)"
else
    check_warn "ML Inference: No requests yet (waiting for data)"
fi

# Check FinBERT sentiment analysis
FINBERT_STATUS=$(curl -s --max-time 5 http://localhost:8001/startup/status 2>/dev/null | grep -o '"finbert":[^,}]*' || echo "")
if echo "$FINBERT_STATUS" | grep -qi "ready"; then
    check_pass "FinBERT Sentiment: READY"
else
    check_warn "FinBERT Sentiment: Initializing..."
fi

# Check autonomous training
TRAINING_SCHEDULES=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM retraining_schedule WHERE enabled=true;" 2>/dev/null | tr -d ' ' || echo "0")
if [ "$TRAINING_SCHEDULES" -gt 0 ]; then
    check_pass "Autonomous Training: $TRAINING_SCHEDULES models scheduled"
else
    check_fail "Autonomous Training: NOT CONFIGURED"
fi

# ========================================
# 5. TRADING SERVICES
# ========================================
echo ""
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${BLUE}5. TRADING SERVICES STATUS${NC}"
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Signal Generator
SIG_HEALTH=$(curl -s http://localhost:8003/health 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
if [ "$SIG_HEALTH" = "healthy" ]; then
    check_pass "Signal Generator: HEALTHY & OPERATIONAL"
else
    check_fail "Signal Generator: $SIG_HEALTH"
fi

# Execution Service
EXEC_HEALTH=$(curl -s http://localhost:8004/health 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
EXEC_READY=$(curl -s http://localhost:8004/ready 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
if [ "$EXEC_HEALTH" = "healthy" ] && [ "$EXEC_READY" = "ready" ]; then
    check_pass "Execution Engine: HEALTHY & READY TO TRADE"
    
    # Check execution capabilities
    EXEC_JSON=$(curl -s http://localhost:8004/health 2>/dev/null)
    SMART_ROUTING=$(echo "$EXEC_JSON" | grep -o '"smart_order_routing":[^,}]*' | grep -o "true\|false")
    ALGOS=$(echo "$EXEC_JSON" | grep -o '"algorithms":\[[^]]*\]' | grep -o '\[.*\]')
    
    if [ "$SMART_ROUTING" = "true" ]; then
        check_pass "  ↳ Smart Order Routing: ENABLED"
    fi
    if [ -n "$ALGOS" ]; then
        ALGO_COUNT=$(echo "$ALGOS" | grep -o ',' | wc -l)
        ALGO_COUNT=$((ALGO_COUNT + 1))
        check_pass "  ↳ Execution Algorithms: $ALGO_COUNT available"
    fi
else
    check_fail "Execution Engine: Health=$EXEC_HEALTH Ready=$EXEC_READY"
fi

# Risk Monitor
RISK_HEALTH=$(curl -s http://localhost:8005/health 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
if [ "$RISK_HEALTH" = "healthy" ]; then
    check_pass "Risk Monitor: HEALTHY & ACTIVE"
else
    check_fail "Risk Monitor: $RISK_HEALTH"
fi

# Strategy Engine
STRAT_HEALTH=$(curl -s http://localhost:8006/health 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
if [ "$STRAT_HEALTH" = "healthy" ]; then
    check_pass "Strategy Engine: HEALTHY & OPTIMIZING"
else
    check_fail "Strategy Engine: $STRAT_HEALTH"
fi

# ========================================
# 6. LIVE DATA VALIDATION
# ========================================
echo ""
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${BLUE}6. LIVE DATA PROCESSING VALIDATION${NC}"
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check recent data ingestion (last 1 hour)
RECENT_BARS=$(curl -s "http://localhost:9000/exec?query=SELECT%20count()%20FROM%20market_data%20WHERE%20ts%20%3E%20dateadd('h',%20-1,%20now())" 2>/dev/null | grep -o 'dataset":\[\[[0-9]*\]\]' | grep -o '[0-9]*' || echo "0")
if [ "$RECENT_BARS" -gt 100 ]; then
    check_pass "Recent Market Data: $RECENT_BARS bars (last hour) - ACTIVELY PROCESSING"
elif [ "$RECENT_BARS" -gt 0 ]; then
    check_warn "Recent Market Data: $RECENT_BARS bars (last hour) - slow ingestion"
else
    # Check if market is open (US market hours in CEST: 15:30-22:00)
    HOUR=$(date +%-H 2>/dev/null || date +%H)
    MINUTE=$(date +%-M 2>/dev/null || date +%M)
    DAY=$(date +%u)  # 1-7, Monday-Sunday
    
    # Remove leading zeros
    HOUR=${HOUR#0}
    MINUTE=${MINUTE#0}
    [ -z "$HOUR" ] && HOUR=0
    [ -z "$MINUTE" ] && MINUTE=0
    
    TOTAL_MINUTES=$((HOUR * 60 + MINUTE))
    MARKET_OPEN=930    # 15:30 in CEST
    MARKET_CLOSE=1320  # 22:00 in CEST
    
    if [ "$DAY" -le 5 ] && [ "$TOTAL_MINUTES" -ge "$MARKET_OPEN" ] && [ "$TOTAL_MINUTES" -le "$MARKET_CLOSE" ]; then
        check_warn "Recent Market Data: No fresh data (market open, check data feed)"
    else
        check_pass "Recent Market Data: Normal (market closed, $MARKET_COUNT total bars)"
    fi
fi

# Check watchlist - using Redis as primary source
REDIS_PASS=$(grep "^REDIS_PASSWORD=" /srv/ai-trading-system/.env 2>/dev/null | cut -d= -f2)
if [ -n "$REDIS_PASS" ]; then
    WATCHLIST_COUNT=$(docker exec trading-redis redis-cli -a "$REDIS_PASS" SCARD "watchlist" 2>&1 | grep -v Warning | tr -d ' \r' || echo "0")
else
    WATCHLIST_COUNT=$(docker exec trading-redis redis-cli SCARD "watchlist" 2>/dev/null | tr -d ' \r' || echo "0")
fi

# Clean up non-numeric values
if ! [[ "$WATCHLIST_COUNT" =~ ^[0-9]+$ ]]; then
    WATCHLIST_COUNT="0"
fi

if [ "$WATCHLIST_COUNT" -gt 50 ]; then
    check_pass "Active Watchlist: $WATCHLIST_COUNT symbols"
elif [ "$WATCHLIST_COUNT" -gt 0 ]; then
    check_warn "Active Watchlist: $WATCHLIST_COUNT symbols (add more optionable stocks)"
else
    check_fail "Active Watchlist: EMPTY - run: bash scripts/populate_watchlist.sh"
fi

# ========================================
# 7. SYSTEM INTEGRATION
# ========================================
echo ""
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${BLUE}7. END-TO-END INTEGRATION${NC}"
echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check if data flows through the entire pipeline
if [ "$MARKET_COUNT" -gt 1000000 ] && [ "$WEAVIATE_NEWS" -gt 10000 ]; then
    check_pass "Data Pipeline: Historical + Real-time + Vector embeddings"
else
    check_warn "Data Pipeline: Building (historical: $MARKET_COUNT, vectors: $WEAVIATE_NEWS)"
fi

if [ "$SIG_HEALTH" = "healthy" ] && [ "$EXEC_READY" = "ready" ] && [ "$RISK_HEALTH" = "healthy" ]; then
    check_pass "Trading Stack: All services integrated and ready"
else
    check_warn "Trading Stack: Some services not ready yet"
fi

if [ "$TRAINING_SCHEDULES" -gt 0 ] && [ "$ML_HEALTH" = "healthy" ]; then
    check_pass "Intelligence Loop: ML processing + Autonomous training active"
else
    check_warn "Intelligence Loop: Needs configuration"
fi

# ========================================
# SUMMARY
# ========================================
echo ""
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}                    VERIFICATION SUMMARY${NC}"
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

TOTAL=$((PASS + WARN + FAIL))
PERCENT=$((PASS * 100 / TOTAL))

echo -e "  ${BOLD}Results:${NC}"
echo -e "  ┌─────────────────────────────────┐"
echo -e "  │ ${GREEN}Passed:${NC}   ${GREEN}$PASS${NC} checks"
echo -e "  │ ${YELLOW}Warnings:${NC} ${YELLOW}$WARN${NC} checks"
echo -e "  │ ${RED}Failed:${NC}   ${RED}$FAIL${NC} checks"
echo -e "  │ ${BOLD}Total:${NC}    $TOTAL checks"
echo -e "  │ ${BOLD}Score:${NC}    ${PERCENT}%"
echo -e "  └─────────────────────────────────┘"
echo ""

if [ $FAIL -eq 0 ] && [ $WARN -le 3 ]; then
    echo -e "  ${GREEN}${BOLD}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "  ${GREEN}${BOLD}║  ✓ SYSTEM READY FOR LIVE TRADING                         ║${NC}"
    echo -e "  ${GREEN}${BOLD}║                                                           ║${NC}"
    echo -e "  ${GREEN}${BOLD}║  • Data Processing: ACTIVE                                ║${NC}"
    echo -e "  ${GREEN}${BOLD}║  • Intelligence: OPERATIONAL                              ║${NC}"
    echo -e "  ${GREEN}${BOLD}║  • Trading Services: READY                                ║${NC}"
    echo -e "  ${GREEN}${BOLD}║  • Autonomous Learning: ENABLED                           ║${NC}"
    echo -e "  ${GREEN}${BOLD}╚═══════════════════════════════════════════════════════════╝${NC}"
    exit 0
elif [ $FAIL -eq 0 ]; then
    echo -e "  ${YELLOW}${BOLD}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "  ${YELLOW}${BOLD}║  ⚠ SYSTEM OPERATIONAL - Minor Issues                     ║${NC}"
    echo -e "  ${YELLOW}${BOLD}║                                                           ║${NC}"
    echo -e "  ${YELLOW}${BOLD}║  Review warnings above and address before live trading    ║${NC}"
    echo -e "  ${YELLOW}${BOLD}╚═══════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "  ${RED}${BOLD}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "  ${RED}${BOLD}║  ✗ SYSTEM NOT READY FOR LIVE TRADING                     ║${NC}"
    echo -e "  ${RED}${BOLD}║                                                           ║${NC}"
    echo -e "  ${RED}${BOLD}║  Critical issues detected. Address failures above.        ║${NC}"
    echo -e "  ${RED}${BOLD}╚═══════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
