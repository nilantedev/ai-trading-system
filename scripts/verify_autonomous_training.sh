#!/bin/bash
#
# Verify Autonomous Training System
# Checks all components of the continuous training infrastructure
#

set +e
cd /srv/ai-trading-system

echo "==========================================="
echo "AUTONOMOUS TRAINING SYSTEM VERIFICATION"
echo "$(date)"
echo "==========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PASS=0
FAIL=0

check() {
    local name="$1"
    local status=$2
    if [ $status -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $name"
        ((PASS++))
    else
        echo -e "${RED}✗${NC} $name"
        ((FAIL++))
    fi
}

# 1. Database Tables
echo "1. DATABASE SCHEMA"
echo "------------------"
DB_USER=$(grep "^DB_USER=" .env | cut -d= -f2)
DB_NAME=$(grep "^DB_NAME=" .env | cut -d= -f2)

TABLES=$(docker exec trading-postgres psql -U "$DB_USER" -d "$DB_NAME" -t -c "
SELECT COUNT(*) FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('trading_decisions', 'decision_outcomes', 'retraining_schedule', 'model_performance_metrics', 'training_runs')
" 2>/dev/null | xargs)

if [ "$TABLES" = "5" ]; then
    check "All 5 training tables exist" 0
else
    check "All 5 training tables exist (found: $TABLES)" 1
fi

# 2. Retraining Schedules
echo ""
echo "2. RETRAINING SCHEDULES"
echo "-----------------------"
SCHEDULES=$(docker exec trading-postgres psql -U "$DB_USER" -d "$DB_NAME" -t -c "
SELECT COUNT(*) FROM retraining_schedule WHERE enabled = true
" 2>/dev/null | xargs)

if [ "$SCHEDULES" -gt 0 ]; then
    check "Automated retraining schedules configured: $SCHEDULES models" 0
    
    # Show schedule details
    echo ""
    docker exec trading-postgres psql -U "$DB_USER" -d "$DB_NAME" -c "
    SELECT 
        model_name,
        frequency,
        preferred_time,
        TO_CHAR(next_scheduled_run, 'YYYY-MM-DD HH24:MI') as next_run
    FROM retraining_schedule 
    WHERE enabled = true
    ORDER BY next_scheduled_run
    " 2>/dev/null | head -15
else
    check "Automated retraining schedules configured" 1
fi

# 3. ML Service Status
echo ""
echo "3. ML SERVICE STATUS"
echo "--------------------"
if docker ps --filter "name=trading-ml" --format "{{.Status}}" | grep -q "healthy"; then
    check "ML service running and healthy" 0
else
    check "ML service running and healthy" 1
fi

# 4. Training Orchestrator
echo ""
echo "4. TRAINING ORCHESTRATOR"
echo "------------------------"
if docker logs trading-ml 2>&1 | grep -q "Continuous training orchestrator initialized"; then
    check "Training orchestrator initialized" 0
else
    check "Training orchestrator initialized" 1
fi

if docker logs trading-ml 2>&1 | grep -q "Continuous training orchestrator started"; then
    check "Training orchestrator running" 0
else
    check "Training orchestrator running" 1
fi

# 5. Decision Logging Capability
echo ""
echo "5. DECISION LOGGING"
echo "-------------------"
DECISION_COUNT=$(docker exec trading-postgres psql -U "$DB_USER" -d "$DB_NAME" -t -c "
SELECT COUNT(*) FROM trading_decisions
" 2>/dev/null | xargs)

if [ -n "$DECISION_COUNT" ]; then
    check "Decision logging table accessible ($DECISION_COUNT decisions logged)" 0
else
    check "Decision logging table accessible" 1
fi

# 6. Performance Tracking
echo ""
echo "6. PERFORMANCE TRACKING"
echo "-----------------------"
PERF_COUNT=$(docker exec trading-postgres psql -U "$DB_USER" -d "$DB_NAME" -t -c "
SELECT COUNT(*) FROM model_performance_metrics
" 2>/dev/null | xargs)

if [ -n "$PERF_COUNT" ]; then
    check "Performance metrics tracking ($PERF_COUNT records)" 0
else
    check "Performance metrics tracking" 1
fi

# 7. Training Data Snapshots
echo ""
echo "7. TRAINING DATA STORAGE"
echo "------------------------"
SNAPSHOT_COUNT=$(docker exec trading-postgres psql -U "$DB_USER" -d "$DB_NAME" -t -c "
SELECT COUNT(*) FROM training_data_snapshots
" 2>/dev/null | xargs)

if [ -n "$SNAPSHOT_COUNT" ]; then
    check "Training data snapshots ($SNAPSHOT_COUNT snapshots)" 0
else
    check "Training data snapshots" 1
fi

# 8. Check training runs history
TRAINING_RUNS=$(docker exec trading-postgres psql -U "$DB_USER" -d "$DB_NAME" -t -c "
SELECT COUNT(*) FROM training_runs
" 2>/dev/null | xargs)

if [ -n "$TRAINING_RUNS" ]; then
    check "Training runs history ($TRAINING_RUNS runs)" 0
else
    check "Training runs history" 1
fi

# Summary
echo ""
echo "==========================================="
echo "SUMMARY"
echo "==========================================="
TOTAL=$((PASS + FAIL))
SUCCESS_RATE=$((PASS * 100 / TOTAL))

echo "Passed: $PASS"
echo "Failed: $FAIL"
echo "Success Rate: $SUCCESS_RATE%"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ AUTONOMOUS TRAINING SYSTEM READY${NC}"
    echo ""
    echo "Features Enabled:"
    echo "  • Automated model retraining during off-hours"
    echo "  • All trading decisions logged for learning"
    echo "  • Performance monitoring and drift detection"
    echo "  • Self-training on captured data"
    echo "  • Continuous model improvement"
    echo ""
    echo "Next Actions:"
    echo "  - System will automatically retrain models based on schedules"
    echo "  - Check logs: docker logs trading-ml -f | grep training"
    echo "  - Monitor performance: psql -U $DB_USER -d $DB_NAME -c 'SELECT * FROM retraining_schedule'"
    exit 0
else
    echo -e "${RED}✗ SETUP INCOMPLETE${NC}"
    echo ""
    echo "Issues found: $FAIL"
    echo "Review the failed checks above."
    exit 1
fi
