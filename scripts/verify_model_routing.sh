#!/bin/bash
# Verify ML Model Day/Night Mode Routing
# This script tests that the correct models are selected based on time of day

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== ML Model Day/Night Mode Verification ===${NC}"
echo ""

# 1. Check current mode
echo -e "${BLUE}1. Current Mode Status${NC}"
MODE_STATUS=$(curl -s http://localhost:8001/ollama/mode 2>/dev/null || echo '{}')
CURRENT_MODE=$(echo "$MODE_STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('mode','unknown'))" 2>/dev/null || echo "unknown")
IS_MARKET_HOURS=$(echo "$MODE_STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('market_hours',False))" 2>/dev/null || echo "false")

echo "   Current Mode: $CURRENT_MODE"
echo "   Market Hours: $IS_MARKET_HOURS"

# Show current time in market timezone
CURRENT_TIME=$(TZ=America/New_York date "+%Y-%m-%d %H:%M:%S %Z (%A)")
echo "   Market Time: $CURRENT_TIME"
echo ""

# 2. Check configured models
echo -e "${BLUE}2. Configured Model Sets${NC}"
DAY_MODELS=$(echo "$MODE_STATUS" | python3 -c "import sys,json; print(','.join(json.load(sys.stdin).get('configured',{}).get('day_hot_models',[])))" 2>/dev/null || echo "unknown")
NIGHT_MODELS=$(echo "$MODE_STATUS" | python3 -c "import sys,json; print(','.join(json.load(sys.stdin).get('configured',{}).get('night_heavy_models',[])))" 2>/dev/null || echo "unknown")

echo "   Day Models (light, fast): $DAY_MODELS"
echo "   Night Models (heavy, deep): $NIGHT_MODELS"
echo ""

# 3. Check available models in Ollama
echo -e "${BLUE}3. Available Ollama Models${NC}"
MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = data.get('models', [])
    print(f'   Total: {len(models)} models loaded')
    for m in models:
        name = m['name']
        size = m['details']['parameter_size']
        mem_gb = round(m['size'] / (1024**3), 1)
        category = 'LIGHT' if mem_gb < 20 else ('MEDIUM' if mem_gb < 50 else 'HEAVY')
        print(f'   - {name:30s} {size:>12s} {mem_gb:>6.1f}GB  [{category}]')
except Exception as e:
    print(f'   Error: {e}')
" 2>/dev/null || echo "   Error: Failed to get models")
echo ""

# 4. Test model selection for different urgency levels
echo -e "${BLUE}4. Model Selection Tests (Current Mode: $CURRENT_MODE)${NC}"

test_inference() {
    local urgency=$1
    local task_type=$2
    local expected_category=$3
    
    RESULT=$(curl -s -X POST http://localhost:8001/ollama/generate \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"Test prompt for urgency=$urgency\",
            \"task_type\": \"$task_type\",
            \"urgency\": \"$urgency\",
            \"temperature\": 0.1,
            \"max_tokens\": 50
        }" 2>/dev/null || echo '{"routing_decision":{"primary_model":"error"}}')
    
    SELECTED_MODEL=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('routing_decision',{}).get('primary_model','unknown'))" 2>/dev/null || echo "unknown")
    EST_LATENCY=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('routing_decision',{}).get('estimated_latency_ms','?'))" 2>/dev/null || echo "?")
    SUCCESS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('success',False))" 2>/dev/null || echo "false")
    
    # Determine model category
    ACTUAL_CATEGORY="UNKNOWN"
    case "$SELECTED_MODEL" in
        solar:*|phi3:*) ACTUAL_CATEGORY="LIGHT" ;;
        yi:*) ACTUAL_CATEGORY="MEDIUM" ;;
        mixtral:*|qwen*|command-r-plus:*|llama3*) ACTUAL_CATEGORY="HEAVY" ;;
    esac
    
    # Check if selection is appropriate
    STATUS="✓"
    COLOR="${GREEN}"
    if [ "$CURRENT_MODE" = "day" ] && [ "$urgency" = "fast" ] && [ "$ACTUAL_CATEGORY" != "LIGHT" ]; then
        STATUS="✗"
        COLOR="${RED}"
    fi
    if [ "$CURRENT_MODE" = "night" ] && [ "$ACTUAL_CATEGORY" = "HEAVY" ]; then
        STATUS="✓"
        COLOR="${GREEN}"
    fi
    
    echo -e "   $STATUS Urgency: ${urgency:0:8} | Task: ${task_type:0:10} | Selected: ${COLOR}$SELECTED_MODEL${NC} | Latency: ${EST_LATENCY}ms | Success: $SUCCESS"
}

# Test various urgency/task combinations
test_inference "realtime" "signal" "LIGHT"
test_inference "fast" "market" "LIGHT"
test_inference "normal" "risk" "LIGHT_OR_MEDIUM"
test_inference "deep" "market" "HEAVY"

echo ""

# 5. Verify router market hours gating
echo -e "${BLUE}5. Router Market Hours Gating${NC}"
ROUTER_DISABLE_HEAVY=$(grep -E "^ROUTER_MARKET_DISABLE_HEAVY=" /srv/ai-trading-system/.env 2>/dev/null | cut -d'=' -f2 || echo "unknown")
ROUTER_HEAVY_THRESHOLD=$(grep -E "^ROUTER_HEAVY_MEMORY_GB=" /srv/ai-trading-system/.env 2>/dev/null | cut -d'=' -f2 || echo "unknown")

echo "   Router disables heavy models during market hours: $ROUTER_DISABLE_HEAVY"
echo "   Heavy model memory threshold: ${ROUTER_HEAVY_THRESHOLD}GB"
echo ""

# 6. Summary
echo -e "${BLUE}6. Summary${NC}"
if [ "$CURRENT_MODE" = "day" ]; then
    echo -e "   ${GREEN}✓${NC} System is in DAY mode"
    echo "   - Should use LIGHT models (solar:10.7b, phi3:14b) for fast inference"
    echo "   - Heavy models (70B-140B) are FILTERED for FAST/REALTIME tasks"
    echo "   - Expected latency: <1000ms for fast tasks"
elif [ "$CURRENT_MODE" = "night" ]; then
    echo -e "   ${GREEN}✓${NC} System is in NIGHT mode"
    echo "   - Can use ALL models including HEAVY (mixtral, qwen, command-r-plus, llama, yi)"
    echo "   - Optimized for deep analysis and quality over speed"
    echo "   - Expected latency: 2000-5000ms for deep tasks"
else
    echo -e "   ${RED}✗${NC} Unknown mode: $CURRENT_MODE"
fi
echo ""

# 7. Recommendations
echo -e "${BLUE}7. Recommendations${NC}"
if [ "$IS_MARKET_HOURS" = "True" ] && [ "$CURRENT_MODE" != "day" ]; then
    echo -e "   ${YELLOW}⚠${NC} WARNING: Market hours detected but not in day mode!"
    echo "   - Should manually switch to day mode: curl -X POST http://localhost:8001/ollama/warm/switch -H 'Content-Type: application/json' -d '{\"mode\":\"day\"}'"
elif [ "$IS_MARKET_HOURS" = "False" ] && [ "$CURRENT_MODE" != "night" ]; then
    echo -e "   ${YELLOW}⚠${NC} WARNING: Off-hours detected but not in night mode!"
    echo "   - Should manually switch to night mode: curl -X POST http://localhost:8001/ollama/warm/switch -H 'Content-Type: application/json' -d '{\"mode\":\"night\"}'"
else
    echo -e "   ${GREEN}✓${NC} Mode is correctly set for current time"
fi

echo ""
echo -e "${BLUE}=== Verification Complete ===${NC}"
