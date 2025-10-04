#!/bin/bash

# ML Processing & Trading System Readiness Check

echo "═══════════════════════════════════════════════════════════════"
echo "  ML PROCESSING & TRADING SYSTEM STATUS"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check ML service container
echo "🤖 ML SERVICE STATUS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ml_status=$(docker inspect -f '{{.State.Status}}' trading-ml 2>/dev/null || echo "not found")
echo "Container status: $ml_status"

if [ "$ml_status" = "running" ]; then
    # Check Ollama models
    echo ""
    echo "Loaded LLM Models:"
    docker exec trading-ml curl -s http://localhost:11434/api/tags 2>/dev/null | jq -r '.models[]? | "  ✓ \(.name) - \(.size / 1024 / 1024 / 1024 | floor)GB"' 2>/dev/null || echo "  Unable to query models"
    
    # Check recent ML activity
    echo ""
    echo "Recent ML Activity (last hour):"
    ml_activity=$(docker logs trading-ml --since 1h 2>&1 | grep -iE "prediction|inference|feature|signal" | wc -l)
    echo "  ML operations: $ml_activity"
fi

# Check signal generator
echo ""
echo "📊 SIGNAL GENERATOR STATUS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
signal_status=$(docker inspect -f '{{.State.Status}}' trading-signal-generator 2>/dev/null || echo "not found")
echo "Container status: $signal_status"

if [ "$signal_status" = "running" ]; then
    # Check recent signals
    echo ""
    echo "Recent Signal Activity (last hour):"
    signals=$(docker logs trading-signal-generator --since 1h 2>&1 | grep -iE "signal.*generated|trading.*signal" | wc -l)
    echo "  Signals generated: $signals"
fi

# Check PostgreSQL ML tables
echo ""
echo "🗄️  ML DATA IN POSTGRESQL:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

model_registry=$(docker exec -i trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM model_registry;" 2>/dev/null | xargs)
training_runs=$(docker exec -i trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM training_runs;" 2>/dev/null | xargs)
model_metrics=$(docker exec -i trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM model_performance_metrics;" 2>/dev/null | xargs)
trading_decisions=$(docker exec -i trading-postgres psql -U trading_user -d trading_db -t -c "SELECT COUNT(*) FROM trading_decisions;" 2>/dev/null | xargs)

printf "%-40s %15s\n" "Registered models:" "${model_registry:-0}"
printf "%-40s %15s\n" "Training runs:" "${training_runs:-0}"
printf "%-40s %15s\n" "Model metrics:" "${model_metrics:-0}"
printf "%-40s %15s\n" "Trading decisions:" "${trading_decisions:-0}"

# Check QuestDB trading signals
echo ""
echo "📡 TRADING SIGNALS (QuestDB):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

signals_result=$(curl -s "http://localhost:9000/exec?query=SELECT%20count%28%2A%29%20FROM%20trading_signals" 2>/dev/null)
signals_count=$(echo "$signals_result" | jq -r '.dataset[]?[0]?' 2>/dev/null || echo "0")
printf "%-40s %15s\n" "Total trading signals:" "${signals_count}"

# Check recent signals (last 24h)
recent_signals_result=$(curl -s "http://localhost:9000/exec" --data-urlencode "query=SELECT count(*) FROM trading_signals WHERE timestamp > dateadd('d', -1, now())" 2>/dev/null)
recent_signals=$(echo "$recent_signals_result" | jq -r '.dataset[]?[0]?' 2>/dev/null || echo "0")
printf "%-40s %15s\n" "Recent signals (24h):" "${recent_signals}"

# Check TRADING SYSTEM components
echo ""
echo "💼 TRADING SYSTEM STATUS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for service in execution risk-monitor strategy-engine backtesting; do
    status=$(docker inspect -f '{{.State.Status}}' trading-$service 2>/dev/null || echo "not found")
    if [ "$status" = "running" ]; then
        echo "✅ trading-$service: Running"
    else
        echo "❌ trading-$service: $status"
    fi
done

# Check trading mode
echo ""
echo "Trading Mode Configuration:"
trading_mode=$(docker exec trading-execution env 2>/dev/null | grep "TRADING_MODE" | cut -d= -f2 || echo "unknown")
echo "  Mode: ${trading_mode:-unknown}"

# Check Pulsar message flow
echo ""
echo "📨 PULSAR MESSAGE STREAMING:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
pulsar_status=$(docker inspect -f '{{.State.Status}}' trading-pulsar 2>/dev/null || echo "not found")
echo "Pulsar broker: $pulsar_status"

if [ "$pulsar_status" = "running" ]; then
    # Check topics
    topics=$(docker exec trading-pulsar /pulsar/bin/pulsar-admin topics list public/default 2>/dev/null | wc -l)
    echo "Active topics: ${topics:-0}"
fi

# Check Redis (feature store)
echo ""
echo "💾 REDIS FEATURE STORE:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
redis_status=$(docker inspect -f '{{.State.Status}}' trading-redis 2>/dev/null || echo "not found")
echo "Redis status: $redis_status"

if [ "$redis_status" = "running" ]; then
    redis_keys=$(docker exec trading-redis redis-cli DBSIZE 2>/dev/null | grep -oP '\d+' || echo "0")
    echo "Cached keys: ${redis_keys:-0}"
fi

# Overall system health
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SYSTEM READINESS ASSESSMENT:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Count healthy services
healthy=0
total=0

for service in ml signal-generator execution risk-monitor strategy-engine backtesting data-ingestion pulsar redis postgres questdb; do
    total=$((total + 1))
    status=$(docker inspect -f '{{.State.Status}}' trading-$service 2>/dev/null || echo "not found")
    if [ "$status" = "running" ]; then
        healthy=$((healthy + 1))
    fi
done

echo "Container Health: $healthy/$total services running"

# Data readiness
if [ "${signals_count:-0}" -gt 0 ]; then
    echo "✅ Trading signals: Available ($signals_count total)"
else
    echo "⚠️  Trading signals: None generated yet"
fi

if [ "${model_registry:-0}" -gt 0 ]; then
    echo "✅ ML models: Registered ($model_registry models)"
else
    echo "⚠️  ML models: None registered"
fi

if [ "$trading_mode" = "PAPER" ] || [ "$trading_mode" = "LIVE" ]; then
    echo "✅ Trading mode: $trading_mode"
else
    echo "⚠️  Trading mode: Not configured"
fi

echo ""
if [ $healthy -ge 10 ] && [ "${signals_count:-0}" -gt 0 ]; then
    echo "🟢 SYSTEM READY FOR TRADING"
elif [ $healthy -ge 8 ]; then
    echo "🟡 SYSTEM PARTIALLY READY - Some services may need attention"
else
    echo "🔴 SYSTEM NOT READY - Multiple services down"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
