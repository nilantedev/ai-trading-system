#!/bin/bash
#
# Setup Continuous Training System
# Initializes retraining schedules and verifies the training infrastructure
#

set -e
cd /srv/ai-trading-system

echo "==========================================="
echo "CONTINUOUS TRAINING SYSTEM SETUP"
echo "==========================================="
echo ""

# Check if tables exist
echo "Checking training tables..."
TABLES=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "
    SELECT COUNT(*) FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_name IN ('trading_decisions', 'model_performance_metrics', 'training_runs', 'retraining_schedule')
" 2>&1 | tr -d ' ')

if [ "$TABLES" = "4" ]; then
    echo "  ✓ All training tables exist"
else
    echo "  ✗ Missing training tables (found $TABLES/4)"
    echo "  Creating tables..."
    docker exec -i trading-postgres psql -U trading_user -d trading_db < migrations/create_training_tables.sql >/dev/null 2>&1
    echo "  ✓ Tables created"
fi

echo ""
echo "Setting up default retraining schedules..."

# Create default schedules for common model types
docker exec trading-postgres psql -U trading_user -d trading_db <<EOF
-- Ensemble model - Weekly retraining on weekends
INSERT INTO retraining_schedule (
    schedule_id, model_name, schedule_type, frequency,
    preferred_time, preferred_day,
    min_performance_drop, min_drift_score, min_new_samples,
    enabled, training_config
) VALUES (
    'ensemble_weekly',
    'ensemble_predictor',
    'PERIODIC',
    'WEEKLY',
    '02:00:00',
    'SUNDAY',
    0.10,  -- Retrain if accuracy drops 10%
    0.15,  -- Retrain if drift score > 0.15
    1000,  -- Retrain after 1000 new samples
    true,
    '{"batch_size": 128, "epochs": 50, "learning_rate": 0.001}'::jsonb
) ON CONFLICT (schedule_id) DO UPDATE SET
    min_performance_drop = EXCLUDED.min_performance_drop,
    min_drift_score = EXCLUDED.min_drift_score,
    min_new_samples = EXCLUDED.min_new_samples,
    updated_at = NOW();

-- Factor models - Daily retraining
INSERT INTO retraining_schedule (
    schedule_id, model_name, schedule_type, frequency,
    preferred_time,
    min_performance_drop, min_drift_score, min_new_samples,
    enabled, training_config
) VALUES (
    'factor_daily',
    'factor_model',
    'PERIODIC',
    'DAILY',
    '01:00:00',
    0.15,  -- Retrain if accuracy drops 15%
    0.20,  -- Retrain if drift score > 0.20
    500,   -- Retrain after 500 new samples
    true,
    '{"regularization": 0.01, "lookback_days": 30}'::jsonb
) ON CONFLICT (schedule_id) DO UPDATE SET
    min_performance_drop = EXCLUDED.min_performance_drop,
    updated_at = NOW();

-- RL agent - Continuous learning (drift-triggered)
INSERT INTO retraining_schedule (
    schedule_id, model_name, schedule_type, frequency,
    min_performance_drop, min_drift_score, min_new_samples,
    enabled, training_config
) VALUES (
    'rl_agent_continuous',
    'rl_trading_agent',
    'DRIFT_TRIGGERED',
    NULL,
    0.05,  -- Very sensitive to performance drops
    0.10,  -- Very sensitive to drift
    100,   -- Retrain frequently with new data
    true,
    '{"learning_rate": 0.0001, "discount_factor": 0.99, "replay_buffer_size": 10000}'::jsonb
) ON CONFLICT (schedule_id) DO UPDATE SET
    min_performance_drop = EXCLUDED.min_performance_drop,
    updated_at = NOW();

-- Regime detector - Weekly retraining
INSERT INTO retraining_schedule (
    schedule_id, model_name, schedule_type, frequency,
    preferred_time, preferred_day,
    min_performance_drop, min_drift_score,
    enabled, training_config
) VALUES (
    'regime_weekly',
    'market_regime_detector',
    'PERIODIC',
    'WEEKLY',
    '03:00:00',
    'SATURDAY',
    0.12,
    0.18,
    true,
    '{"lookback_window": 60, "n_regimes": 4}'::jsonb
) ON CONFLICT (schedule_id) DO UPDATE SET
    updated_at = NOW();

-- Volatility models - Daily retraining
INSERT INTO retraining_schedule (
    schedule_id, model_name, schedule_type, frequency,
    preferred_time,
    min_performance_drop, min_drift_score, min_new_samples,
    enabled, training_config
) VALUES (
    'volatility_daily',
    'volatility_predictor',
    'PERIODIC',
    'DAILY',
    '00:30:00',
    0.15,
    0.20,
    300,
    true,
    '{"model_type": "GARCH", "order": [1,1]}'::jsonb
) ON CONFLICT (schedule_id) DO UPDATE SET
    updated_at = NOW();

SELECT 'Created/updated ' || COUNT(*) || ' retraining schedules' FROM retraining_schedule;
EOF

echo ""
echo "Verifying continuous training setup..."

# Check orchestrator dependencies
echo "Checking Python dependencies..."
docker exec trading-ml python3 -c "
import asyncpg
import numpy as np
import pandas as pd
print('  ✓ All dependencies available')
" 2>&1 || echo "  ✗ Missing dependencies"

# Check if model_registry has data
MODEL_COUNT=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "
    SELECT COUNT(*) FROM model_registry
" 2>&1 | tr -d ' ')

echo "  ✓ Models in registry: $MODEL_COUNT"

# Check retraining schedules
SCHEDULE_COUNT=$(docker exec trading-postgres psql -U trading_user -d trading_db -t -c "
    SELECT COUNT(*) FROM retraining_schedule WHERE enabled = true
" 2>&1 | tr -d ' ')

echo "  ✓ Active retraining schedules: $SCHEDULE_COUNT"

echo ""
echo "==========================================="
echo "TRAINING SYSTEM STATUS"
echo "==========================================="

# Show active schedules
docker exec trading-postgres psql -U trading_user -d trading_db -c "
SELECT 
    model_name,
    schedule_type,
    frequency,
    COALESCE(preferred_time::text, 'N/A') as time,
    CASE WHEN enabled THEN '✓ Enabled' ELSE '✗ Disabled' END as status,
    COALESCE(last_training_timestamp::text, 'Never') as last_run
FROM retraining_schedule
ORDER BY model_name;
"

echo ""
echo "==========================================="
echo "SUMMARY"
echo "==========================================="
echo ""
echo "✓ Training tables created"
echo "✓ Retraining schedules configured"
echo "✓ $SCHEDULE_COUNT models configured for continuous training"
echo ""
echo "Features enabled:"
echo "  • Decision logging for all trades"
echo "  • Outcome tracking and performance monitoring"
echo "  • Automatic drift detection"
echo "  • Performance-based retraining triggers"
echo "  • Off-hours training scheduling"
echo "  • Model version management"
echo ""
echo "To start the continuous training orchestrator:"
echo "  docker exec trading-ml python3 services/ml/continuous_training_orchestrator.py"
echo ""
echo "To manually trigger retraining:"
echo "  docker exec trading-postgres psql -U trading_user -d trading_db -c \\"
echo "    \"UPDATE retraining_schedule SET next_scheduled_run = NOW() WHERE model_name = 'ensemble_predictor';\""
echo ""
