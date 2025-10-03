"""
create_training_tables.sql - Database schema for continuous training system
Tracks trade decisions, outcomes, and model performance for self-training
"""

-- Table: trading_decisions
-- Logs every trading decision made by the system for later analysis
CREATE TABLE IF NOT EXISTS trading_decisions (
    id SERIAL PRIMARY KEY,
    decision_id VARCHAR(64) UNIQUE NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    
    -- Decision details
    action VARCHAR(20) NOT NULL, -- BUY, SELL, HOLD
    confidence FLOAT NOT NULL,
    position_size FLOAT,
    entry_price FLOAT,
    target_price FLOAT,
    stop_loss FLOAT,
    
    -- Model predictions
    predicted_return FLOAT,
    predicted_direction VARCHAR(10), -- UP, DOWN, NEUTRAL
    prediction_horizon_minutes INT,
    
    -- Features used in decision
    features_json JSONB NOT NULL,
    
    -- Model information
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    model_ensemble JSONB, -- Array of models used
    
    -- Market context
    market_regime VARCHAR(50),
    volatility_level VARCHAR(20),
    volume_profile VARCHAR(20),
    
    -- Risk metrics at decision time
    portfolio_exposure FLOAT,
    var_95 FLOAT,
    sharpe_ratio FLOAT,
    
    -- Outcome tracking (updated after execution)
    executed BOOLEAN DEFAULT FALSE,
    execution_price FLOAT,
    execution_timestamp TIMESTAMP,
    actual_return FLOAT,
    outcome VARCHAR(20), -- WIN, LOSS, BREAKEVEN
    pnl FLOAT,
    
    -- Performance metrics (updated after close)
    max_favorable_excursion FLOAT,
    max_adverse_excursion FLOAT,
    hold_duration_minutes INT,
    exit_reason VARCHAR(50),
    
    -- Metadata
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_trading_decisions_symbol ON trading_decisions(symbol);
CREATE INDEX idx_trading_decisions_timestamp ON trading_decisions(timestamp DESC);
CREATE INDEX idx_trading_decisions_outcome ON trading_decisions(outcome);
CREATE INDEX idx_trading_decisions_executed ON trading_decisions(executed);
CREATE INDEX idx_trading_decisions_model ON trading_decisions(model_name, model_version);

-- Table: model_performance_metrics
-- Tracks model performance over time for drift detection and retraining triggers
CREATE TABLE IF NOT EXISTS model_performance_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    
    -- Time window
    evaluation_period VARCHAR(20) NOT NULL, -- HOURLY, DAILY, WEEKLY
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    
    -- Prediction accuracy
    total_predictions INT NOT NULL,
    correct_predictions INT,
    accuracy FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    
    -- Regression metrics (for return prediction)
    mse FLOAT,
    rmse FLOAT,
    mae FLOAT,
    r2_score FLOAT,
    
    -- Trading performance
    total_trades INT,
    winning_trades INT,
    losing_trades INT,
    win_rate FLOAT,
    avg_win FLOAT,
    avg_loss FLOAT,
    profit_factor FLOAT,
    sharpe_ratio FLOAT,
    max_drawdown FLOAT,
    total_pnl FLOAT,
    
    -- Drift indicators
    feature_drift_score FLOAT,
    prediction_drift_score FLOAT,
    performance_degradation FLOAT,
    
    -- Confidence calibration
    avg_confidence FLOAT,
    confidence_accuracy_correlation FLOAT,
    overconfidence_ratio FLOAT,
    
    -- Metadata
    symbols_traded JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_model_performance_timestamp ON model_performance_metrics(timestamp DESC);
CREATE INDEX idx_model_performance_model ON model_performance_metrics(model_name, model_version);
CREATE INDEX idx_model_performance_period ON model_performance_metrics(evaluation_period, period_start);

-- Table: training_data_snapshots
-- Stores snapshots of training data for reproducibility and incremental learning
CREATE TABLE IF NOT EXISTS training_data_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_id VARCHAR(64) UNIQUE NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Data details
    data_type VARCHAR(50) NOT NULL, -- MARKET_DATA, NEWS, SOCIAL, FEATURES
    symbol VARCHAR(20),
    
    -- Time range
    data_start TIMESTAMP NOT NULL,
    data_end TIMESTAMP NOT NULL,
    total_records INT NOT NULL,
    
    -- Storage location
    storage_backend VARCHAR(50), -- MINIO, S3, LOCAL
    storage_path VARCHAR(500),
    
    -- Data statistics
    feature_stats JSONB,
    target_stats JSONB,
    
    -- Quality metrics
    completeness_score FLOAT,
    outlier_count INT,
    null_count INT,
    
    -- Metadata
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_training_snapshots_timestamp ON training_data_snapshots(timestamp DESC);
CREATE INDEX idx_training_snapshots_symbol ON training_data_snapshots(symbol);
CREATE INDEX idx_training_snapshots_type ON training_data_snapshots(data_type);

-- Table: retraining_schedule
-- Manages automated retraining schedule and triggers
CREATE TABLE IF NOT EXISTS retraining_schedule (
    id SERIAL PRIMARY KEY,
    schedule_id VARCHAR(64) UNIQUE NOT NULL,
    
    -- Schedule details
    model_name VARCHAR(100) NOT NULL,
    schedule_type VARCHAR(50) NOT NULL, -- PERIODIC, DRIFT_TRIGGERED, PERFORMANCE_TRIGGERED
    frequency VARCHAR(50), -- DAILY, WEEKLY, MONTHLY
    preferred_time TIME, -- e.g., '02:00:00' for 2 AM
    preferred_day VARCHAR(20), -- MONDAY, TUESDAY, etc.
    
    -- Trigger conditions
    min_performance_drop FLOAT, -- Retrain if performance drops by this %
    min_drift_score FLOAT, -- Retrain if drift score exceeds this
    min_new_samples INT, -- Retrain after this many new samples
    
    -- Execution tracking
    enabled BOOLEAN DEFAULT TRUE,
    last_training_timestamp TIMESTAMP,
    last_training_duration_seconds INT,
    last_training_status VARCHAR(50),
    next_scheduled_run TIMESTAMP,
    
    -- Training configuration
    training_config JSONB, -- Hyperparameters, data sources, etc.
    
    -- Metadata
    created_by VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_retraining_schedule_model ON retraining_schedule(model_name);
CREATE INDEX idx_retraining_schedule_next_run ON retraining_schedule(next_scheduled_run);
CREATE INDEX idx_retraining_schedule_enabled ON retraining_schedule(enabled);

-- Table: training_runs
-- Tracks individual training runs for audit and comparison
CREATE TABLE IF NOT EXISTS training_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) UNIQUE NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Model details
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    parent_version VARCHAR(50), -- Previous version this was trained from
    
    -- Training details
    training_type VARCHAR(50), -- FULL_RETRAIN, INCREMENTAL, FINE_TUNE
    trigger_reason VARCHAR(100), -- SCHEDULED, DRIFT_DETECTED, MANUAL
    data_snapshot_id VARCHAR(64),
    
    -- Dataset info
    training_samples INT,
    validation_samples INT,
    test_samples INT,
    feature_count INT,
    training_duration_seconds INT,
    
    -- Training results
    train_loss FLOAT,
    val_loss FLOAT,
    test_loss FLOAT,
    train_accuracy FLOAT,
    val_accuracy FLOAT,
    test_accuracy FLOAT,
    
    -- Model comparison
    baseline_metrics JSONB,
    new_model_metrics JSONB,
    improvement_percentage FLOAT,
    
    -- Hyperparameters
    hyperparameters JSONB,
    
    -- Promotion decision
    promoted_to_production BOOLEAN DEFAULT FALSE,
    promotion_timestamp TIMESTAMP,
    promotion_reason TEXT,
    
    -- Artifacts
    model_artifact_path VARCHAR(500),
    training_logs_path VARCHAR(500),
    
    -- Metadata
    trained_by VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_training_runs_timestamp ON training_runs(timestamp DESC);
CREATE INDEX idx_training_runs_model ON training_runs(model_name, model_version);
CREATE INDEX idx_training_runs_promoted ON training_runs(promoted_to_production);

-- Table: feature_importance_history
-- Tracks feature importance over time to detect concept drift
CREATE TABLE IF NOT EXISTS feature_importance_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    
    -- Feature details
    feature_name VARCHAR(200) NOT NULL,
    importance_score FLOAT NOT NULL,
    importance_rank INT,
    
    -- Context
    evaluation_period VARCHAR(20),
    symbol VARCHAR(20),
    
    -- Statistical tests
    importance_std FLOAT,
    importance_change_from_baseline FLOAT,
    
    -- Metadata
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_feature_importance_timestamp ON feature_importance_history(timestamp DESC);
CREATE INDEX idx_feature_importance_model ON feature_importance_history(model_name, model_version);
CREATE INDEX idx_feature_importance_feature ON feature_importance_history(feature_name);

-- View: model_health_dashboard
-- Real-time view of model health for monitoring
CREATE OR REPLACE VIEW model_health_dashboard AS
SELECT 
    m.model_name,
    m.model_version,
    m.status,
    m.updated_at as last_updated,
    
    -- Recent performance
    pm.accuracy as recent_accuracy,
    pm.sharpe_ratio as recent_sharpe,
    pm.win_rate as recent_win_rate,
    pm.total_pnl as recent_pnl,
    
    -- Drift indicators
    pm.feature_drift_score,
    pm.performance_degradation,
    
    -- Training info
    tr.timestamp as last_training,
    tr.improvement_percentage as last_improvement,
    tr.promoted_to_production,
    
    -- Schedule info
    rs.next_scheduled_run,
    rs.enabled as retraining_enabled
    
FROM model_registry m
LEFT JOIN LATERAL (
    SELECT * FROM model_performance_metrics
    WHERE model_name = m.model_name 
    AND model_version = m.version
    ORDER BY timestamp DESC LIMIT 1
) pm ON true
LEFT JOIN LATERAL (
    SELECT * FROM training_runs
    WHERE model_name = m.model_name 
    AND model_version = m.version
    ORDER BY timestamp DESC LIMIT 1
) tr ON true
LEFT JOIN retraining_schedule rs ON rs.model_name = m.model_name;

-- Function: update_decision_outcome
-- Updates trading decision with actual outcome
CREATE OR REPLACE FUNCTION update_decision_outcome(
    p_decision_id VARCHAR(64),
    p_execution_price FLOAT,
    p_actual_return FLOAT,
    p_pnl FLOAT,
    p_exit_reason VARCHAR(50) DEFAULT NULL,
    p_hold_duration_minutes INT DEFAULT NULL
) RETURNS void AS $$
BEGIN
    UPDATE trading_decisions
    SET 
        executed = TRUE,
        execution_timestamp = NOW(),
        execution_price = p_execution_price,
        actual_return = p_actual_return,
        pnl = p_pnl,
        outcome = CASE 
            WHEN p_pnl > 0.001 THEN 'WIN'
            WHEN p_pnl < -0.001 THEN 'LOSS'
            ELSE 'BREAKEVEN'
        END,
        exit_reason = COALESCE(p_exit_reason, exit_reason),
        hold_duration_minutes = COALESCE(p_hold_duration_minutes, hold_duration_minutes),
        updated_at = NOW()
    WHERE decision_id = p_decision_id;
END;
$$ LANGUAGE plpgsql;

-- Function: calculate_model_performance
-- Calculates performance metrics for a model over a time period
CREATE OR REPLACE FUNCTION calculate_model_performance(
    p_model_name VARCHAR(100),
    p_model_version VARCHAR(50),
    p_period_start TIMESTAMP,
    p_period_end TIMESTAMP
) RETURNS TABLE (
    total_decisions BIGINT,
    executed_decisions BIGINT,
    accuracy FLOAT,
    win_rate FLOAT,
    total_pnl FLOAT,
    sharpe FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_decisions,
        COUNT(CASE WHEN executed THEN 1 END)::BIGINT as executed_decisions,
        AVG(CASE 
            WHEN executed AND predicted_direction = 
                CASE WHEN actual_return > 0 THEN 'UP' 
                     WHEN actual_return < 0 THEN 'DOWN' 
                     ELSE 'NEUTRAL' END
            THEN 1.0 ELSE 0.0 
        END) as accuracy,
        AVG(CASE WHEN outcome = 'WIN' THEN 1.0 ELSE 0.0 END) as win_rate,
        SUM(COALESCE(pnl, 0.0)) as total_pnl,
        CASE 
            WHEN STDDEV(pnl) > 0 THEN AVG(pnl) / STDDEV(pnl) * SQRT(COUNT(*))
            ELSE 0
        END as sharpe
    FROM trading_decisions
    WHERE model_name = p_model_name
    AND model_version = p_model_version
    AND timestamp >= p_period_start
    AND timestamp <= p_period_end;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO trading_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO trading_user;
