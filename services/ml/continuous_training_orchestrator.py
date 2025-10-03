#!/usr/bin/env python3
"""
Continuous Training Orchestrator - Manages self-training and continuous improvement
Monitors model performance, triggers retraining, and captures all decisions for learning
"""

import asyncio
import asyncpg
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))

from trading_common import get_settings, get_logger
from trading_common.cache import get_trading_cache

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class TradingDecision:
    """Trading decision to be logged for training."""
    decision_id: str
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    position_size: float
    entry_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    predicted_return: float
    predicted_direction: str  # UP, DOWN, NEUTRAL
    prediction_horizon_minutes: int
    features: Dict[str, float]
    model_name: str
    model_version: str
    model_ensemble: List[str]
    market_regime: str
    volatility_level: str
    portfolio_exposure: float
    metadata: Dict[str, Any]


@dataclass
class DecisionOutcome:
    """Outcome of a trading decision."""
    decision_id: str
    execution_price: float
    actual_return: float
    pnl: float
    exit_reason: str
    hold_duration_minutes: int
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None


@dataclass
class RetrainingTrigger:
    """Trigger conditions for retraining."""
    trigger_id: str
    model_name: str
    trigger_type: str  # DRIFT, PERFORMANCE, SCHEDULED, MANUAL
    trigger_time: datetime
    trigger_reason: str
    metrics: Dict[str, float]
    priority: int  # 1=LOW, 2=MEDIUM, 3=HIGH, 4=CRITICAL


class ContinuousTrainingOrchestrator:
    """
    Orchestrates continuous training and self-improvement.
    - Logs all trading decisions
    - Tracks outcomes
    - Monitors model performance
    - Triggers retraining when needed
    - Manages training schedules
    """
    
    def __init__(self):
        self.db_pool: Optional[asyncpg.Pool] = None
        self.cache = get_trading_cache()
        self.decision_buffer: List[TradingDecision] = []
        self.outcome_buffer: List[DecisionOutcome] = []
        self.buffer_flush_interval = 60  # Flush every 60 seconds
        self.performance_check_interval = 300  # Check performance every 5 minutes
        self.running = False
        
    async def initialize(self):
        """Initialize database connection."""
        try:
            # Get DATABASE_URL from environment or construct from settings
            import os
            db_url = os.getenv('DATABASE_URL', '').replace('+asyncpg', '')
            if not db_url:
                # Fallback to constructing from settings
                db_url = f"postgresql://{settings.database.postgres_user}:{settings.database.postgres_password}@{settings.database.postgres_host}:{settings.database.postgres_port}/{settings.database.postgres_database}"
            
            self.db_pool = await asyncpg.create_pool(
                db_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("Continuous training orchestrator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    async def log_decision(self, decision: TradingDecision):
        """Log a trading decision for future training."""
        try:
            self.decision_buffer.append(decision)
            
            # Also cache for quick access
            await self.cache.setex(
                f"decision:{decision.decision_id}",
                3600,  # 1 hour TTL
                json.dumps(asdict(decision), default=str)
            )
            
            logger.debug(f"Logged decision {decision.decision_id} for {decision.symbol}")
            
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
    
    async def log_outcome(self, outcome: DecisionOutcome):
        """Log the outcome of a trading decision."""
        try:
            self.outcome_buffer.append(outcome)
            
            # Update cached decision
            cached_decision = await self.cache.get(f"decision:{outcome.decision_id}")
            if cached_decision:
                decision_data = json.loads(cached_decision)
                decision_data['outcome'] = asdict(outcome)
                await self.cache.setex(
                    f"decision:{outcome.decision_id}",
                    3600,
                    json.dumps(decision_data, default=str)
                )
            
            logger.debug(f"Logged outcome for decision {outcome.decision_id}: PnL={outcome.pnl:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to log outcome: {e}")
    
    async def flush_buffers(self):
        """Flush decision and outcome buffers to database."""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                # Flush decisions
                if self.decision_buffer:
                    logger.info(f"Flushing {len(self.decision_buffer)} decisions to database")
                    
                    for decision in self.decision_buffer:
                        await conn.execute("""
                            INSERT INTO trading_decisions (
                                decision_id, timestamp, symbol, action, confidence,
                                position_size, entry_price, target_price, stop_loss,
                                predicted_return, predicted_direction, prediction_horizon_minutes,
                                features_json, model_name, model_version, model_ensemble,
                                market_regime, volatility_level, portfolio_exposure, metadata
                            ) VALUES ($1, NOW(), $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                            ON CONFLICT (decision_id) DO NOTHING
                        """,
                            decision.decision_id, decision.symbol, decision.action,
                            decision.confidence, decision.position_size, decision.entry_price,
                            decision.target_price, decision.stop_loss, decision.predicted_return,
                            decision.predicted_direction, decision.prediction_horizon_minutes,
                            json.dumps(decision.features), decision.model_name, decision.model_version,
                            json.dumps(decision.model_ensemble), decision.market_regime,
                            decision.volatility_level, decision.portfolio_exposure,
                            json.dumps(decision.metadata, default=str)
                        )
                    
                    self.decision_buffer.clear()
                
                # Flush outcomes
                if self.outcome_buffer:
                    logger.info(f"Flushing {len(self.outcome_buffer)} outcomes to database")
                    
                    for outcome in self.outcome_buffer:
                        await conn.execute("""
                            SELECT update_decision_outcome($1, $2, $3, $4, $5, $6)
                        """,
                            outcome.decision_id, outcome.execution_price, outcome.actual_return,
                            outcome.pnl, outcome.exit_reason, outcome.hold_duration_minutes
                        )
                        
                        if outcome.max_favorable_excursion is not None:
                            await conn.execute("""
                                UPDATE trading_decisions 
                                SET max_favorable_excursion = $2,
                                    max_adverse_excursion = $3
                                WHERE decision_id = $1
                            """, outcome.decision_id, outcome.max_favorable_excursion, 
                                outcome.max_adverse_excursion)
                    
                    self.outcome_buffer.clear()
                    
        except Exception as e:
            logger.error(f"Failed to flush buffers: {e}")
    
    async def check_model_performance(self):
        """Check model performance and trigger retraining if needed."""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get all active models
                models = await conn.fetch("""
                    SELECT model_name, version FROM model_registry 
                    WHERE is_active = true
                """)
                
                for model in models:
                    model_name = model['model_name']
                    model_version = model['version']
                    
                    # Calculate recent performance (last 24 hours)
                    perf = await conn.fetchrow("""
                        SELECT * FROM calculate_model_performance($1, $2, $3, $4)
                    """, model_name, model_version, 
                        datetime.now() - timedelta(hours=24), datetime.now())
                    
                    if not perf or perf['total_decisions'] < 10:
                        continue
                    
                    # Log performance metrics
                    await conn.execute("""
                        INSERT INTO model_performance_metrics (
                            model_name, model_version, evaluation_period,
                            period_start, period_end, total_predictions,
                            correct_predictions, accuracy, total_trades,
                            winning_trades, losing_trades, win_rate,
                            total_pnl, sharpe_ratio
                        ) VALUES ($1, $2, 'HOURLY', $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """,
                        model_name, model_version,
                        datetime.now() - timedelta(hours=24), datetime.now(),
                        perf['total_decisions'], perf['executed_decisions'],
                        perf['accuracy'], perf['executed_decisions'],
                        int(perf['executed_decisions'] * perf['win_rate']),
                        int(perf['executed_decisions'] * (1 - perf['win_rate'])),
                        perf['win_rate'], perf['total_pnl'], perf['sharpe']
                    )
                    
                    # Check if retraining is needed
                    await self._check_retraining_triggers(
                        conn, model_name, model_version, perf
                    )
                    
                    logger.info(
                        f"Model {model_name} v{model_version}: "
                        f"Accuracy={perf['accuracy']:.2%}, "
                        f"WinRate={perf['win_rate']:.2%}, "
                        f"PnL=${perf['total_pnl']:.2f}, "
                        f"Sharpe={perf['sharpe']:.2f}"
                    )
                    
        except Exception as e:
            logger.error(f"Failed to check model performance: {e}")
    
    async def _check_retraining_triggers(
        self, 
        conn: asyncpg.Connection, 
        model_name: str, 
        model_version: str,
        current_perf: Dict
    ):
        """Check if model needs retraining based on performance."""
        try:
            # Get retraining schedule for this model
            schedule = await conn.fetchrow("""
                SELECT * FROM retraining_schedule 
                WHERE model_name = $1 AND enabled = true
                LIMIT 1
            """, model_name)
            
            if not schedule:
                return
            
            triggers = []
            
            # Check performance drop
            if schedule['min_performance_drop']:
                # Get baseline performance (first week)
                baseline = await conn.fetchrow("""
                    SELECT AVG(accuracy) as baseline_accuracy, AVG(sharpe_ratio) as baseline_sharpe
                    FROM model_performance_metrics
                    WHERE model_name = $1 AND model_version = $2
                    AND timestamp >= created_at
                    AND timestamp <= created_at + interval '7 days'
                """, model_name, model_version)
                
                if baseline and baseline['baseline_accuracy']:
                    accuracy_drop = (baseline['baseline_accuracy'] - current_perf['accuracy']) / baseline['baseline_accuracy']
                    
                    if accuracy_drop > schedule['min_performance_drop']:
                        triggers.append(RetrainingTrigger(
                            trigger_id=hashlib.md5(f"{model_name}_{datetime.now()}".encode()).hexdigest(),
                            model_name=model_name,
                            trigger_type='PERFORMANCE',
                            trigger_time=datetime.now(),
                            trigger_reason=f"Accuracy dropped {accuracy_drop:.1%} below baseline",
                            metrics={
                                'baseline_accuracy': baseline['baseline_accuracy'],
                                'current_accuracy': current_perf['accuracy'],
                                'drop_percentage': accuracy_drop
                            },
                            priority=3 if accuracy_drop > 0.2 else 2
                        ))
            
            # Check drift score
            if schedule['min_drift_score']:
                drift_reports = await conn.fetchrow("""
                    SELECT MAX(drift_score) as max_drift
                    FROM model_drift_reports
                    WHERE model_name = $1 AND model_version = $2
                    AND timestamp >= NOW() - interval '24 hours'
                """, model_name, model_version)
                
                if drift_reports and drift_reports['max_drift'] and drift_reports['max_drift'] > schedule['min_drift_score']:
                    triggers.append(RetrainingTrigger(
                        trigger_id=hashlib.md5(f"{model_name}_drift_{datetime.now()}".encode()).hexdigest(),
                        model_name=model_name,
                        trigger_type='DRIFT',
                        trigger_time=datetime.now(),
                        trigger_reason=f"Drift score {drift_reports['max_drift']:.2f} exceeded threshold",
                        metrics={'drift_score': drift_reports['max_drift']},
                        priority=4  # Critical
                    ))
            
            # Check new samples threshold
            if schedule['min_new_samples']:
                new_samples = await conn.fetchval("""
                    SELECT COUNT(*) FROM trading_decisions
                    WHERE model_name = $1 AND model_version = $2
                    AND timestamp > COALESCE(
                        (SELECT last_training_timestamp FROM retraining_schedule WHERE model_name = $1),
                        NOW() - interval '30 days'
                    )
                """, model_name, model_version)
                
                if new_samples >= schedule['min_new_samples']:
                    triggers.append(RetrainingTrigger(
                        trigger_id=hashlib.md5(f"{model_name}_samples_{datetime.now()}".encode()).hexdigest(),
                        model_name=model_name,
                        trigger_type='SCHEDULED',
                        trigger_time=datetime.now(),
                        trigger_reason=f"{new_samples} new samples available for training",
                        metrics={'new_samples': new_samples},
                        priority=1
                    ))
            
            # Process triggers
            for trigger in triggers:
                await self._process_retraining_trigger(conn, trigger, schedule)
                
        except Exception as e:
            logger.error(f"Failed to check retraining triggers: {e}")
    
    async def _process_retraining_trigger(
        self,
        conn: asyncpg.Connection,
        trigger: RetrainingTrigger,
        schedule: Dict
    ):
        """Process a retraining trigger."""
        try:
            logger.warning(
                f"Retraining trigger for {trigger.model_name}: "
                f"{trigger.trigger_type} - {trigger.trigger_reason} (Priority: {trigger.priority})"
            )
            
            # Check if we're in off-hours (preferred retraining time)
            now = datetime.now()
            is_off_hours = self._is_off_hours(now)
            
            # High priority triggers can run anytime, lower priority wait for off-hours
            if trigger.priority >= 3 or is_off_hours:
                # Cache trigger for the training service to pick up
                await self.cache.lpush(
                    "training:triggers",
                    json.dumps(asdict(trigger), default=str)
                )
                
                # Update schedule
                await conn.execute("""
                    UPDATE retraining_schedule
                    SET next_scheduled_run = NOW()
                    WHERE model_name = $1
                """, trigger.model_name)
                
                logger.info(f"Queued retraining for {trigger.model_name}")
            else:
                # Schedule for next off-hours period
                next_run = self._next_off_hours_time(now)
                await conn.execute("""
                    UPDATE retraining_schedule
                    SET next_scheduled_run = $2
                    WHERE model_name = $1
                """, trigger.model_name, next_run)
                
                logger.info(f"Scheduled retraining for {trigger.model_name} at {next_run}")
                
        except Exception as e:
            logger.error(f"Failed to process retraining trigger: {e}")
    
    def _is_off_hours(self, dt: datetime) -> bool:
        """Check if current time is in off-hours (market closed)."""
        # Market hours: 9:30 AM - 4:00 PM ET weekdays
        # Off-hours: Evenings, nights, weekends
        
        # Convert to ET
        # For simplicity, assume we're already in ET or check system timezone
        hour = dt.hour
        weekday = dt.weekday()
        
        # Weekend
        if weekday >= 5:  # Saturday=5, Sunday=6
            return True
        
        # Weekday evening/night (after 6 PM or before 9 AM)
        if hour >= 18 or hour < 9:
            return True
        
        # During lunch hour (12-1 PM) is acceptable for quick retraining
        if 12 <= hour < 13:
            return True
        
        return False
    
    def _next_off_hours_time(self, dt: datetime) -> datetime:
        """Calculate next off-hours time."""
        hour = dt.hour
        weekday = dt.weekday()
        
        # If Friday evening, wait until Saturday morning
        if weekday == 4 and hour >= 16:
            return dt.replace(hour=2, minute=0, second=0) + timedelta(days=1)
        
        # If weekend, use early morning
        if weekday >= 5:
            return dt.replace(hour=2, minute=0, second=0) + timedelta(days=1)
        
        # If before market open, use this evening
        if hour < 9:
            return dt.replace(hour=20, minute=0, second=0)
        
        # Otherwise use tonight
        return dt.replace(hour=20, minute=0, second=0)
    
    async def run(self):
        """Main run loop."""
        self.running = True
        await self.initialize()
        
        last_flush = datetime.now()
        last_perf_check = datetime.now()
        
        logger.info("Continuous training orchestrator started")
        
        try:
            while self.running:
                now = datetime.now()
                
                # Flush buffers periodically
                if (now - last_flush).total_seconds() >= self.buffer_flush_interval:
                    await self.flush_buffers()
                    last_flush = now
                
                # Check performance periodically
                if (now - last_perf_check).total_seconds() >= self.performance_check_interval:
                    await self.check_model_performance()
                    last_perf_check = now
                
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in continuous training orchestrator: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Final flush
            await self.flush_buffers()
            
            if self.db_pool:
                await self.db_pool.close()
            
            logger.info("Continuous training orchestrator stopped")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Singleton instance
_orchestrator: Optional[ContinuousTrainingOrchestrator] = None

def get_orchestrator() -> ContinuousTrainingOrchestrator:
    """Get singleton orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ContinuousTrainingOrchestrator()
    return _orchestrator


async def main():
    """Main entry point."""
    orchestrator = get_orchestrator()
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
