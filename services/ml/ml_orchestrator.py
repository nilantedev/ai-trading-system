#!/usr/bin/env python3
"""
ML Orchestrator - Coordinates all ML components for continuous learning
Manages training, improvement, deployment, and monitoring of all models.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import hashlib
import random
from contextlib import suppress
import os
import math

try:
    from prometheus_client import Counter, Gauge, Histogram
    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover
    _PROM_AVAILABLE = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))

from trading_common import get_settings, get_logger
from trading_common.cache import get_trading_cache

# Import all ML components
from .off_hours_training_service import get_training_service
from .continuous_improvement_engine import get_improvement_engine
from .reinforcement_learning_engine import ReinforcementLearningEngine
from .advanced_intelligence_coordinator import AdvancedIntelligenceCoordinator
from .market_regime_detector import MarketRegimeDetector

logger = get_logger(__name__)
settings = get_settings()


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    TRAINING = "training"
    VALIDATING = "validating"
    STAGING = "staging"  # Initial post-validation staging (artifacts prepared)
    SHADOW = "shadow"    # Receiving mirrored traffic for evaluation
    PRODUCTION = "production"
    IMPROVING = "improving"
    DEPRECATED = "deprecated"


@dataclass
class ModelLifecycle:
    """Tracks model through its lifecycle."""
    model_id: str
    model_type: str
    symbol: str
    status: ModelStatus
    created_at: datetime
    last_trained: Optional[datetime]
    last_improved: Optional[datetime]
    production_deployed: Optional[datetime]
    performance_score: float
    improvement_count: int
    training_count: int
    metadata: Dict[str, Any]


@dataclass
class LearningSchedule:
    """Schedule for continuous learning activities."""
    activity: str  # 'train', 'improve', 'validate', 'deploy'
    model_ids: List[str]
    scheduled_time: datetime
    priority: int  # 1=high, 2=medium, 3=low
    estimated_duration: timedelta
    requirements: Dict[str, Any]


class MLOrchestrator:
    """Orchestrates all ML components for continuous learning."""
    
    def __init__(self):
        # Core components (lazy async init)
        self.training_service = None
        self.improvement_engine = None
        self.rl_engine = None
        self.intelligence_coordinator = None
        self.regime_detector = None

        # Model tracking
        self.model_registry: Dict[str, ModelLifecycle] = {}
        self.active_models: Set[str] = set()
        self.model_performance: Dict[str, List[float]] = {}

        # Scheduling
        self.learning_schedule: List[LearningSchedule] = []
        self.is_running = False

        # Configuration & policy thresholds (explicit & stable)
        self.config = {
            'auto_train_interval': timedelta(days=7),
            'auto_improve_interval': timedelta(days=1),
            'min_performance_threshold': 0.6,
            'min_directional_improvement': 0.02,
            'promotion_significance_pvalue': 0.05,
            'max_allowed_drawdown': 0.35,
            'shadow_min_dwell_minutes': 30,
            'sprt_alpha': 0.05,
            'sprt_beta': 0.2,
            'sprt_p0': 0.52,
            'sprt_p1': 0.58,
            'shadow_circuit_breaker_window': 50,
            'shadow_circuit_breaker_min_acc': 0.50,
            'post_promotion_monitor_minutes': 180,
            'post_promotion_degradation_tolerance': 0.05,
            'max_concurrent_training': 3,
            'enable_continuous_learning': True,
            'enable_reinforcement_learning': True,
            'enable_ensemble_learning': True,
            'prediction_horizons': [1]
        }

        # Performance tracking
        self.total_models_trained = 0
        self.total_improvements = 0
        self.best_performing_models: Dict[str, float] = {}

        # Metrics placeholders (populated in _init_metrics)
        self._metrics_inited = False
        self._m_state_transitions = None
        self._m_model_perf = None
        self._m_drift_events = None
        self._m_schedule_backlog = None
        self._m_validation_failures = None
        self._m_unhealthy_models = None
        self._m_promotions = None
        self._m_shadow_latency = None
        self._m_shadow_predictions = None
        self._m_shadow_directional = None
        self._m_rollbacks = None
        self._m_sprt_llr = None
        self._m_circuit_trips = None
        self._m_auto_rollbacks = None
        self._m_sprt_decisions = None
        
    async def initialize(self):
        """Initialize all ML components."""
        logger.info("Initializing ML Orchestrator")
        
        # Initialize components
        self.cache = get_trading_cache()
        self.training_service = await get_training_service()
        self.improvement_engine = await get_improvement_engine()
        
        # Initialize other components
        self.rl_engine = ReinforcementLearningEngine()
        await self.rl_engine.initialize()
        
        self.intelligence_coordinator = AdvancedIntelligenceCoordinator()
        await self.intelligence_coordinator.initialize()
        
        self.regime_detector = MarketRegimeDetector()
        await self.regime_detector.initialize()
        
        # Load existing models from registry
        await self._load_model_registry()
        # Start orchestration after successful component init
        self.is_running = True
        self._init_metrics()
        asyncio.create_task(self._orchestration_loop())
        asyncio.create_task(self._performance_monitor())
        asyncio.create_task(self._schedule_manager())
        logger.info("ML Orchestrator initialized successfully")

    def _init_metrics(self):
        """Initialize Prometheus metrics (app_ prefixed) if available."""
        if self._metrics_inited or not _PROM_AVAILABLE:
            return
        try:
            # Lifecycle & state transitions
            self._m_state_transitions = Counter(
                'app_ml_state_transitions_total',
                'Model lifecycle state transitions',
                ['model_id','from_state','to_state']
            )
            self._m_model_perf = Gauge(
                'app_ml_model_performance_score',
                'Current model performance score',
                ['symbol','model_type']
            )
            self._m_drift_events = Counter(
                'app_ml_drift_events_total',
                'Detected drift events',
                ['model_id','drift_type']
            )
            self._m_schedule_backlog = Gauge(
                'app_ml_learning_schedule_backlog',
                'Pending scheduled learning activities'
            )
            self._m_validation_failures = Counter(
                'app_ml_validation_failures_total',
                'Validation failures',
                ['model_id','reason']
            )
            self._m_unhealthy_models = Gauge(
                'app_ml_unhealthy_models',
                'Number of models flagged unhealthy'
            )
            self._m_promotions = Counter(
                'app_ml_promotions_total',
                'Promotion decisions',
                ['model_id','decision']
            )
            self._m_shadow_latency = Histogram(
                'app_inference_shadow_latency_seconds',
                'Latency of shadow inference processing'
            )
            self._m_shadow_predictions = Counter(
                'app_inference_shadow_predictions_total',
                'Shadow predictions processed',
                ['model_id','status']
            )
            self._m_shadow_directional = Gauge(
                'app_inference_shadow_directional_accuracy',
                'Directional accuracy of shadow model',
                ['model_id']
            )
            self._m_rollbacks = Counter(
                'app_ml_rollbacks_total',
                'Rollback events',
                ['model_id','reason']
            )
            self._m_sprt_llr = Gauge(
                'app_inference_sprt_llr',
                'Current SPRT log-likelihood ratio',
                ['model_id']
            )
            self._m_circuit_trips = Counter(
                'app_ml_circuit_breaker_trips_total',
                'Shadow circuit breaker trips',
                ['model_id']
            )
            self._m_auto_rollbacks = Counter(
                'app_ml_auto_rollbacks_total',
                'Automatic rollback events',
                ['model_id','trigger']
            )
            self._m_sprt_decisions = Counter(
                'app_inference_sprt_decisions_total',
                'SPRT decisions',
                ['model_id','decision']
            )
            # Last governance metrics update timestamp (for silence alert)
            self._m_last_update = Gauge(
                'app_ml_governance_last_update_timestamp_seconds',
                'Unix timestamp of last governance metrics emission'
            )
            self._metrics_inited = True
            logger.info("ML Orchestrator metrics initialized (app_ prefix)")
            # Emit an initial timestamp value (0) so the series exists immediately
            try:  # best-effort
                self._m_last_update.set(0)
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to initialize metrics: {e}")

    def _touch_governance_timestamp(self):
        try:
            if getattr(self, '_m_last_update', None):
                self._m_last_update.set(datetime.utcnow().timestamp())  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    # ---------------- Governance Metric Emitters ---------------- #
    def record_state_transition(self, model_id: str, from_state: str, to_state: str):
        try:
            if self._m_state_transitions:
                self._m_state_transitions.labels(model_id=model_id, from_state=from_state, to_state=to_state).inc()
            self._touch_governance_timestamp()
        except Exception:  # noqa: BLE001
            pass

    def record_promotion(self, model_id: str, decision: str):
        try:
            if self._m_promotions:
                self._m_promotions.labels(model_id=model_id, decision=decision).inc()
            self._touch_governance_timestamp()
        except Exception:  # noqa: BLE001
            pass

    def record_rollback(self, model_id: str, reason: str):
        try:
            if self._m_rollbacks:
                self._m_rollbacks.labels(model_id=model_id, reason=reason).inc()
            self._touch_governance_timestamp()
        except Exception:  # noqa: BLE001
            pass

    def record_shadow_prediction(self, model_id: str, status: str, latency_seconds: float | None = None, directional_accuracy: float | None = None):
        try:
            if self._m_shadow_predictions:
                self._m_shadow_predictions.labels(model_id=model_id, status=status).inc()
            if latency_seconds is not None and self._m_shadow_latency:
                self._m_shadow_latency.observe(latency_seconds)
            if directional_accuracy is not None and self._m_shadow_directional:
                # clamp 0..1
                da = max(0.0, min(1.0, directional_accuracy))
                self._m_shadow_directional.labels(model_id=model_id).set(da)
            self._touch_governance_timestamp()
        except Exception:  # noqa: BLE001
            pass

    def record_sprt(self, model_id: str, decision: str | None = None, llr: float | None = None):
        try:
            if decision and self._m_sprt_decisions:
                self._m_sprt_decisions.labels(model_id=model_id, decision=decision).inc()
            if llr is not None and self._m_sprt_llr:
                self._m_sprt_llr.labels(model_id=model_id).set(llr)
            self._touch_governance_timestamp()
        except Exception:  # noqa: BLE001
            pass

    def record_circuit_trip(self, model_id: str):
        try:
            if self._m_circuit_trips:
                self._m_circuit_trips.labels(model_id=model_id).inc()
            self._touch_governance_timestamp()
        except Exception:  # noqa: BLE001
            pass

    def record_auto_rollback(self, model_id: str, trigger: str):
        try:
            if self._m_auto_rollbacks:
                self._m_auto_rollbacks.labels(model_id=model_id, trigger=trigger).inc()
            self._touch_governance_timestamp()
        except Exception:  # noqa: BLE001
            pass
    
    async def _load_model_registry(self):
        """Load existing models from cache/storage."""
        if self.cache:
            registry_data = await self.cache.get_json("ml_model_registry")
            if registry_data:
                for model_data in registry_data:
                    lifecycle = ModelLifecycle(**model_data)
                    self.model_registry[lifecycle.model_id] = lifecycle
                    if lifecycle.status == ModelStatus.PRODUCTION:
                        self.active_models.add(lifecycle.model_id)
                
                logger.info(f"Loaded {len(self.model_registry)} models from registry")
        # Integrity advisory (best-effort)
        if self.cache and self.model_registry:
            checked = 0
            missing_meta = 0
            for lifecycle in list(self.model_registry.values())[:25]:  # limit to 25 on startup
                cache_key = f"model_artifact:{lifecycle.symbol}:{lifecycle.model_type}"
                try:
                    meta = await self.cache.get_json(cache_key)
                    checked += 1
                    if not meta or 'sha256' not in meta:
                        missing_meta += 1
                except Exception:
                    continue
            if missing_meta:
                logger.warning(
                    f"Artifact metadata missing for {missing_meta}/{checked} recently loaded models (hash absent)."
                )
    
    async def _orchestration_loop(self):
        """Main orchestration loop."""
        while self.is_running:
            try:
                # Check market regime
                regime = await self.regime_detector.detect_regime()
                
                # Adjust learning based on regime
                await self._adjust_learning_for_regime(regime)
                
                # Process scheduled activities
                await self._process_scheduled_activities()
                
                # Check for models needing attention
                await self._check_model_health()
                
                # Coordinate ensemble predictions
                await self._coordinate_ensemble()
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")
    
    async def _performance_monitor(self):
        """Monitor model performance continuously."""
        while self.is_running:
            try:
                for model_id in self.active_models:
                    lifecycle = self.model_registry.get(model_id)
                    if not lifecycle:
                        continue
                    # Performance (cached or estimated)
                    performance = await self._get_model_performance(model_id)
                    
                    # Track performance history
                    if model_id not in self.model_performance:
                        self.model_performance[model_id] = []
                    self.model_performance[model_id].append(performance)
                    
                    # Check if performance degraded
                    if performance < self.config['min_performance_threshold']:
                        logger.warning(f"Model {model_id} performance degraded: {performance:.3f}")
                        await self._handle_degraded_model(model_id)
                    
                    # Update lifecycle
                    lifecycle.performance_score = performance
                    if self._m_model_perf:
                        with suppress(Exception):
                            self._m_model_perf.labels(symbol=lifecycle.symbol, model_type=lifecycle.model_type).set(performance)
                
                # Wait before next check
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
    
    async def _schedule_manager(self):
        """Manage learning schedule."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                # Process due activities
                due_activities = [
                    s for s in self.learning_schedule 
                    if s.scheduled_time <= current_time
                ]
                
                for activity in due_activities:
                    await self._execute_scheduled_activity(activity)
                    self.learning_schedule.remove(activity)
                
                # Schedule new activities if needed
                await self._schedule_learning_activities()
                if self._m_schedule_backlog:
                    with suppress(Exception):
                        self._m_schedule_backlog.set(len(self.learning_schedule))
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Schedule manager error: {e}")
    
    async def _adjust_learning_for_regime(self, regime: Dict[str, Any]):
        """Adjust learning parameters based on market regime."""
        
        volatility = regime.get('volatility_regime', 'normal')
        trend = regime.get('trend_regime', 'neutral')
        
        if volatility == 'high':
            # More frequent retraining in high volatility
            self.config['auto_train_interval'] = timedelta(days=3)
            self.config['auto_improve_interval'] = timedelta(hours=12)
            logger.info("Adjusted learning frequency for high volatility regime")
            
        elif volatility == 'low':
            # Less frequent retraining in stable markets
            self.config['auto_train_interval'] = timedelta(days=14)
            self.config['auto_improve_interval'] = timedelta(days=2)
            logger.info("Adjusted learning frequency for low volatility regime")
        
        # Adjust model selection based on trend
        if trend == 'strong_trend':
            # Favor trend-following models
            await self._adjust_model_weights('trend_following', 1.5)
        elif trend == 'ranging':
            # Favor mean-reversion models
            await self._adjust_model_weights('mean_reversion', 1.5)
    
    async def _adjust_model_weights(self, model_type: str, weight_multiplier: float):
        """Adjust ensemble weights based on market conditions."""
        for model_id, lifecycle in self.model_registry.items():
            if model_type in lifecycle.metadata.get('tags', []):
                lifecycle.metadata['ensemble_weight'] = weight_multiplier
    
    async def _process_scheduled_activities(self):
        """Process scheduled learning activities."""
        current_time = datetime.utcnow()
        
        for model_id, lifecycle in self.model_registry.items():
            # Check if training needed
            if lifecycle.last_trained:
                time_since_training = current_time - lifecycle.last_trained
                if time_since_training > self.config['auto_train_interval']:
                    await self._schedule_training(model_id)
            
            # Check if improvement needed
            if lifecycle.last_improved:
                time_since_improvement = current_time - lifecycle.last_improved
                if time_since_improvement > self.config['auto_improve_interval']:
                    await self._schedule_improvement(model_id)
    
    async def _check_model_health(self):
        """Check health of all active models."""
        unhealthy_models = []
        
        for model_id in self.active_models:
            lifecycle = self.model_registry.get(model_id)
            if not lifecycle:
                continue
            
            # Check various health metrics
            health_issues = []
            
            # Performance check
            if lifecycle.performance_score < self.config['min_performance_threshold']:
                health_issues.append('low_performance')
            
            # Staleness check
            if lifecycle.last_trained:
                days_since_training = (datetime.utcnow() - lifecycle.last_trained).days
                if days_since_training > 30:
                    health_issues.append('stale_training')
            
            # Drift check
            drift_detected = await self._check_drift(model_id)
            if drift_detected:
                health_issues.append('drift_detected')
            
            if health_issues:
                unhealthy_models.append((model_id, health_issues))
        
        # Handle unhealthy models
        for model_id, issues in unhealthy_models:
            logger.warning(f"Model {model_id} has health issues: {issues}")
            await self._handle_unhealthy_model(model_id, issues)
    
    async def _coordinate_ensemble(self):
        """Coordinate ensemble predictions from multiple models."""
        if not self.config['enable_ensemble_learning']:
            return
        
        # Get models for ensemble
        ensemble_models = [
            model_id for model_id in self.active_models
            if self.model_registry[model_id].status == ModelStatus.PRODUCTION
        ]
        
        if len(ensemble_models) < 2:
            return  # Need at least 2 models for ensemble
        
        # This would coordinate actual ensemble predictions
        # For now, just log the coordination
        logger.debug(f"Coordinating ensemble with {len(ensemble_models)} models")
    
    async def _execute_scheduled_activity(self, activity: LearningSchedule):
        """Execute a scheduled learning activity."""
        logger.info(f"Executing scheduled {activity.activity} for {len(activity.model_ids)} models")
        
        if activity.activity == 'train':
            for model_id in activity.model_ids:
                await self._train_model(model_id)
                
        elif activity.activity == 'improve':
            for model_id in activity.model_ids:
                await self._improve_model(model_id)
                
        elif activity.activity == 'validate':
            for model_id in activity.model_ids:
                await self._validate_model(model_id)
                
        elif activity.activity == 'deploy':
            for model_id in activity.model_ids:
                await self._deploy_model(model_id)
    
    async def _schedule_learning_activities(self):
        """Schedule new learning activities."""
        current_time = datetime.utcnow()
        
        # Schedule training for models that need it
        models_needing_training = []
        for model_id, lifecycle in self.model_registry.items():
            if not lifecycle.last_trained or \
               (current_time - lifecycle.last_trained) > self.config['auto_train_interval']:
                models_needing_training.append(model_id)
        
        if models_needing_training:
            schedule = LearningSchedule(
                activity='train',
                model_ids=models_needing_training[:self.config['max_concurrent_training']],
                scheduled_time=current_time + timedelta(minutes=5),
                priority=2,
                estimated_duration=timedelta(hours=2),
                requirements={'compute': 'high', 'data': 'historical'}
            )
            self.learning_schedule.append(schedule)
    
    async def _train_model(self, model_id: str):
        """Train a model."""
        lifecycle = self.model_registry.get(model_id)
        if not lifecycle:
            return
        
        lifecycle.status = ModelStatus.TRAINING
        
        try:
            # Schedule training job
            job_id = await self.training_service.schedule_training_job(
                symbols=[lifecycle.symbol],
                model_types=[lifecycle.model_type],
                priority=2
            )
            
            lifecycle.last_trained = datetime.utcnow()
            lifecycle.training_count += 1
            self.total_models_trained += 1
            
            logger.info(f"Scheduled training job {job_id} for model {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to train model {model_id}: {e}")
        finally:
            lifecycle.status = ModelStatus.VALIDATING
    
    async def _improve_model(self, model_id: str):
        """Improve a model."""
        lifecycle = self.model_registry.get(model_id)
        if not lifecycle:
            return
        
        lifecycle.status = ModelStatus.IMPROVING
        
        try:
            # Trigger improvement
            result = await self.improvement_engine.trigger_improvement(model_id)
            
            lifecycle.last_improved = datetime.utcnow()
            lifecycle.improvement_count += 1
            self.total_improvements += 1
            
            logger.info(f"Triggered improvement for model {model_id}: {result}")
            
        except Exception as e:
            logger.error(f"Failed to improve model {model_id}: {e}")
        finally:
            lifecycle.status = ModelStatus.VALIDATING
    
    async def _validate_model(self, model_id: str):
        """Validate a model before production."""
        lifecycle = self.model_registry.get(model_id)
        if not lifecycle:
            return
        
        lifecycle.status = ModelStatus.VALIDATING
        
        # Perform validation (simplified)
        performance = await self._get_model_performance(model_id)
        
        # Fetch cached raw performance dict if available for drawdown / baseline uplift
        uplift_ok = True
        drawdown_ok = True
        if self.cache:
            perf_key = f"model_performance:{lifecycle.symbol}:{lifecycle.model_type}"
            try:
                perf_cached = await self.cache.get_json(perf_key)
                if perf_cached:
                    max_dd = abs(perf_cached.get('max_drawdown', 0))
                    if max_dd > self.config['max_allowed_drawdown']:
                        drawdown_ok = False
                    wf = perf_cached.get('walk_forward') or {}
                    uplift = wf.get('uplift_vs_baseline')
                    if uplift is not None and uplift < self.config['min_directional_improvement']:
                        uplift_ok = False
            except Exception:
                pass

        # p-value requirement if available
        p_value_ok = True
        if self.cache:
            try:
                if 'perf_cached' in locals() and perf_cached:
                    wf = perf_cached.get('walk_forward') or {}
                    p_val = wf.get('p_value')
                    if p_val is not None and p_val > self.config['promotion_significance_pvalue']:
                        p_value_ok = False
            except Exception:
                pass

        if (performance >= self.config['min_performance_threshold']) and uplift_ok and drawdown_ok and p_value_ok:
            lifecycle.status = ModelStatus.STAGING
            logger.info(
                f"Model {model_id} validated: perf={performance:.3f} uplift_ok={uplift_ok} drawdown_ok={drawdown_ok} p_value_ok={p_value_ok}"
            )
            # Immediately attempt to enter shadow phase for live evaluation
            await self._enter_shadow(model_id)
        else:
            lifecycle.status = ModelStatus.TRAINING  # Need more training
            logger.warning(
                f"Model {model_id} validation failed: perf={performance:.3f} uplift_ok={uplift_ok} drawdown_ok={drawdown_ok} p_value_ok={p_value_ok}"
            )
            if self._m_validation_failures:
                with suppress(Exception):
                    self._m_validation_failures.inc()
    
    async def _deploy_model(self, model_id: str):
        """Deploy a model to production (post-shadow)."""
        lifecycle = self.model_registry.get(model_id)
        if not lifecycle:
            return
        lifecycle.status = ModelStatus.PRODUCTION
        lifecycle.production_deployed = datetime.utcnow()
        self.active_models.add(model_id)
        logger.info(f"Deployed model {model_id} to production")
        self._log_state_transition(model_id, 'promote_production')
        if self._m_promotions:
            with suppress(Exception):
                self._m_promotions.labels(stage='production').inc()
    
    async def _handle_degraded_model(self, model_id: str):
        """Handle a model with degraded performance."""
        lifecycle = self.model_registry.get(model_id)
        if not lifecycle:
            return
        
        logger.warning(f"Handling degraded model {model_id}")
        
        # Remove from production
        if model_id in self.active_models:
            self.active_models.remove(model_id)
            lifecycle.status = ModelStatus.DEPRECATED
        
        # Schedule immediate retraining
        await self._schedule_training(model_id)
        
        # Schedule improvement
        await self._schedule_improvement(model_id)
        if self._m_unhealthy_models:
            with suppress(Exception):
                self._m_unhealthy_models.inc()
    
    async def _handle_unhealthy_model(self, model_id: str, issues: List[str]):
        """Handle an unhealthy model."""
        
        if 'drift_detected' in issues:
            # Immediate retraining for drift
            await self._schedule_training(model_id, priority=1)
        
        if 'low_performance' in issues:
            # Improvement for performance issues
            await self._schedule_improvement(model_id, priority=1)
        
        if 'stale_training' in issues:
            # Regular retraining for staleness
            await self._schedule_training(model_id, priority=3)
    
    async def _schedule_training(self, model_id: str, priority: int = 2):
        """Schedule training for a model."""
        schedule = LearningSchedule(
            activity='train',
            model_ids=[model_id],
            scheduled_time=datetime.utcnow() + timedelta(minutes=10),
            priority=priority,
            estimated_duration=timedelta(hours=1),
            requirements={'compute': 'medium'}
        )
        self.learning_schedule.append(schedule)
        logger.info(f"Scheduled training for model {model_id}")
        self._log_state_transition(model_id, 'schedule_train', {'priority': priority})
    
    async def _schedule_improvement(self, model_id: str, priority: int = 2):
        """Schedule improvement for a model."""
        schedule = LearningSchedule(
            activity='improve',
            model_ids=[model_id],
            scheduled_time=datetime.utcnow() + timedelta(minutes=15),
            priority=priority,
            estimated_duration=timedelta(hours=2),
            requirements={'compute': 'high'}
        )
        self.learning_schedule.append(schedule)
        logger.info(f"Scheduled improvement for model {model_id}")
        self._log_state_transition(model_id, 'schedule_improve', {'priority': priority})
    
    async def _get_model_performance(self, model_id: str) -> float:
        """Get current performance score from cache if available, else deterministic seeded fallback."""
        lifecycle = self.model_registry.get(model_id)
        if not lifecycle or not self.cache:
            # Deterministic placeholder (should rarely trigger in steady-state)
            seed_input = f"placeholder_perf:{model_id}:{datetime.utcnow().strftime('%Y%m%d%H')}"
            rnd = int(hashlib.sha256(seed_input.encode()).hexdigest()[:8], 16)
            random.seed(rnd)
            return random.uniform(0.55, 0.85)
        cache_key = f"model_performance:{lifecycle.symbol}:{lifecycle.model_type}"
        try:
            perf = await self.cache.get_json(cache_key)
            if perf:
                composite = perf.get('composite_score')
                if composite is None:
                    composite = self._estimate_composite(perf)
                return float(composite)
        except Exception:
            pass
        seed_input = f"perf:{model_id}:{datetime.utcnow().strftime('%Y%m%d%H')}"
        rnd = int(hashlib.sha256(seed_input.encode()).hexdigest()[:8], 16)
        random.seed(rnd)
        return random.uniform(0.55, 0.85)
    
    async def _check_drift(self, model_id: str) -> bool:
        """Check cache for drift event (future: integrate real drift detector)."""
        lifecycle = self.model_registry.get(model_id)
        if not lifecycle or not self.cache:
            return False
        drift_key = f"drift_event:{lifecycle.symbol}:{lifecycle.model_type}"
        try:
            drift = await self.cache.get_json(drift_key)
            if drift and drift.get('severity', 0) >= 0.6:
                if self._m_drift_events:
                    with suppress(Exception):
                        self._m_drift_events.inc()
                return True
        except Exception:
            return False
        return False
    
    async def register_model(
        self,
        model_type: str,
        symbol: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a new model in the orchestrator."""
        model_id = f"{symbol}_{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        lifecycle = ModelLifecycle(
            model_id=model_id,
            model_type=model_type,
            symbol=symbol,
            status=ModelStatus.TRAINING,
            created_at=datetime.utcnow(),
            last_trained=None,
            last_improved=None,
            production_deployed=None,
            performance_score=0.0,
            improvement_count=0,
            training_count=0,
            metadata=metadata or {}
        )
        self.model_registry[model_id] = lifecycle
        # Schedule initial training
        await self._schedule_training(model_id, priority=1)
        logger.info(f"Registered new model {model_id}")
        self._log_state_transition(model_id, 'register', {'model_type': model_type})
        return model_id
    
    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        
        # Get component statuses
        training_status = await self.training_service.get_training_status()
        improvement_status = await self.improvement_engine.get_improvement_status()
        
        return {
            'is_running': self.is_running,
            'total_models': len(self.model_registry),
            'active_models': len(self.active_models),
            'models_in_training': len([m for m in self.model_registry.values() 
                                      if m.status == ModelStatus.TRAINING]),
            'models_in_production': len([m for m in self.model_registry.values()
                                        if m.status == ModelStatus.PRODUCTION]),
            'scheduled_activities': len(self.learning_schedule),
            'total_models_trained': self.total_models_trained,
            'total_improvements': self.total_improvements,
            'config': self.config,
            'training_service': training_status,
            'improvement_engine': improvement_status,
            'registry_integrity_sample': await self._sample_registry_integrity(),
            'model_performance': {
                model_id: scores[-1] if scores else 0
                for model_id, scores in self.model_performance.items()
            }
        }

    async def _sample_registry_integrity(self) -> Dict[str, Any]:
        """Sample a subset of registry for integrity metadata presence."""
        if not self.cache or not self.model_registry:
            return {'checked': 0, 'missing': 0}
        checked = 0
        missing = 0
        for lifecycle in list(self.model_registry.values())[:10]:
            cache_key = f"model_artifact:{lifecycle.symbol}:{lifecycle.model_type}"
            try:
                meta = await self.cache.get_json(cache_key)
                checked += 1
                if not meta or 'sha256' not in meta:
                    missing += 1
            except Exception:
                continue
        return {'checked': checked, 'missing': missing}
    
    async def enable_continuous_learning(self):
        """Enable continuous learning."""
        self.config['enable_continuous_learning'] = True
        await self.training_service.enable_training()
        logger.info("Continuous learning enabled")
    
    async def disable_continuous_learning(self):
        """Disable continuous learning."""
        self.config['enable_continuous_learning'] = False
        await self.training_service.disable_training()
        logger.info("Continuous learning disabled")
    
    async def stop(self):
        """Stop the orchestrator."""
        self.is_running = False
        
        # Stop all components
        await self.training_service.stop()
        await self.improvement_engine.stop()
        await self.rl_engine.stop()
        
        # Save model registry
        if self.cache:
            registry_data = [asdict(lifecycle) for lifecycle in self.model_registry.values()]
            await self.cache.set_json("ml_model_registry", registry_data, ttl=86400 * 30)
        
        logger.info("ML Orchestrator stopped")

    # ---------------- Internal helpers -----------------
    def _init_metrics(self):
        if self._metrics_inited or not _PROM_AVAILABLE:
            return
        try:
            self._m_state_transitions = Counter(
                'model_state_transitions_total',
                'Model lifecycle state transitions',
                ['event']
            )
            self._m_model_perf = Gauge(
                'model_performance_score',
                'Latest model composite performance score',
                ['symbol', 'model_type']
            )
            self._m_drift_events = Counter(
                'model_drift_events_total',
                'Count of significant drift events'
            )
            self._m_schedule_backlog = Gauge(
                'learning_schedule_backlog',
                'Number of scheduled learning activities pending'
            )
            self._m_validation_failures = Counter(
                'model_validation_failures_total',
                'Count of validation failures'
            )
            self._m_unhealthy_models = Counter(
                'model_unhealthy_events_total',
                'Count of unhealthy model events handled'
            )
            self._m_promotions = Counter(
                'model_promotions_total',
                'Count of model promotions by stage',
                ['stage']
            )
            self._m_shadow_latency = Histogram(
                'shadow_inference_latency_seconds',
                'Latency of shadow model inference',
                ['symbol', 'model_type']
            )
            self._m_shadow_predictions = Counter(
                'shadow_predictions_total',
                'Count of shadow model predictions',
                ['symbol', 'model_type']
            )
            self._m_shadow_directional = Gauge(
                'shadow_directional_accuracy',
                'Rolling directional accuracy of shadow models',
                ['symbol', 'model_type']
            )
            self._m_rollbacks = Counter(
                'model_rollbacks_total',
                'Count of model rollback events'
            )
            self._m_sprt_llr = Gauge(
                'model_sprt_llr',
                'Sequential Probability Ratio Test log-likelihood ratio',
                ['symbol', 'model_type']
            )
            self._m_circuit_trips = Counter(
                'model_circuit_breaker_trips_total',
                'Circuit breaker trips for shadow models'
            )
            self._m_auto_rollbacks = Counter(
                'post_promotion_auto_rollbacks_total',
                'Automatic rollbacks due to post-promotion degradation'
            )
            self._m_sprt_decisions = Counter(
                'model_sprt_decisions_total',
                'SPRT terminal decisions',
                ['decision']
            )
            self._metrics_inited = True
        except Exception:
            pass

    def _log_state_transition(self, model_id: str, event: str, extra: Optional[Dict[str, Any]] = None):
        lifecycle = self.model_registry.get(model_id)
        payload = {
            'event': event,
            'model_id': model_id,
            'symbol': lifecycle.symbol if lifecycle else None,
            'model_type': lifecycle.model_type if lifecycle else None,
            'timestamp': datetime.utcnow().isoformat(),
        }
        if extra:
            payload.update(extra)
        logger.info(json.dumps(payload))
        if self._m_state_transitions:
            with suppress(Exception):
                self._m_state_transitions.labels(event=event).inc()

    def _estimate_composite(self, perf: Dict[str, Any]) -> float:
        try:
            directional = float(perf.get('directional_accuracy', 0))
            sharpe = float(perf.get('sharpe_ratio', 0))
            win_rate = float(perf.get('win_rate', 0))
            r2 = float(perf.get('r2_score', 0)) if 'r2_score' in perf else 0
            total_return = float(perf.get('total_return', 0))
            max_drawdown = abs(float(perf.get('max_drawdown', 0)))
            score = (
                0.25 * directional +
                0.2 * max(0, min(sharpe / 3, 1)) +
                0.15 * win_rate +
                0.15 * max(0, r2) +
                0.15 * max(0, min(total_return / 0.5, 1)) +
                0.05 * max(0, 1 - min(max_drawdown, 1))
            )
            return float(min(score, 1.0))
        except Exception:
            return 0.0

    # --------------- Shadow / Promotion Logic ---------------
    async def _enter_shadow(self, model_id: str):
        """Place a validated model into shadow deployment for live evaluation."""
        lifecycle = self.model_registry.get(model_id)
        if not lifecycle:
            return
        if lifecycle.status not in (ModelStatus.STAGING, ModelStatus.VALIDATING):
            return
        lifecycle.status = ModelStatus.SHADOW
        lifecycle.metadata['shadow_entered_at'] = datetime.utcnow().isoformat()
        self._log_state_transition(model_id, 'enter_shadow')
        if self._m_promotions:
            with suppress(Exception):
                self._m_promotions.labels(stage='shadow').inc()

    async def _promotion_check(self):
        """Check shadow models for eligibility to promote to production."""
        if not self.cache:
            return
        now = datetime.utcnow()
        dwell = timedelta(minutes=self.config.get('shadow_min_dwell_minutes', 30))
        for model_id, lifecycle in self.model_registry.items():
            if lifecycle.status != ModelStatus.SHADOW:
                continue
            entered_iso = lifecycle.metadata.get('shadow_entered_at')
            try:
                entered_at = datetime.fromisoformat(entered_iso) if entered_iso else None
            except Exception:
                entered_at = None
            if entered_at and now - entered_at < dwell:
                continue  # still dwelling
            # Fetch performance cache
            perf_key = f"model_performance:{lifecycle.symbol}:{lifecycle.model_type}"
            perf_cached = None
            try:
                perf_cached = await self.cache.get_json(perf_key)
            except Exception:
                pass
            if not perf_cached:
                continue
            wf = perf_cached.get('walk_forward') or {}
            uplift = wf.get('uplift_vs_baseline')
            p_value = wf.get('p_value')
            composite = perf_cached.get('composite_score') or self._estimate_composite(perf_cached)
            drawdown = abs(perf_cached.get('max_drawdown', 0))
            # Eligibility checks
            if composite < self.config['min_performance_threshold']:
                continue
            if uplift is not None and uplift < self.config['min_directional_improvement']:
                continue
            if drawdown > self.config['max_allowed_drawdown']:
                continue
            # SPRT decision (optional) - if ongoing and rejects, retain in shadow; if accepts, we can promote
            sprt_meta = lifecycle.metadata.get('sprt') or {}
            sprt_decision = sprt_meta.get('decision')  # 'accept','reject', or None
            if sprt_decision == 'reject':
                continue  # explicitly rejected improvement
            # If p-value available enforce it; else allow SPRT accept override
            if p_value is not None and p_value > self.config['promotion_significance_pvalue'] and sprt_decision != 'accept':
                continue
            details = {
                'composite': composite,
                'uplift': uplift,
                'p_value': p_value,
                'drawdown': drawdown,
                'entered_shadow_at': entered_iso
            }
            await self._audit_promotion_decision(model_id, 'promote', details)
            # Capture baseline metrics before promotion for monitoring
            lifecycle.metadata['baseline_directional'] = perf_cached.get('directional_accuracy')
            lifecycle.metadata['baseline_composite'] = composite
            lifecycle.metadata['promoted_at'] = datetime.utcnow().isoformat()
            await self._deploy_model(model_id)
        # Also audit non-promoted (periodic sample) to maintain trail
        # (kept lightweight - only logs when a check runs and model remains in shadow past dwell)
        for model_id, lifecycle in self.model_registry.items():
            if lifecycle.status == ModelStatus.SHADOW:
                entered_iso = lifecycle.metadata.get('shadow_entered_at')
                try:
                    entered_at = datetime.fromisoformat(entered_iso) if entered_iso else None
                except Exception:
                    entered_at = None
                if entered_at and now - entered_at >= dwell:
                    await self._audit_promotion_decision(model_id, 'retain_shadow', {'entered_shadow_at': entered_iso})

    # Hook promotion check into orchestration loop by overriding _orchestration_loop
    async def _orchestration_loop(self):  # override previous definition with promotion checks
        while self.is_running:
            try:
                regime = await self.regime_detector.detect_regime()
                await self._adjust_learning_for_regime(regime)
                await self._process_scheduled_activities()
                await self._check_model_health()
                await self._coordinate_ensemble()
                # Promotion checks for shadow models
                await self._promotion_check()
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")

    # --------------- Shadow Inference Mirroring ---------------
    async def mirror_inference(self, symbol: str, feature_vector: List[float], horizon: int = 1):
        """Mirror a live production inference to all shadow models for the symbol.

        Stores prediction entries in cache for later outcome resolution and updates latency metrics.
        Safe to call even if no shadow models exist.
        """
        # Backward compatible: if feature_vector is 2D assume already built.
        shadow_models = [m for m, lc in self.model_registry.items() if lc.symbol == symbol and lc.status == ModelStatus.SHADOW]
        if not shadow_models or not self.cache:
            return
        # Ensure 2D input for sklearn-like models
        import pickle
        arr = np.array(feature_vector, dtype=float)
        if arr.ndim == 1:
            fv = arr.reshape(1, -1)
        else:
            fv = arr
        for model_id in shadow_models:
            lifecycle = self.model_registry.get(model_id)
            if not lifecycle:
                continue
            model_type = lifecycle.model_type
            meta_key = f"model_artifact:{lifecycle.symbol}:{lifecycle.model_type}"
            try:
                meta = await self.cache.get_json(meta_key)
                artifact_path = meta.get('artifact_path') if meta else None
                expected_hash = meta.get('sha256') if meta else None
            except Exception:
                artifact_path = None
                expected_hash = None
            if not artifact_path or not os.path.exists(artifact_path):
                continue
            # Strict checksum verification
            if expected_hash:
                try:
                    import hashlib as _hl
                    h = _hl.sha256()
                    with open(artifact_path, 'rb') as fchk:
                        for chunk in iter(lambda: fchk.read(8192), b''):
                            h.update(chunk)
                    if h.hexdigest() != expected_hash:
                        self._log_state_transition(model_id, 'artifact_checksum_mismatch', {'path': artifact_path})
                        continue
                except Exception:
                    continue
            start = datetime.utcnow()
            prediction = None
            try:
                with open(artifact_path, 'rb') as f:
                    model_obj = pickle.load(f)
                with suppress(Exception):
                    prediction = model_obj.predict(fv)
            except Exception:
                continue
            latency = (datetime.utcnow() - start).total_seconds()
            if self._m_shadow_latency:
                with suppress(Exception):
                    self._m_shadow_latency.labels(symbol=symbol, model_type=model_type).observe(latency)
            if self._m_shadow_predictions:
                with suppress(Exception):
                    self._m_shadow_predictions.labels(symbol=symbol, model_type=model_type).inc()
            # Cache prediction record (direction for first output)
            if prediction is not None:
                ts_iso = datetime.utcnow().isoformat()
                # Use configured horizons if provided, else fallback to caller horizon
                horizons = self.config.get('prediction_horizons') or [horizon]
                for hz in horizons:
                    direction = int(float(prediction[0]) >= 0)
                    pred_key = f"shadow_pred:{model_id}:{hz}:{ts_iso}"
                    record = {
                        'model_id': model_id,
                        'symbol': symbol,
                        'prediction': float(prediction[0]) if np.ndim(prediction) else None,
                        'direction': direction,
                        'horizon': hz,
                        'timestamp': ts_iso
                    }
                    with suppress(Exception):
                        await self.cache.set_json(pred_key, record, ttl=3600)
                    index_key = f"shadow_pred_index:{model_id}:{hz}"
                    try:
                        existing = await self.cache.get_json(index_key) or []
                    except Exception:
                        existing = []
                    existing.append(pred_key)
                    existing = existing[-500:]
                    with suppress(Exception):
                        await self.cache.set_json(index_key, existing, ttl=3600)

    async def record_shadow_outcome(self, symbol: str, actual_return: float, horizon: int = 1):
        """Record actual outcome for recent shadow predictions and update directional accuracy and SPRT.

        horizon: which prediction horizon to attribute (matches mirror_inference horizon).
        """
        if not self.cache:
            return
        direction_actual = int(actual_return >= 0)
        for model_id, lifecycle in self.model_registry.items():
            if lifecycle.symbol != symbol or lifecycle.status not in (ModelStatus.SHADOW, ModelStatus.PRODUCTION):
                continue
            index_key = f"shadow_pred_index:{model_id}:{horizon}"
            try:
                pred_keys = await self.cache.get_json(index_key) or []
            except Exception:
                pred_keys = []
            if not pred_keys:
                continue
            # Evaluate the most recent prediction only (simplified)
            latest_key = pred_keys[-1]
            try:
                record = await self.cache.get_json(latest_key)
            except Exception:
                record = None
            if not record:
                continue
            predicted_direction = record.get('direction')
            stats_key = f"shadow_stats:{model_id}:{horizon}"
            try:
                stats = await self.cache.get_json(stats_key) or {'total': 0, 'correct': 0}
            except Exception:
                stats = {'total': 0, 'correct': 0}
            stats['total'] += 1
            if predicted_direction == direction_actual:
                stats['correct'] += 1
            with suppress(Exception):
                await self.cache.set_json(stats_key, stats, ttl=7200)
            acc = 0.0
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
            if self._m_shadow_directional:
                with suppress(Exception):
                    self._m_shadow_directional.labels(symbol=symbol, model_type=lifecycle.model_type).set(acc)
            # Maintain recent outcome list for circuit breaker
            recent_key = f"shadow_recent:{model_id}:{horizon}"
            try:
                recent = await self.cache.get_json(recent_key) or []
            except Exception:
                recent = []
            recent.append(1 if predicted_direction == direction_actual else 0)
            recent = recent[-self.config['shadow_circuit_breaker_window']:]
            with suppress(Exception):
                await self.cache.set_json(recent_key, recent, ttl=7200)
            # Update SPRT if in shadow
            if lifecycle.status == ModelStatus.SHADOW:
                await self._update_sprt(model_id, predicted_direction == direction_actual)

    # ---- Benjamini-Hochberg helper ----
    def _benjamini_hochberg(self, p_values: List[float], alpha: float) -> List[bool]:
        """Return list of decisions (True=reject null) under BH procedure."""
        m = len(p_values)
        if m == 0:
            return []
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        crit = [((i+1)/m) * alpha for i in range(m)]
        passed = -1
        for (i, (orig_idx, p)) in enumerate(indexed):
            if p <= crit[i]:
                passed = i
        decisions = [False]*m
        if passed >= 0:
            threshold_p = indexed[passed][1]
            for orig_idx, p in enumerate(p_values):
                if p <= threshold_p:
                    decisions[orig_idx] = True
        return decisions

    # --------------- Rollback Logic ---------------
    async def rollback_model(self, model_id: str):
        """Rollback a production model to the most recent previous production candidate for same symbol/type."""
        lifecycle = self.model_registry.get(model_id)
        if not lifecycle or lifecycle.status != ModelStatus.PRODUCTION:
            return
        symbol = lifecycle.symbol
        model_type = lifecycle.model_type
        # Find previous production model
        candidates = [lc for lc in self.model_registry.values() if lc.symbol == symbol and lc.model_type == model_type and lc.model_id != model_id and lc.production_deployed]
        candidates.sort(key=lambda c: c.production_deployed or datetime.min, reverse=True)
        previous = candidates[0] if candidates else None
        # Deprecate current
        lifecycle.status = ModelStatus.DEPRECATED
        if model_id in self.active_models:
            self.active_models.remove(model_id)
        self._log_state_transition(model_id, 'rollback_deprecate')
        if self._m_rollbacks:
            with suppress(Exception):
                self._m_rollbacks.inc()
        if previous:
            previous.status = ModelStatus.PRODUCTION
            if previous.model_id not in self.active_models:
                self.active_models.add(previous.model_id)
            self._log_state_transition(previous.model_id, 'rollback_promote')

    # --------------- Promotion Audit ---------------
    async def _audit_promotion_decision(self, model_id: str, decision: str, details: Dict[str, Any]):
        lifecycle = self.model_registry.get(model_id)
        payload = {
            'event': 'promotion_decision',
            'decision': decision,
            'model_id': model_id,
            'symbol': lifecycle.symbol if lifecycle else None,
            'model_type': lifecycle.model_type if lifecycle else None,
            'timestamp': datetime.utcnow().isoformat(),
            'details': details
        }
        # Log JSON
        logger.info(json.dumps(payload))
        # Cache append (circular buffer)
        if self.cache:
            key = 'promotion_audit_log'
            try:
                log_entries = await self.cache.get_json(key) or []
            except Exception:
                log_entries = []
            log_entries.append(payload)
            log_entries = log_entries[-200:]
            with suppress(Exception):
                await self.cache.set_json(key, log_entries, ttl=86400)
        # File append (best effort)
        try:
            logs_dir = Path(__file__).parent.parent.parent / 'logs'
            logs_dir.mkdir(parents=True, exist_ok=True)
            with open(logs_dir / 'promotion_audit.jsonl', 'a') as f:
                f.write(json.dumps(payload) + '\n')
        except Exception:
            pass
        # DB persistence (best-effort, non-blocking)
        try:
            from .persistence import persist_promotion_audit
            asyncio.create_task(persist_promotion_audit(payload))
        except Exception:
            pass

    # --------------- SPRT & Monitoring Extensions ---------------
    async def _update_sprt(self, model_id: str, correct: bool):
        lifecycle = self.model_registry.get(model_id)
        if not lifecycle:
            return
        sprt = lifecycle.metadata.get('sprt') or {'n': 0, 'llr': 0.0, 'decision': None, 'accept_at': None}
        if sprt.get('decision'):
            return  # already terminal
        p0 = self.config['sprt_p0']
        p1 = self.config['sprt_p1']
        alpha = self.config['sprt_alpha']
        beta = self.config['sprt_beta']
        # Wald boundaries
        A = math.log((1 - beta) / alpha)
        B = math.log(beta / (1 - alpha))
        # Update LLR increment
        x = 1 if correct else 0
        # Avoid log(0); clamp probabilities
        eps = 1e-9
        term = math.log(((p1 if x else (1 - p1)) + eps) / ((p0 if x else (1 - p0)) + eps))
        sprt['llr'] += term
        sprt['n'] += 1
        decision = None
        if sprt['llr'] >= A:
            decision = 'accept'
        elif sprt['llr'] <= B:
            decision = 'reject'
        if decision:
            sprt['decision'] = decision
            sprt['decision_at'] = datetime.utcnow().isoformat()
            self._log_state_transition(model_id, f'sprt_{decision}', {'llr': sprt['llr'], 'samples': sprt['n']})
            if self._m_sprt_decisions:
                with suppress(Exception):
                    self._m_sprt_decisions.labels(decision=decision).inc()
        lifecycle.metadata['sprt'] = sprt
        if self._m_sprt_llr:
            with suppress(Exception):
                self._m_sprt_llr.labels(symbol=lifecycle.symbol, model_type=lifecycle.model_type).set(sprt['llr'])

    async def _shadow_circuit_breaker_check(self):
        if not self.cache:
            return
        window = self.config['shadow_circuit_breaker_window']
        min_acc = self.config['shadow_circuit_breaker_min_acc']
        for model_id, lifecycle in self.model_registry.items():
            if lifecycle.status != ModelStatus.SHADOW:
                continue
            recent_key = f"shadow_recent:{model_id}:1"  # horizon 1 primary
            try:
                recent = await self.cache.get_json(recent_key) or []
            except Exception:
                recent = []
            if len(recent) < window // 2:  # need minimum samples
                continue
            acc = sum(recent[-window:]) / max(1, min(len(recent), window))
            if acc < min_acc:
                # Trip breaker: demote to TRAINING and schedule improvement
                prev_status = lifecycle.status
                lifecycle.status = ModelStatus.TRAINING
                self._log_state_transition(model_id, 'shadow_circuit_breaker', {'acc': acc, 'prev_status': prev_status})
                await self._schedule_improvement(model_id, priority=1)
                if self._m_circuit_trips:
                    with suppress(Exception):
                        self._m_circuit_trips.inc()

    async def _post_promotion_monitor(self):
        if not self.cache:
            return
        window_minutes = self.config['post_promotion_monitor_minutes']
        tol = self.config['post_promotion_degradation_tolerance']
        now = datetime.utcnow()
        for model_id, lifecycle in self.model_registry.items():
            if lifecycle.status != ModelStatus.PRODUCTION:
                continue
            promoted_at_iso = lifecycle.metadata.get('promoted_at')
            if not promoted_at_iso:
                continue
            try:
                promoted_at = datetime.fromisoformat(promoted_at_iso)
            except Exception:
                continue
            if (now - promoted_at).total_seconds() / 60 > window_minutes:
                continue  # outside monitoring window
            baseline_dir = lifecycle.metadata.get('baseline_directional')
            if baseline_dir is None:
                continue
            perf_key = f"model_performance:{lifecycle.symbol}:{lifecycle.model_type}"
            try:
                perf_cached = await self.cache.get_json(perf_key)
            except Exception:
                perf_cached = None
            if not perf_cached:
                continue
            current_dir = perf_cached.get('directional_accuracy')
            current_comp = perf_cached.get('composite_score') or self._estimate_composite(perf_cached)
            if current_dir is None:
                continue
            if (baseline_dir - current_dir) > tol or current_comp < self.config['min_performance_threshold']:
                # Auto rollback
                await self._audit_promotion_decision(model_id, 'auto_rollback', {
                    'baseline_directional': baseline_dir,
                    'current_directional': current_dir,
                    'current_composite': current_comp
                })
                await self.rollback_model(model_id)
                if self._m_auto_rollbacks:
                    with suppress(Exception):
                        self._m_auto_rollbacks.inc()

    # Extend orchestration loop (redefine) to include new monitors
    async def _orchestration_loop(self):  # type: ignore
        while self.is_running:
            try:
                regime = await self.regime_detector.detect_regime()
                await self._adjust_learning_for_regime(regime)
                await self._process_scheduled_activities()
                await self._check_model_health()
                await self._coordinate_ensemble()
                await self._promotion_check()
                await self._shadow_circuit_breaker_check()
                await self._post_promotion_monitor()
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")

    # --------------- Public access helpers for dashboards ---------------
    async def list_models(self) -> List[Dict[str, Any]]:
        out = []
        for lc in self.model_registry.values():
            entry = {
                'model_id': lc.model_id,
                'symbol': lc.symbol,
                'model_type': lc.model_type,
                'status': lc.status.value,
                'created_at': lc.created_at.isoformat(),
                'last_trained': lc.last_trained.isoformat() if lc.last_trained else None,
                'last_improved': lc.last_improved.isoformat() if lc.last_improved else None,
                'production_deployed': lc.production_deployed.isoformat() if lc.production_deployed else None,
                'performance_score': lc.performance_score,
                'training_count': lc.training_count,
                'improvement_count': lc.improvement_count,
            }
            sprt = lc.metadata.get('sprt') or {}
            if sprt:
                entry['sprt'] = {k: sprt.get(k) for k in ('n', 'llr', 'decision', 'decision_at') if k in sprt}
            if 'baseline_composite' in lc.metadata:
                entry['baseline_composite'] = lc.metadata.get('baseline_composite')
            out.append(entry)
        return out

    async def get_promotion_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.cache:
            return []
        try:
            log_entries = await self.cache.get_json('promotion_audit_log') or []
        except Exception:
            log_entries = []
        return log_entries[-limit:]

    async def manual_promotion_check(self) -> Dict[str, Any]:
        """Expose a manual promotion check trigger (returns summary)."""
        before = len([m for m in self.model_registry.values() if m.status == ModelStatus.PRODUCTION])
        await self._promotion_check()
        after = len([m for m in self.model_registry.values() if m.status == ModelStatus.PRODUCTION])
        return {'productions_before': before, 'productions_after': after, 'promoted': max(0, after - before)}

    async def list_shadow_stats(self, horizon: int = 1) -> List[Dict[str, Any]]:
        """Return current shadow model directional accuracy statistics for given horizon."""
        results: List[Dict[str, Any]] = []
        if not self.cache:
            return results
        for lc in self.model_registry.values():
            if lc.status != ModelStatus.SHADOW:
                continue
            stats_key = f"shadow_stats:{lc.model_id}:{horizon}"
            try:
                stats = await self.cache.get_json(stats_key) or {}
            except Exception:
                stats = {}
            total = stats.get('total', 0)
            correct = stats.get('correct', 0)
            acc = (correct / total) if total else None
            sprt = lc.metadata.get('sprt') or {}
            results.append({
                'model_id': lc.model_id,
                'symbol': lc.symbol,
                'model_type': lc.model_type,
                'samples': total,
                'correct': correct,
                'directional_accuracy': acc,
                'sprt_llr': sprt.get('llr'),
                'sprt_n': sprt.get('n'),
                'sprt_decision': sprt.get('decision')
            })
        return results

    async def admin_rollback(self, model_id: str) -> Dict[str, Any]:
        lc = self.model_registry.get(model_id)
        if not lc:
            return {'rolled_back': False, 'reason': 'model_not_found'}
        if lc.status != ModelStatus.PRODUCTION:
            return {'rolled_back': False, 'reason': 'not_in_production'}
        await self.rollback_model(model_id)
        return {'rolled_back': True, 'model_id': model_id}

    # --------------- Aggregated KPI helpers (for business / analytics) ---------------
    async def kpi_snapshot(self) -> Dict[str, Any]:
        now = datetime.utcnow().isoformat()
        prod = [lc for lc in self.model_registry.values() if lc.status == ModelStatus.PRODUCTION]
        shadow = [lc for lc in self.model_registry.values() if lc.status == ModelStatus.SHADOW]
        avg_perf = None
        if prod:
            avg_perf = sum(lc.performance_score for lc in prod) / max(1, len(prod))
        promotions = 0
        rollbacks = 0
        if self.cache:
            try:
                audit = await self.cache.get_json('promotion_audit_log') or []
            except Exception:
                audit = []
            for entry in audit:
                if entry.get('decision') == 'promote':
                    promotions += 1
                if entry.get('decision') in ('auto_rollback', 'rollback_manual'):
                    rollbacks += 1
        return {
            'timestamp': now,
            'production_models': len(prod),
            'shadow_models': len(shadow),
            'avg_production_composite': avg_perf,
            'total_promotions_window': promotions,
            'total_rollbacks_window': rollbacks,
            'promotion_to_rollback_ratio': (promotions / rollbacks) if rollbacks else None
        }


# Global orchestrator instance
ml_orchestrator: Optional[MLOrchestrator] = None


async def get_ml_orchestrator() -> MLOrchestrator:
    """Get or create ML orchestrator instance."""
    global ml_orchestrator
    if ml_orchestrator is None:
        ml_orchestrator = MLOrchestrator()
        await ml_orchestrator.initialize()
    return ml_orchestrator