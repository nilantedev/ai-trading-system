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
    STAGING = "staging"
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
        # Core components (will be initialized)
        self.training_service = None
        self.improvement_engine = None
        self.rl_engine = None
        self.intelligence_coordinator = None
        self.regime_detector = None
        self.cache = None
        
        # Model tracking
        self.model_registry: Dict[str, ModelLifecycle] = {}
        self.active_models: Set[str] = set()
        self.model_performance: Dict[str, List[float]] = {}
        
        # Scheduling
        self.learning_schedule: List[LearningSchedule] = []
        self.is_running = False
        
        # Configuration
        self.config = {
            'auto_train_interval': timedelta(days=7),  # Weekly retraining
            'auto_improve_interval': timedelta(days=1),  # Daily improvement
            'min_performance_threshold': 0.6,  # Minimum acceptable performance
            'max_concurrent_training': 3,
            'enable_continuous_learning': True,
            'enable_reinforcement_learning': True,
            'enable_ensemble_learning': True
        }
        
        # Performance tracking
        self.total_models_trained = 0
        self.total_improvements = 0
        self.best_performing_models = {}
        
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
        
        # Start orchestration
        self.is_running = True
        asyncio.create_task(self._orchestration_loop())
        asyncio.create_task(self._performance_monitor())
        asyncio.create_task(self._schedule_manager())
        
        logger.info("ML Orchestrator initialized successfully")
    
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
                    
                    # Get recent performance
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
        
        if performance >= self.config['min_performance_threshold']:
            lifecycle.status = ModelStatus.STAGING
            logger.info(f"Model {model_id} validated successfully: {performance:.3f}")
        else:
            lifecycle.status = ModelStatus.TRAINING  # Need more training
            logger.warning(f"Model {model_id} validation failed: {performance:.3f}")
    
    async def _deploy_model(self, model_id: str):
        """Deploy a model to production."""
        lifecycle = self.model_registry.get(model_id)
        if not lifecycle:
            return
        
        lifecycle.status = ModelStatus.PRODUCTION
        lifecycle.production_deployed = datetime.utcnow()
        self.active_models.add(model_id)
        
        logger.info(f"Deployed model {model_id} to production")
    
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
    
    async def _get_model_performance(self, model_id: str) -> float:
        """Get current performance score for a model."""
        # This would fetch actual performance metrics
        # For now, return simulated performance
        return np.random.uniform(0.5, 0.9)
    
    async def _check_drift(self, model_id: str) -> bool:
        """Check if model has drift."""
        # This would check actual drift metrics
        # For now, return random drift detection
        return np.random.random() < 0.1  # 10% chance of drift
    
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
            'model_performance': {
                model_id: scores[-1] if scores else 0
                for model_id, scores in self.model_performance.items()
            }
        }
    
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


# Global orchestrator instance
ml_orchestrator: Optional[MLOrchestrator] = None


async def get_ml_orchestrator() -> MLOrchestrator:
    """Get or create ML orchestrator instance."""
    global ml_orchestrator
    if ml_orchestrator is None:
        ml_orchestrator = MLOrchestrator()
        await ml_orchestrator.initialize()
    return ml_orchestrator