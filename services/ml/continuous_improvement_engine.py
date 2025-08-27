#!/usr/bin/env python3
"""
Continuous Improvement Engine - Enhances models and strategies continuously
Integrates with local LLMs for strategy refinement and performance optimization.
"""

import asyncio
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, time
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
from collections import defaultdict

from trading_common import get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.local_swarm import LocalSwarm, LocalAgent, RECOMMENDED_MODELS
from trading_common.ai_models import get_model_router, ModelType

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ModelPerformanceMetrics:
    """Comprehensive model performance tracking."""
    model_id: str
    symbol: str
    period: str  # daily, weekly, monthly
    
    # Accuracy metrics
    directional_accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Trading metrics
    total_trades: int
    profitable_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    value_at_risk: float
    
    # Learning metrics
    prediction_confidence: float
    feature_importance: Dict[str, float]
    error_analysis: Dict[str, Any]
    
    timestamp: datetime


@dataclass
class ImprovementStrategy:
    """Strategy for model improvement."""
    strategy_type: str  # 'feature_engineering', 'hyperparameter', 'ensemble', 'architecture'
    description: str
    expected_improvement: float  # Expected percentage improvement
    confidence: float
    implementation_steps: List[str]
    resources_required: Dict[str, Any]
    priority: int  # 1=high, 2=medium, 3=low


@dataclass
class LearningInsight:
    """Insights from continuous learning."""
    insight_type: str  # 'pattern', 'anomaly', 'opportunity', 'weakness'
    description: str
    affected_models: List[str]
    recommended_actions: List[str]
    evidence: Dict[str, Any]
    confidence: float
    discovered_at: datetime


class PerformanceAnalyzer:
    """Analyzes model performance and identifies improvement opportunities."""
    
    def __init__(self):
        self.performance_history = defaultdict(list)
        self.insights = []
        self.improvement_threshold = 0.6  # Minimum performance for production
        
    async def analyze_model_performance(
        self, 
        model_id: str, 
        predictions: np.ndarray,
        actuals: np.ndarray,
        trades: List[Dict[str, Any]]
    ) -> ModelPerformanceMetrics:
        """Comprehensive performance analysis."""
        
        # Calculate accuracy metrics
        directional_pred = np.sign(predictions)
        directional_actual = np.sign(actuals)
        directional_accuracy = np.mean(directional_pred == directional_actual)
        
        # Trading metrics
        profitable_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        win_rate = len(profitable_trades) / max(len(trades), 1)
        avg_win = np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0
        profit_factor = (avg_win * len(profitable_trades)) / max(avg_loss * len(losing_trades), 0.001)
        
        # Risk metrics
        returns = np.diff(predictions) / predictions[:-1]
        sharpe_ratio = np.mean(returns) / max(np.std(returns), 0.001) * np.sqrt(252)
        
        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0.001
        sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252)
        
        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Value at Risk (95% confidence)
        value_at_risk = np.percentile(returns, 5)
        
        # Error analysis
        errors = predictions - actuals
        error_analysis = {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'max_error': float(np.max(np.abs(errors))),
            'error_skew': float(self._calculate_skew(errors)),
            'error_patterns': self._identify_error_patterns(errors)
        }
        
        metrics = ModelPerformanceMetrics(
            model_id=model_id,
            symbol="",  # Will be set by caller
            period="daily",
            directional_accuracy=directional_accuracy,
            precision=0.0,  # Would calculate properly
            recall=0.0,
            f1_score=0.0,
            total_trades=len(trades),
            profitable_trades=len(profitable_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            value_at_risk=value_at_risk,
            prediction_confidence=0.0,  # Would calculate
            feature_importance={},  # Would extract from model
            error_analysis=error_analysis,
            timestamp=datetime.utcnow()
        )
        
        # Store for trend analysis
        self.performance_history[model_id].append(metrics)
        
        return metrics
    
    def _calculate_skew(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _identify_error_patterns(self, errors: np.ndarray) -> Dict[str, Any]:
        """Identify patterns in prediction errors."""
        patterns = {
            'trending': False,
            'cyclic': False,
            'volatility_dependent': False,
            'time_dependent': False
        }
        
        # Check for trending errors
        if len(errors) > 10:
            correlation = np.corrcoef(range(len(errors)), errors)[0, 1]
            patterns['trending'] = abs(correlation) > 0.3
        
        # Would add more sophisticated pattern detection
        
        return patterns
    
    async def identify_weaknesses(
        self, 
        metrics: ModelPerformanceMetrics
    ) -> List[LearningInsight]:
        """Identify model weaknesses and improvement opportunities."""
        
        insights = []
        
        # Check directional accuracy
        if metrics.directional_accuracy < 0.55:
            insights.append(LearningInsight(
                insight_type='weakness',
                description=f"Low directional accuracy: {metrics.directional_accuracy:.2%}",
                affected_models=[metrics.model_id],
                recommended_actions=[
                    "Add more predictive features",
                    "Increase training data window",
                    "Try ensemble methods",
                    "Adjust prediction horizon"
                ],
                evidence={'directional_accuracy': metrics.directional_accuracy},
                confidence=0.9,
                discovered_at=datetime.utcnow()
            ))
        
        # Check risk-reward ratio
        if metrics.profit_factor < 1.5:
            insights.append(LearningInsight(
                insight_type='weakness',
                description=f"Poor risk-reward ratio: {metrics.profit_factor:.2f}",
                affected_models=[metrics.model_id],
                recommended_actions=[
                    "Improve stop-loss placement",
                    "Optimize position sizing",
                    "Filter low-confidence trades",
                    "Adjust entry criteria"
                ],
                evidence={
                    'profit_factor': metrics.profit_factor,
                    'avg_win': metrics.avg_win,
                    'avg_loss': metrics.avg_loss
                },
                confidence=0.85,
                discovered_at=datetime.utcnow()
            ))
        
        # Check drawdown
        if abs(metrics.max_drawdown) > 0.15:
            insights.append(LearningInsight(
                insight_type='weakness',
                description=f"Excessive drawdown: {abs(metrics.max_drawdown):.2%}",
                affected_models=[metrics.model_id],
                recommended_actions=[
                    "Implement dynamic position sizing",
                    "Add volatility-based filters",
                    "Use portfolio diversification",
                    "Implement regime detection"
                ],
                evidence={'max_drawdown': metrics.max_drawdown},
                confidence=0.9,
                discovered_at=datetime.utcnow()
            ))
        
        # Check for error patterns
        if metrics.error_analysis['error_patterns'].get('trending'):
            insights.append(LearningInsight(
                insight_type='pattern',
                description="Systematic bias detected in predictions",
                affected_models=[metrics.model_id],
                recommended_actions=[
                    "Recalibrate model",
                    "Add detrending preprocessing",
                    "Check for data leakage",
                    "Update feature normalization"
                ],
                evidence=metrics.error_analysis,
                confidence=0.8,
                discovered_at=datetime.utcnow()
            ))
        
        return insights


class StrategyOptimizer:
    """Optimizes trading strategies using local LLM analysis."""
    
    def __init__(self):
        self.swarm = LocalSwarm()
        self.optimization_agent = self._create_optimization_agent()
        self.swarm.add_agent(self.optimization_agent)
        
    def _create_optimization_agent(self) -> LocalAgent:
        """Create specialized optimization agent."""
        return LocalAgent(
            name="Strategy Optimizer",
            model=RECOMMENDED_MODELS["strategy_generation"]["model"],
            instructions="""You are an expert quantitative trading strategy optimizer.

Your responsibilities:
- Analyze model performance metrics and identify improvements
- Suggest feature engineering opportunities
- Recommend hyperparameter adjustments
- Design ensemble strategies
- Propose risk management enhancements

When analyzing performance:
1. Identify the root cause of underperformance
2. Suggest specific, actionable improvements
3. Estimate expected performance gain
4. Consider implementation complexity
5. Prioritize by impact and feasibility

Always provide concrete, implementable recommendations."""
        )
    
    async def generate_improvement_strategies(
        self,
        metrics: ModelPerformanceMetrics,
        insights: List[LearningInsight]
    ) -> List[ImprovementStrategy]:
        """Generate improvement strategies using LLM analysis."""
        
        # Prepare context for LLM
        context = {
            'current_performance': {
                'directional_accuracy': metrics.directional_accuracy,
                'sharpe_ratio': metrics.sharpe_ratio,
                'profit_factor': metrics.profit_factor,
                'max_drawdown': metrics.max_drawdown,
                'win_rate': metrics.win_rate
            },
            'weaknesses': [insight.description for insight in insights if insight.insight_type == 'weakness'],
            'patterns': [insight.description for insight in insights if insight.insight_type == 'pattern']
        }
        
        prompt = f"""Analyze this trading model's performance and suggest improvements:

Current Performance:
{json.dumps(context['current_performance'], indent=2)}

Identified Weaknesses:
{json.dumps(context['weaknesses'], indent=2)}

Patterns Detected:
{json.dumps(context['patterns'], indent=2)}

Generate 3-5 specific improvement strategies. For each strategy provide:
1. Type (feature_engineering/hyperparameter/ensemble/architecture)
2. Detailed description
3. Expected improvement percentage
4. Implementation steps
5. Priority (1=high, 2=medium, 3=low)

Format as JSON list."""

        messages = [{"role": "user", "content": prompt}]
        
        response = await self.swarm.run(
            self.optimization_agent,
            messages,
            context
        )
        
        strategies = []
        
        try:
            # Parse LLM response
            suggestions = self._parse_llm_suggestions(response.content)
            
            for suggestion in suggestions:
                strategy = ImprovementStrategy(
                    strategy_type=suggestion.get('type', 'general'),
                    description=suggestion.get('description', ''),
                    expected_improvement=suggestion.get('expected_improvement', 5.0),
                    confidence=response.confidence,
                    implementation_steps=suggestion.get('steps', []),
                    resources_required=suggestion.get('resources', {}),
                    priority=suggestion.get('priority', 2)
                )
                strategies.append(strategy)
                
        except Exception as e:
            logger.error(f"Failed to parse improvement strategies: {e}")
            
            # Fallback strategies
            strategies.append(ImprovementStrategy(
                strategy_type='feature_engineering',
                description='Add market microstructure features',
                expected_improvement=10.0,
                confidence=0.7,
                implementation_steps=[
                    'Calculate order flow imbalance',
                    'Add bid-ask spread features',
                    'Include volume profile analysis'
                ],
                resources_required={'data': 'level2', 'compute': 'medium'},
                priority=2
            ))
        
        return strategies
    
    def _parse_llm_suggestions(self, content: str) -> List[Dict[str, Any]]:
        """Parse LLM suggestions from response."""
        # Try to extract JSON from response
        try:
            # Look for JSON in the response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback parsing
        return []


class AutoMLEngine:
    """Automated machine learning with continuous improvement."""
    
    def __init__(self):
        self.experiment_history = []
        self.best_models = {}
        self.active_experiments = set()
        
    async def run_experiment(
        self,
        symbol: str,
        strategy: ImprovementStrategy,
        current_model: Any
    ) -> Dict[str, Any]:
        """Run an improvement experiment."""
        
        experiment_id = f"exp_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_experiments.add(experiment_id)
        
        try:
            logger.info(f"Starting experiment {experiment_id}: {strategy.description}")
            
            result = {
                'experiment_id': experiment_id,
                'symbol': symbol,
                'strategy': asdict(strategy),
                'start_time': datetime.utcnow(),
                'status': 'running'
            }
            
            if strategy.strategy_type == 'feature_engineering':
                result.update(await self._experiment_feature_engineering(symbol, strategy))
            elif strategy.strategy_type == 'hyperparameter':
                result.update(await self._experiment_hyperparameter_tuning(symbol, strategy))
            elif strategy.strategy_type == 'ensemble':
                result.update(await self._experiment_ensemble_methods(symbol, strategy))
            else:
                result.update(await self._experiment_architecture_change(symbol, strategy))
            
            result['end_time'] = datetime.utcnow()
            result['duration'] = (result['end_time'] - result['start_time']).total_seconds()
            result['status'] = 'completed'
            
            # Store experiment results
            self.experiment_history.append(result)
            
            # Update best model if improved
            if result.get('improvement', 0) > 0:
                current_best = self.best_models.get(symbol, {}).get('score', 0)
                if result.get('final_score', 0) > current_best:
                    self.best_models[symbol] = {
                        'model': result.get('model'),
                        'score': result['final_score'],
                        'experiment_id': experiment_id,
                        'timestamp': datetime.utcnow()
                    }
                    logger.info(f"New best model for {symbol}: {result['final_score']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}")
            return {
                'experiment_id': experiment_id,
                'status': 'failed',
                'error': str(e)
            }
        finally:
            self.active_experiments.discard(experiment_id)
    
    async def _experiment_feature_engineering(
        self, 
        symbol: str, 
        strategy: ImprovementStrategy
    ) -> Dict[str, Any]:
        """Experiment with new features."""
        
        # Simulate feature engineering experiment
        # In production, would actually create and test new features
        
        baseline_score = 0.65
        new_score = baseline_score + (strategy.expected_improvement / 100)
        
        return {
            'baseline_score': baseline_score,
            'final_score': new_score,
            'improvement': (new_score - baseline_score) / baseline_score * 100,
            'new_features': ['feature_1', 'feature_2'],  # Would be actual features
            'feature_importance': {'feature_1': 0.15, 'feature_2': 0.10}
        }
    
    async def _experiment_hyperparameter_tuning(
        self,
        symbol: str,
        strategy: ImprovementStrategy
    ) -> Dict[str, Any]:
        """Experiment with hyperparameter optimization."""
        
        # Simulate hyperparameter tuning
        # In production, would use Optuna or similar
        
        baseline_score = 0.65
        new_score = baseline_score + (strategy.expected_improvement / 100) * 0.8
        
        return {
            'baseline_score': baseline_score,
            'final_score': new_score,
            'improvement': (new_score - baseline_score) / baseline_score * 100,
            'best_params': {
                'learning_rate': 0.01,
                'max_depth': 7,
                'n_estimators': 200
            },
            'trials_run': 50
        }
    
    async def _experiment_ensemble_methods(
        self,
        symbol: str,
        strategy: ImprovementStrategy
    ) -> Dict[str, Any]:
        """Experiment with ensemble methods."""
        
        baseline_score = 0.65
        new_score = baseline_score + (strategy.expected_improvement / 100) * 1.2
        
        return {
            'baseline_score': baseline_score,
            'final_score': new_score,
            'improvement': (new_score - baseline_score) / baseline_score * 100,
            'ensemble_type': 'voting',
            'models_combined': ['rf', 'xgb', 'lgb'],
            'weights': [0.4, 0.3, 0.3]
        }
    
    async def _experiment_architecture_change(
        self,
        symbol: str,
        strategy: ImprovementStrategy
    ) -> Dict[str, Any]:
        """Experiment with model architecture changes."""
        
        baseline_score = 0.65
        new_score = baseline_score + (strategy.expected_improvement / 100) * 0.9
        
        return {
            'baseline_score': baseline_score,
            'final_score': new_score,
            'improvement': (new_score - baseline_score) / baseline_score * 100,
            'architecture': 'transformer',
            'layers': 6,
            'parameters': 1000000
        }


class ContinuousImprovementEngine:
    """Main continuous improvement orchestrator."""
    
    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
        self.optimizer = StrategyOptimizer()
        self.automl = AutoMLEngine()
        self.cache = None
        
        # Improvement configuration
        self.improvement_interval = timedelta(hours=24)  # Daily improvement cycle
        self.min_data_points = 100  # Minimum data for analysis
        self.improvement_threshold = 5.0  # Minimum % improvement to deploy
        
        # Tracking
        self.last_improvement = {}
        self.improvement_history = []
        self.is_running = False
        
    async def initialize(self):
        """Initialize the improvement engine."""
        self.cache = get_trading_cache()
        self.is_running = True
        
        # Start improvement cycle
        asyncio.create_task(self._continuous_improvement_loop())
        
        logger.info("Continuous Improvement Engine initialized")
    
    async def _continuous_improvement_loop(self):
        """Main improvement loop."""
        while self.is_running:
            try:
                # Run improvement cycle during off-hours
                if self._is_off_hours():
                    await self._run_improvement_cycle()
                
                # Wait before next cycle
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Improvement cycle error: {e}")
    
    def _is_off_hours(self) -> bool:
        """Check if it's off-hours for improvement."""
        now = datetime.now()
        current_time = now.time()
        weekday = now.weekday()
        
        # Weekend or after 6 PM or before 4 AM on weekdays
        return weekday >= 5 or current_time >= time(18, 0) or current_time <= time(4, 0)
    
    async def _run_improvement_cycle(self):
        """Run a complete improvement cycle."""
        logger.info("Starting improvement cycle")
        
        # Get models to improve
        models_to_improve = await self._get_models_for_improvement()
        
        for model_info in models_to_improve:
            try:
                symbol = model_info['symbol']
                model_id = model_info['model_id']
                
                # Check if recently improved
                last_improved = self.last_improvement.get(model_id)
                if last_improved and (datetime.utcnow() - last_improved) < self.improvement_interval:
                    continue
                
                # Get recent performance data
                performance_data = await self._get_performance_data(model_id)
                
                if len(performance_data.get('predictions', [])) < self.min_data_points:
                    logger.info(f"Insufficient data for {model_id}, skipping")
                    continue
                
                # Analyze performance
                metrics = await self.analyzer.analyze_model_performance(
                    model_id,
                    np.array(performance_data['predictions']),
                    np.array(performance_data['actuals']),
                    performance_data.get('trades', [])
                )
                metrics.symbol = symbol
                
                # Identify weaknesses
                insights = await self.analyzer.identify_weaknesses(metrics)
                
                if insights:
                    logger.info(f"Found {len(insights)} insights for {model_id}")
                    
                    # Generate improvement strategies
                    strategies = await self.optimizer.generate_improvement_strategies(metrics, insights)
                    
                    # Run experiments for top strategies
                    for strategy in strategies[:2]:  # Top 2 strategies
                        if strategy.priority <= 2:  # High or medium priority
                            result = await self.automl.run_experiment(
                                symbol, 
                                strategy,
                                model_info.get('model')
                            )
                            
                            if result.get('improvement', 0) >= self.improvement_threshold:
                                logger.info(f"Significant improvement found for {model_id}: "
                                          f"{result['improvement']:.1f}%")
                                
                                # Deploy improved model
                                await self._deploy_improved_model(model_id, result)
                
                # Update tracking
                self.last_improvement[model_id] = datetime.utcnow()
                
            except Exception as e:
                logger.error(f"Failed to improve model {model_id}: {e}")
        
        logger.info("Improvement cycle completed")
    
    async def _get_models_for_improvement(self) -> List[Dict[str, Any]]:
        """Get list of models that need improvement."""
        # This would fetch from model registry
        # For now, return placeholder
        return [
            {'model_id': 'model_1', 'symbol': 'AAPL', 'model': None},
            {'model_id': 'model_2', 'symbol': 'GOOGL', 'model': None}
        ]
    
    async def _get_performance_data(self, model_id: str) -> Dict[str, Any]:
        """Get recent performance data for model."""
        # This would fetch from performance tracking system
        # For now, return synthetic data
        return {
            'predictions': np.random.randn(200).tolist(),
            'actuals': np.random.randn(200).tolist(),
            'trades': [
                {'pnl': np.random.randn() * 100} 
                for _ in range(50)
            ]
        }
    
    async def _deploy_improved_model(self, model_id: str, experiment_result: Dict[str, Any]):
        """Deploy an improved model to production."""
        
        deployment_record = {
            'model_id': model_id,
            'experiment_id': experiment_result['experiment_id'],
            'improvement': experiment_result['improvement'],
            'deployed_at': datetime.utcnow(),
            'previous_score': experiment_result.get('baseline_score'),
            'new_score': experiment_result.get('final_score')
        }
        
        # Cache deployment record
        if self.cache:
            cache_key = f"model_deployment:{model_id}"
            await self.cache.set_json(cache_key, deployment_record, ttl=86400 * 30)
        
        # Store in history
        self.improvement_history.append(deployment_record)
        
        logger.info(f"Deployed improved model {model_id} with {experiment_result['improvement']:.1f}% improvement")
    
    async def get_improvement_status(self) -> Dict[str, Any]:
        """Get current improvement engine status."""
        return {
            'is_running': self.is_running,
            'is_off_hours': self._is_off_hours(),
            'active_experiments': len(self.automl.active_experiments),
            'total_experiments': len(self.automl.experiment_history),
            'total_improvements': len(self.improvement_history),
            'best_models': {
                symbol: {
                    'score': info['score'],
                    'experiment_id': info['experiment_id'],
                    'timestamp': info['timestamp'].isoformat()
                }
                for symbol, info in self.automl.best_models.items()
            },
            'recent_improvements': self.improvement_history[-5:] if self.improvement_history else []
        }
    
    async def trigger_improvement(self, model_id: str) -> str:
        """Manually trigger improvement for a specific model."""
        logger.info(f"Manual improvement triggered for {model_id}")
        
        # Queue for improvement
        # This would add to improvement queue
        
        return f"Improvement scheduled for {model_id}"
    
    async def stop(self):
        """Stop the improvement engine."""
        self.is_running = False
        logger.info("Continuous Improvement Engine stopped")


# Global improvement engine instance
improvement_engine: Optional[ContinuousImprovementEngine] = None


async def get_improvement_engine() -> ContinuousImprovementEngine:
    """Get or create improvement engine instance."""
    global improvement_engine
    if improvement_engine is None:
        improvement_engine = ContinuousImprovementEngine()
        await improvement_engine.initialize()
    return improvement_engine