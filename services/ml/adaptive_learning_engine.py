#!/usr/bin/env python3
"""
Adaptive Learning Engine - Continuous Self-Improvement System
Learns from every trade, adapts to market changes, and evolves strategies
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import logging
import json
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Adaptive learning modes"""
    ONLINE = "online"  # Learn from each new data point
    BATCH = "batch"  # Learn from batches of data
    REINFORCEMENT = "reinforcement"  # Learn from rewards/penalties
    TRANSFER = "transfer"  # Transfer learning from similar patterns
    META = "meta"  # Learn how to learn better


@dataclass
class LearningUpdate:
    """Record of a learning update"""
    timestamp: datetime
    update_type: str
    parameters_before: Dict[str, Any]
    parameters_after: Dict[str, Any]
    performance_impact: float
    confidence: float
    lessons_learned: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptiveStrategy:
    """Self-adapting trading strategy"""
    name: str
    base_parameters: Dict[str, Any]
    current_parameters: Dict[str, Any]
    performance_history: List[float]
    adaptation_rate: float
    confidence_threshold: float
    last_update: datetime
    total_adaptations: int
    success_rate: float


class AdaptiveLearningEngine:
    """
    Continuous learning system that improves trading performance over time
    """
    
    def __init__(self):
        self.learning_history = deque(maxlen=10000)
        self.strategy_pool = {}
        self.performance_memory = defaultdict(list)
        self.market_patterns = {}
        self.meta_learner = None
        self.adaptation_models = {}
        self.initialize_learning_system()
    
    def initialize_learning_system(self):
        """Initialize the adaptive learning components"""
        # Initialize base learners
        self.base_learners = {
            'trend_learner': RandomForestRegressor(n_estimators=50, max_depth=5),
            'volatility_learner': GradientBoostingRegressor(n_estimators=50, max_depth=3),
            'pattern_learner': Ridge(alpha=1.0),
            'momentum_learner': LinearRegression()
        }
        
        # Initialize meta-learner
        self.meta_learner = RandomForestRegressor(n_estimators=100, max_depth=7)
        
        # Learning parameters
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
        self.memory_size = 1000
        self.update_frequency = 50
        
        # Performance tracking
        self.cumulative_reward = 0
        self.learning_curve = []
        self.adaptation_success_rate = 0.5
    
    async def learn_from_trade(
        self,
        trade_result: Dict,
        market_state: Dict,
        strategy_used: str
    ) -> LearningUpdate:
        """
        Learn from a completed trade and update strategies
        """
        # Extract features and outcome
        features = self._extract_learning_features(market_state)
        outcome = trade_result.get('return', 0)
        
        # Store in memory
        self.learning_history.append({
            'timestamp': datetime.utcnow(),
            'features': features,
            'outcome': outcome,
            'strategy': strategy_used,
            'market_state': market_state,
            'trade_result': trade_result
        })
        
        # Update performance memory
        self.performance_memory[strategy_used].append(outcome)
        
        # Determine if adaptation is needed
        if await self._should_adapt(strategy_used):
            update = await self._adapt_strategy(strategy_used, features, outcome)
        else:
            update = LearningUpdate(
                timestamp=datetime.utcnow(),
                update_type="observation",
                parameters_before={},
                parameters_after={},
                performance_impact=0,
                confidence=0.5,
                lessons_learned=["Data collected for future learning"]
            )
        
        # Update cumulative metrics
        self.cumulative_reward += outcome
        self.learning_curve.append(self.cumulative_reward)
        
        # Periodic meta-learning
        if len(self.learning_history) % self.update_frequency == 0:
            await self._perform_meta_learning()
        
        return update
    
    async def _should_adapt(self, strategy: str) -> bool:
        """Determine if strategy adaptation is warranted"""
        if strategy not in self.performance_memory:
            return False
        
        recent_performance = self.performance_memory[strategy][-20:]
        if len(recent_performance) < 10:
            return False
        
        # Check for performance degradation
        first_half = np.mean(recent_performance[:10])
        second_half = np.mean(recent_performance[10:])
        
        # Adapt if performance is declining or consistently negative
        return second_half < first_half * 0.9 or second_half < -0.01
    
    async def _adapt_strategy(
        self,
        strategy_name: str,
        features: np.ndarray,
        outcome: float
    ) -> LearningUpdate:
        """Adapt strategy parameters based on learning"""
        
        if strategy_name not in self.strategy_pool:
            # Initialize strategy
            self.strategy_pool[strategy_name] = AdaptiveStrategy(
                name=strategy_name,
                base_parameters=self._get_default_parameters(strategy_name),
                current_parameters=self._get_default_parameters(strategy_name),
                performance_history=[],
                adaptation_rate=self.learning_rate,
                confidence_threshold=0.6,
                last_update=datetime.utcnow(),
                total_adaptations=0,
                success_rate=0.5
            )
        
        strategy = self.strategy_pool[strategy_name]
        parameters_before = strategy.current_parameters.copy()
        
        # Learn optimal adjustments
        adjustments = await self._calculate_parameter_adjustments(
            strategy, features, outcome
        )
        
        # Apply adjustments with learning rate
        parameters_after = {}
        for param, value in parameters_before.items():
            if param in adjustments:
                if isinstance(value, (int, float)):
                    # Gradient-based update
                    new_value = value + self.learning_rate * adjustments[param]
                    # Clip to reasonable bounds
                    new_value = self._clip_parameter(param, new_value)
                    parameters_after[param] = new_value
                else:
                    parameters_after[param] = value
            else:
                parameters_after[param] = value
        
        # Update strategy
        strategy.current_parameters = parameters_after
        strategy.performance_history.append(outcome)
        strategy.total_adaptations += 1
        strategy.last_update = datetime.utcnow()
        
        # Calculate performance impact
        old_performance = np.mean(strategy.performance_history[-10:-1]) if len(strategy.performance_history) > 1 else 0
        performance_impact = outcome - old_performance
        
        # Update success rate
        if performance_impact > 0:
            strategy.success_rate = strategy.success_rate * 0.95 + 0.05
        else:
            strategy.success_rate = strategy.success_rate * 0.95
        
        # Generate lessons learned
        lessons = self._extract_lessons(parameters_before, parameters_after, outcome)
        
        return LearningUpdate(
            timestamp=datetime.utcnow(),
            update_type="parameter_adaptation",
            parameters_before=parameters_before,
            parameters_after=parameters_after,
            performance_impact=performance_impact,
            confidence=strategy.success_rate,
            lessons_learned=lessons,
            metadata={
                'strategy': strategy_name,
                'adaptation_number': strategy.total_adaptations,
                'learning_rate': self.learning_rate
            }
        )
    
    async def _calculate_parameter_adjustments(
        self,
        strategy: AdaptiveStrategy,
        features: np.ndarray,
        outcome: float
    ) -> Dict[str, float]:
        """Calculate optimal parameter adjustments using gradient estimation"""
        adjustments = {}
        current_params = strategy.current_parameters
        
        # Estimate gradients using finite differences
        epsilon = 0.01
        
        for param, value in current_params.items():
            if isinstance(value, (int, float)):
                # Estimate gradient
                gradient = await self._estimate_gradient(
                    strategy.name, param, value, features, outcome, epsilon
                )
                
                # Calculate adjustment
                adjustments[param] = gradient * outcome  # Reinforce good outcomes
        
        return adjustments
    
    async def _estimate_gradient(
        self,
        strategy_name: str,
        param: str,
        current_value: float,
        features: np.ndarray,
        outcome: float,
        epsilon: float
    ) -> float:
        """Estimate gradient of performance with respect to parameter"""
        # Use historical data to estimate gradient
        similar_trades = self._find_similar_trades(features, n=20)
        
        if len(similar_trades) < 5:
            # Not enough data, use random exploration
            return np.random.randn() * 0.1
        
        # Estimate local gradient
        param_values = [t.get('market_state', {}).get(param, current_value) for t in similar_trades]
        outcomes = [t.get('outcome', 0) for t in similar_trades]
        
        if len(set(param_values)) < 2:
            # Not enough variation
            return np.random.randn() * 0.1
        
        # Simple linear regression for gradient
        X = np.array(param_values).reshape(-1, 1)
        y = np.array(outcomes)
        
        try:
            reg = LinearRegression().fit(X, y)
            gradient = reg.coef_[0]
        except:
            gradient = 0
        
        return np.clip(gradient, -1, 1)
    
    def _find_similar_trades(self, features: np.ndarray, n: int = 20) -> List[Dict]:
        """Find similar historical trades based on features"""
        if len(self.learning_history) == 0:
            return []
        
        # Calculate distances to all historical trades
        distances = []
        for trade in self.learning_history:
            trade_features = trade.get('features', [])
            if len(trade_features) == len(features):
                distance = np.linalg.norm(features - trade_features)
                distances.append((distance, trade))
        
        # Sort by distance and return n closest
        distances.sort(key=lambda x: x[0])
        return [trade for _, trade in distances[:n]]
    
    async def _perform_meta_learning(self):
        """Learn how to learn better - optimize learning parameters"""
        if len(self.learning_history) < 100:
            return
        
        # Analyze learning performance
        recent_performance = [h.get('outcome', 0) for h in list(self.learning_history)[-100:]]
        performance_trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        
        # Adjust learning rate based on performance trend
        if performance_trend > 0:
            # Learning is working, can be more aggressive
            self.learning_rate = min(0.1, self.learning_rate * 1.05)
            self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
        else:
            # Learning not effective, be more conservative
            self.learning_rate = max(0.001, self.learning_rate * 0.95)
            self.exploration_rate = min(0.3, self.exploration_rate * 1.05)
        
        # Update meta-learner with aggregated features
        await self._train_meta_learner()
        
        logger.info(f"Meta-learning update: lr={self.learning_rate:.4f}, explore={self.exploration_rate:.3f}")
    
    async def _train_meta_learner(self):
        """Train meta-learner on aggregated learning experiences"""
        if len(self.learning_history) < 200:
            return
        
        # Prepare meta-features
        X_meta = []
        y_meta = []
        
        for i in range(100, len(self.learning_history)):
            # Look at past 100 trades
            past_trades = list(self.learning_history)[i-100:i]
            
            # Extract meta-features
            meta_features = self._extract_meta_features(past_trades)
            X_meta.append(meta_features)
            
            # Target is next period performance
            next_performance = self.learning_history[i].get('outcome', 0)
            y_meta.append(next_performance)
        
        if len(X_meta) > 50:
            # Train meta-learner
            X_meta = np.array(X_meta)
            y_meta = np.array(y_meta)
            
            try:
                self.meta_learner.fit(X_meta, y_meta)
                logger.info("Meta-learner updated successfully")
            except Exception as e:
                logger.error(f"Meta-learner training failed: {e}")
    
    def _extract_meta_features(self, trades: List[Dict]) -> np.ndarray:
        """Extract meta-level features from trading history"""
        if not trades:
            return np.zeros(10)
        
        outcomes = [t.get('outcome', 0) for t in trades]
        
        meta_features = [
            np.mean(outcomes),  # Average performance
            np.std(outcomes),  # Performance variance
            np.max(outcomes),  # Best performance
            np.min(outcomes),  # Worst performance
            len([o for o in outcomes if o > 0]) / len(outcomes),  # Win rate
            np.mean([abs(o) for o in outcomes]),  # Average magnitude
            self._calculate_trend(outcomes),  # Performance trend
            self._calculate_volatility_regime(trades),  # Market regime
            self.learning_rate,  # Current learning rate
            self.exploration_rate  # Current exploration rate
        ]
        
        return np.array(meta_features)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values"""
        if len(values) < 2:
            return 0
        return np.polyfit(range(len(values)), values, 1)[0]
    
    def _calculate_volatility_regime(self, trades: List[Dict]) -> float:
        """Estimate market volatility regime from trades"""
        if not trades:
            return 0.5
        
        volatilities = []
        for trade in trades:
            market_state = trade.get('market_state', {})
            vol = market_state.get('volatility', 0.2)
            volatilities.append(vol)
        
        return np.mean(volatilities) if volatilities else 0.2
    
    async def predict_strategy_performance(
        self,
        strategy_name: str,
        market_state: Dict
    ) -> Dict[str, float]:
        """Predict expected performance of a strategy given market state"""
        features = self._extract_learning_features(market_state)
        
        # Use ensemble of base learners
        predictions = {}
        
        for learner_name, model in self.base_learners.items():
            if hasattr(model, 'predict'):
                try:
                    # Check if model is trained
                    if hasattr(model, 'n_features_in_'):
                        prediction = model.predict(features.reshape(1, -1))[0]
                        predictions[learner_name] = prediction
                except:
                    predictions[learner_name] = 0
        
        # Meta-learner ensemble
        if self.meta_learner and hasattr(self.meta_learner, 'n_features_in_'):
            meta_features = self._extract_meta_features(list(self.learning_history)[-100:])
            try:
                meta_prediction = self.meta_learner.predict(meta_features.reshape(1, -1))[0]
                predictions['meta'] = meta_prediction
            except:
                pass
        
        # Weighted average
        if predictions:
            avg_prediction = np.mean(list(predictions.values()))
            confidence = 1.0 / (1.0 + np.std(list(predictions.values())))
        else:
            avg_prediction = 0
            confidence = 0.1
        
        return {
            'expected_return': avg_prediction,
            'confidence': confidence,
            'predictions': predictions
        }
    
    async def recommend_strategy_adaptations(self) -> List[Dict]:
        """Recommend strategy adaptations based on learning"""
        recommendations = []
        
        for strategy_name, strategy in self.strategy_pool.items():
            if len(strategy.performance_history) < 20:
                continue
            
            recent_performance = strategy.performance_history[-20:]
            avg_performance = np.mean(recent_performance)
            performance_trend = self._calculate_trend(recent_performance)
            
            # Generate recommendations
            if avg_performance < 0:
                recommendations.append({
                    'strategy': strategy_name,
                    'recommendation': 'Consider reducing position size',
                    'reason': f'Negative average performance: {avg_performance:.3f}',
                    'confidence': 0.8
                })
            
            if performance_trend < -0.001:
                recommendations.append({
                    'strategy': strategy_name,
                    'recommendation': 'Strategy parameters may need adjustment',
                    'reason': f'Declining performance trend: {performance_trend:.4f}',
                    'confidence': 0.7
                })
            
            if strategy.success_rate < 0.4:
                recommendations.append({
                    'strategy': strategy_name,
                    'recommendation': 'Consider pausing strategy for retraining',
                    'reason': f'Low success rate: {strategy.success_rate:.2%}',
                    'confidence': 0.9
                })
        
        return recommendations
    
    def _extract_learning_features(self, market_state: Dict) -> np.ndarray:
        """Extract features for learning from market state"""
        features = [
            market_state.get('price', 0),
            market_state.get('volume', 0),
            market_state.get('volatility', 0.2),
            market_state.get('rsi', 50),
            market_state.get('macd', 0),
            market_state.get('sma_ratio', 1.0),
            market_state.get('volume_ratio', 1.0),
            market_state.get('bid_ask_spread', 0.001),
            market_state.get('market_cap', 1e9),
            market_state.get('pe_ratio', 15)
        ]
        return np.array(features)
    
    def _get_default_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """Get default parameters for a strategy"""
        defaults = {
            'position_size': 0.02,
            'stop_loss': 0.02,
            'take_profit': 0.05,
            'entry_threshold': 0.6,
            'exit_threshold': 0.4,
            'max_holding_period': 5,
            'use_trailing_stop': False,
            'risk_multiplier': 1.0
        }
        return defaults
    
    def _clip_parameter(self, param: str, value: float) -> float:
        """Clip parameter to reasonable bounds"""
        bounds = {
            'position_size': (0.001, 0.1),
            'stop_loss': (0.005, 0.1),
            'take_profit': (0.01, 0.2),
            'entry_threshold': (0.3, 0.9),
            'exit_threshold': (0.1, 0.7),
            'max_holding_period': (1, 20),
            'risk_multiplier': (0.1, 3.0)
        }
        
        if param in bounds:
            min_val, max_val = bounds[param]
            return np.clip(value, min_val, max_val)
        return value
    
    def _extract_lessons(
        self,
        params_before: Dict,
        params_after: Dict,
        outcome: float
    ) -> List[str]:
        """Extract actionable lessons from parameter changes"""
        lessons = []
        
        for param, old_value in params_before.items():
            if param in params_after and isinstance(old_value, (int, float)):
                new_value = params_after[param]
                change = new_value - old_value
                
                if abs(change) > 0.001:
                    if outcome > 0:
                        if change > 0:
                            lessons.append(f"Increasing {param} improved performance")
                        else:
                            lessons.append(f"Decreasing {param} improved performance")
                    else:
                        if change > 0:
                            lessons.append(f"Increasing {param} may have hurt performance")
                        else:
                            lessons.append(f"Decreasing {param} may have hurt performance")
        
        if not lessons:
            lessons.append("Parameters fine-tuned based on recent performance")
        
        return lessons
    
    async def get_learning_insights(self) -> Dict:
        """Get insights from the learning system"""
        if not self.learning_history:
            return {
                'status': 'initializing',
                'total_experiences': 0,
                'insights': []
            }
        
        recent_outcomes = [h.get('outcome', 0) for h in list(self.learning_history)[-100:]]
        
        insights = {
            'status': 'active',
            'total_experiences': len(self.learning_history),
            'cumulative_return': self.cumulative_reward,
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate,
            'recent_performance': {
                'mean': np.mean(recent_outcomes) if recent_outcomes else 0,
                'std': np.std(recent_outcomes) if recent_outcomes else 0,
                'trend': self._calculate_trend(recent_outcomes) if len(recent_outcomes) > 1 else 0
            },
            'strategy_performance': {},
            'top_lessons': [],
            'adaptation_success_rate': self.adaptation_success_rate
        }
        
        # Strategy-specific insights
        for strategy_name, strategy in self.strategy_pool.items():
            if strategy.performance_history:
                insights['strategy_performance'][strategy_name] = {
                    'avg_return': np.mean(strategy.performance_history),
                    'total_adaptations': strategy.total_adaptations,
                    'success_rate': strategy.success_rate,
                    'last_update': strategy.last_update.isoformat()
                }
        
        # Extract top lessons
        if self.learning_history:
            recent_updates = [h for h in self.learning_history if 'lessons_learned' in h]
            if recent_updates:
                all_lessons = []
                for update in recent_updates[-10:]:
                    all_lessons.extend(update.get('lessons_learned', []))
                
                # Count frequency of lessons
                lesson_counts = {}
                for lesson in all_lessons:
                    lesson_counts[lesson] = lesson_counts.get(lesson, 0) + 1
                
                # Top 5 most common lessons
                sorted_lessons = sorted(lesson_counts.items(), key=lambda x: x[1], reverse=True)
                insights['top_lessons'] = [lesson for lesson, _ in sorted_lessons[:5]]
        
        return insights
    
    async def save_learning_state(self, filepath: str):
        """Save the current learning state to disk"""
        state = {
            'learning_history': list(self.learning_history),
            'strategy_pool': self.strategy_pool,
            'performance_memory': dict(self.performance_memory),
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate,
            'cumulative_reward': self.cumulative_reward,
            'learning_curve': self.learning_curve
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Learning state saved to {filepath}")
    
    async def load_learning_state(self, filepath: str):
        """Load learning state from disk"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.learning_history = deque(state['learning_history'], maxlen=10000)
            self.strategy_pool = state['strategy_pool']
            self.performance_memory = defaultdict(list, state['performance_memory'])
            self.learning_rate = state['learning_rate']
            self.exploration_rate = state['exploration_rate']
            self.cumulative_reward = state['cumulative_reward']
            self.learning_curve = state['learning_curve']
            
            logger.info(f"Learning state loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load learning state: {e}")


# Global instance
adaptive_learning = AdaptiveLearningEngine()


async def get_adaptive_learning() -> AdaptiveLearningEngine:
    """Get the adaptive learning engine instance"""
    return adaptive_learning