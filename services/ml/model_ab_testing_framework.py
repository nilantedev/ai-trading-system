#!/usr/bin/env python3
"""
Advanced A/B Testing and Model Comparison Framework
Implements statistical testing, multi-armed bandits, and champion/challenger systems
for optimal model selection in production.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp
import bayesian_testing.experiments as bt
from sklearn.model_selection import cross_val_score
import mlflow
from collections import defaultdict
import json

from trading_common import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ModelCandidate:
    """Model candidate for A/B testing."""
    model_id: str
    model_type: str  # 'champion', 'challenger', 'experimental'
    model_object: Any
    
    # Performance tracking
    predictions: List[float] = field(default_factory=list)
    actual_returns: List[float] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    
    # Metrics
    cumulative_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Traffic allocation
    traffic_weight: float = 0.0
    exploration_bonus: float = 0.0
    
    # Statistical testing
    p_value_vs_champion: Optional[float] = None
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    bayesian_probability: float = 0.5
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    total_predictions: int = 0
    deployment_stage: str = 'shadow'  # 'shadow', 'canary', 'production'


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    # Test parameters
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    power: float = 0.8
    minimum_detectable_effect: float = 0.02
    
    # Traffic allocation
    champion_traffic_pct: float = 0.7
    challenger_traffic_pct: float = 0.2
    experimental_traffic_pct: float = 0.1
    
    # Multi-armed bandit
    use_thompson_sampling: bool = True
    exploration_rate: float = 0.1
    ucb_c_param: float = 2.0
    
    # Evaluation frequency
    evaluation_interval_minutes: int = 60
    reallocation_frequency_hours: int = 24
    
    # Safety thresholds
    max_drawdown_threshold: float = 0.1
    min_sharpe_threshold: float = 1.0
    auto_rollback: bool = True


class MultiArmedBandit:
    """
    Multi-armed bandit for dynamic traffic allocation.
    Implements Thompson Sampling, UCB, and Epsilon-Greedy.
    """
    
    def __init__(self, n_arms: int, method: str = 'thompson'):
        self.n_arms = n_arms
        self.method = method
        
        # Track rewards
        self.successes = np.zeros(n_arms)
        self.failures = np.zeros(n_arms)
        self.total_pulls = np.zeros(n_arms)
        
        # UCB specific
        self.arm_values = np.zeros(n_arms)
        
    def select_arm(self, exploration_rate: float = 0.1) -> int:
        """Select next arm to pull."""
        if self.method == 'thompson':
            return self._thompson_sampling()
        elif self.method == 'ucb':
            return self._upper_confidence_bound()
        else:
            return self._epsilon_greedy(exploration_rate)
    
    def _thompson_sampling(self) -> int:
        """Thompson Sampling for Bernoulli rewards."""
        # Sample from Beta distributions
        samples = np.random.beta(
            self.successes + 1,
            self.failures + 1
        )
        return np.argmax(samples)
    
    def _upper_confidence_bound(self, c: float = 2.0) -> int:
        """Upper Confidence Bound algorithm."""
        total = np.sum(self.total_pulls)
        if total == 0:
            return np.random.randint(self.n_arms)
        
        ucb_values = self.arm_values + c * np.sqrt(
            np.log(total) / (self.total_pulls + 1e-5)
        )
        return np.argmax(ucb_values)
    
    def _epsilon_greedy(self, epsilon: float) -> int:
        """Epsilon-greedy selection."""
        if np.random.random() < epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.arm_values)
    
    def update(self, arm: int, reward: float):
        """Update arm statistics."""
        self.total_pulls[arm] += 1
        
        if reward > 0:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1
        
        # Update running average
        n = self.total_pulls[arm]
        self.arm_values[arm] = ((n - 1) * self.arm_values[arm] + reward) / n


class ModelABTestingFramework:
    """
    Comprehensive A/B testing framework for model comparison.
    """
    
    def __init__(self, config: ABTestConfig):
        self.config = config
        self.models: Dict[str, ModelCandidate] = {}
        self.champion_model: Optional[ModelCandidate] = None
        self.test_results: List[Dict] = []
        self.bandit: Optional[MultiArmedBandit] = None
        
        # Traffic routing
        self.traffic_allocations: Dict[str, float] = {}
        
        # Performance tracking
        self.performance_history: defaultdict = defaultdict(list)
        
    async def add_model(
        self,
        model_id: str,
        model: Any,
        model_type: str = 'challenger'
    ) -> ModelCandidate:
        """Add a model to the testing framework."""
        candidate = ModelCandidate(
            model_id=model_id,
            model_type=model_type,
            model_object=model
        )
        
        self.models[model_id] = candidate
        
        if model_type == 'champion':
            self.champion_model = candidate
        
        # Initialize bandit if using Thompson Sampling
        if self.config.use_thompson_sampling:
            self.bandit = MultiArmedBandit(
                n_arms=len(self.models),
                method='thompson'
            )
        
        logger.info(f"Added model {model_id} as {model_type}")
        return candidate
    
    async def route_prediction_request(
        self,
        features: np.ndarray
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Route prediction request to appropriate model based on traffic allocation.
        """
        # Select model based on current traffic allocation
        model_id = self._select_model_for_request()
        model = self.models[model_id]
        
        # Get prediction
        prediction = await self._get_model_prediction(model, features)
        
        # Track for evaluation
        model.total_predictions += 1
        
        return model_id, prediction, {
            'confidence': model.confidence_scores[-1] if model.confidence_scores else 0.5,
            'model_type': model.model_type,
            'deployment_stage': model.deployment_stage
        }
    
    def _select_model_for_request(self) -> str:
        """Select model based on traffic allocation strategy."""
        if self.config.use_thompson_sampling and self.bandit:
            # Use multi-armed bandit
            arm_idx = self.bandit.select_arm(self.config.exploration_rate)
            model_ids = list(self.models.keys())
            return model_ids[arm_idx]
        else:
            # Use fixed traffic allocation
            rand = np.random.random()
            cumulative = 0.0
            
            for model_id, weight in self.traffic_allocations.items():
                cumulative += weight
                if rand < cumulative:
                    return model_id
            
            # Fallback to champion
            return self.champion_model.model_id if self.champion_model else list(self.models.keys())[0]
    
    async def evaluate_models(
        self,
        actual_returns: pd.Series
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with statistical testing.
        """
        evaluation_results = {}
        
        for model_id, model in self.models.items():
            if len(model.predictions) < self.config.min_sample_size:
                continue
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(model, actual_returns)
            
            # Statistical testing vs champion
            if model.model_type != 'champion' and self.champion_model:
                test_results = self._statistical_test(model, self.champion_model)
                metrics.update(test_results)
            
            # Bayesian testing
            if model.model_type == 'challenger':
                bayesian_results = self._bayesian_test(model, self.champion_model)
                metrics.update(bayesian_results)
            
            evaluation_results[model_id] = metrics
            
            # Update model metrics
            self._update_model_metrics(model, metrics)
        
        # Determine if we should promote challenger
        promotion_decision = self._evaluate_promotion(evaluation_results)
        
        # Reallocate traffic based on performance
        if self.config.use_thompson_sampling:
            self._update_bandit_rewards(evaluation_results)
        else:
            self._reallocate_traffic(evaluation_results)
        
        # Log results
        self._log_evaluation_results(evaluation_results, promotion_decision)
        
        return {
            'evaluation_results': evaluation_results,
            'promotion_decision': promotion_decision,
            'traffic_allocations': self.traffic_allocations
        }
    
    def _calculate_performance_metrics(
        self,
        model: ModelCandidate,
        actual_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        predictions = np.array(model.predictions[-self.config.min_sample_size:])
        actuals = actual_returns.iloc[-len(predictions):].values
        
        # Returns-based metrics
        strategy_returns = predictions * actuals
        
        metrics = {
            'total_return': np.sum(strategy_returns),
            'avg_return': np.mean(strategy_returns),
            'volatility': np.std(strategy_returns),
            'sharpe_ratio': np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252),
            'win_rate': np.mean(strategy_returns > 0),
            'max_drawdown': self._calculate_max_drawdown(strategy_returns),
            'calmar_ratio': np.mean(strategy_returns) / (abs(self._calculate_max_drawdown(strategy_returns)) + 1e-8),
            
            # Prediction accuracy
            'directional_accuracy': np.mean(np.sign(predictions) == np.sign(actuals)),
            'mse': np.mean((predictions - actuals) ** 2),
            'mae': np.mean(np.abs(predictions - actuals))
        }
        
        return metrics
    
    def _statistical_test(
        self,
        challenger: ModelCandidate,
        champion: ModelCandidate
    ) -> Dict[str, Any]:
        """Perform statistical testing between models."""
        # Get returns for both models
        challenger_returns = np.array(challenger.predictions) * np.array(challenger.actual_returns)
        champion_returns = np.array(champion.predictions) * np.array(champion.actual_returns)
        
        # Ensure equal length
        min_len = min(len(challenger_returns), len(champion_returns))
        challenger_returns = challenger_returns[-min_len:]
        champion_returns = champion_returns[-min_len:]
        
        # T-test for mean returns
        t_stat, p_value_mean = ttest_ind(challenger_returns, champion_returns)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, p_value_median = mannwhitneyu(challenger_returns, champion_returns)
        
        # Kolmogorov-Smirnov test for distribution
        ks_stat, p_value_dist = ks_2samp(challenger_returns, champion_returns)
        
        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_confidence_interval(
            challenger_returns - champion_returns
        )
        
        return {
            'p_value_mean': p_value_mean,
            'p_value_median': p_value_median,
            'p_value_distribution': p_value_dist,
            'confidence_interval': (ci_lower, ci_upper),
            'statistically_significant': p_value_mean < (1 - self.config.confidence_level),
            'effect_size': np.mean(challenger_returns) - np.mean(champion_returns)
        }
    
    def _bayesian_test(
        self,
        challenger: ModelCandidate,
        champion: ModelCandidate
    ) -> Dict[str, float]:
        """Bayesian A/B testing for probabilistic comparison."""
        # Use Bayesian testing library
        test = bt.BinaryDataTest()
        
        # Add data
        test.add_variant_data(
            'champion',
            champion.total_predictions,
            sum(1 for r in champion.actual_returns if r > 0)
        )
        test.add_variant_data(
            'challenger',
            challenger.total_predictions,
            sum(1 for r in challenger.actual_returns if r > 0)
        )
        
        # Calculate probabilities
        prob_challenger_better = test.probability('challenger', 'champion')
        
        return {
            'bayesian_probability': prob_challenger_better,
            'expected_uplift': test.expected_uplift('challenger', 'champion'),
            'credible_interval': test.credible_interval('challenger')
        }
    
    def _evaluate_promotion(
        self,
        evaluation_results: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Determine if challenger should be promoted to champion."""
        promotion_decision = {
            'promote': False,
            'model_to_promote': None,
            'reasons': []
        }
        
        for model_id, metrics in evaluation_results.items():
            model = self.models[model_id]
            
            if model.model_type != 'challenger':
                continue
            
            # Check promotion criteria
            criteria_met = []
            
            # Statistical significance
            if metrics.get('statistically_significant', False):
                criteria_met.append('statistically_significant')
            
            # Bayesian probability
            if metrics.get('bayesian_probability', 0) > 0.95:
                criteria_met.append('high_bayesian_probability')
            
            # Performance thresholds
            if metrics['sharpe_ratio'] > self.config.min_sharpe_threshold:
                criteria_met.append('sharpe_threshold_met')
            
            if metrics['max_drawdown'] < self.config.max_drawdown_threshold:
                criteria_met.append('drawdown_threshold_met')
            
            # Consistent outperformance
            if metrics.get('effect_size', 0) > self.config.minimum_detectable_effect:
                criteria_met.append('meaningful_effect_size')
            
            # Make promotion decision
            if len(criteria_met) >= 3:  # Require multiple criteria
                promotion_decision['promote'] = True
                promotion_decision['model_to_promote'] = model_id
                promotion_decision['reasons'] = criteria_met
                break
        
        return promotion_decision
    
    def _reallocate_traffic(
        self,
        evaluation_results: Dict[str, Dict]
    ):
        """Reallocate traffic based on model performance."""
        # Calculate performance scores
        scores = {}
        for model_id, metrics in evaluation_results.items():
            # Composite score based on multiple metrics
            score = (
                metrics['sharpe_ratio'] * 0.4 +
                metrics['win_rate'] * 0.3 +
                (1 - metrics['max_drawdown']) * 0.3
            )
            scores[model_id] = max(0, score)
        
        # Normalize to get traffic weights
        total_score = sum(scores.values())
        if total_score > 0:
            for model_id, score in scores.items():
                base_weight = score / total_score
                
                # Apply constraints based on model type
                model = self.models[model_id]
                if model.model_type == 'champion':
                    weight = max(base_weight, self.config.champion_traffic_pct)
                elif model.model_type == 'experimental':
                    weight = min(base_weight, self.config.experimental_traffic_pct)
                else:
                    weight = base_weight
                
                self.traffic_allocations[model_id] = weight
        
        # Normalize again to ensure sum = 1
        total_weight = sum(self.traffic_allocations.values())
        for model_id in self.traffic_allocations:
            self.traffic_allocations[model_id] /= total_weight
    
    def _update_bandit_rewards(
        self,
        evaluation_results: Dict[str, Dict]
    ):
        """Update multi-armed bandit with rewards."""
        model_ids = list(self.models.keys())
        
        for i, model_id in enumerate(model_ids):
            if model_id in evaluation_results:
                metrics = evaluation_results[model_id]
                # Use Sharpe ratio as reward
                reward = metrics['sharpe_ratio']
                self.bandit.update(i, reward)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))
    
    def _bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval."""
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        lower = np.percentile(bootstrap_means, (1 - self.config.confidence_level) * 50)
        upper = np.percentile(bootstrap_means, 100 - (1 - self.config.confidence_level) * 50)
        
        return lower, upper
    
    def _log_evaluation_results(
        self,
        results: Dict[str, Dict],
        promotion_decision: Dict
    ):
        """Log evaluation results to MLflow."""
        with mlflow.start_run():
            # Log metrics for each model
            for model_id, metrics in results.items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{model_id}_{metric_name}", value)
            
            # Log promotion decision
            mlflow.log_dict(promotion_decision, "promotion_decision.json")
            
            # Log traffic allocations
            mlflow.log_dict(self.traffic_allocations, "traffic_allocations.json")
    
    async def promote_challenger(self, model_id: str):
        """Promote challenger to champion."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        # Demote current champion
        if self.champion_model:
            self.champion_model.model_type = 'previous_champion'
            self.champion_model.deployment_stage = 'shadow'
        
        # Promote challenger
        new_champion = self.models[model_id]
        new_champion.model_type = 'champion'
        new_champion.deployment_stage = 'production'
        self.champion_model = new_champion
        
        # Update traffic allocation
        self.traffic_allocations[model_id] = self.config.champion_traffic_pct
        
        logger.info(f"Promoted {model_id} to champion")
    
    def get_current_champion(self) -> Optional[ModelCandidate]:
        """Get current champion model."""
        return self.champion_model
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get model leaderboard with rankings."""
        leaderboard_data = []
        
        for model_id, model in self.models.items():
            leaderboard_data.append({
                'model_id': model_id,
                'model_type': model.model_type,
                'sharpe_ratio': model.sharpe_ratio,
                'win_rate': model.win_rate,
                'max_drawdown': model.max_drawdown,
                'total_predictions': model.total_predictions,
                'traffic_weight': self.traffic_allocations.get(model_id, 0),
                'deployment_stage': model.deployment_stage
            })
        
        df = pd.DataFrame(leaderboard_data)
        return df.sort_values('sharpe_ratio', ascending=False)


# Global A/B testing framework
_ab_testing_framework: Optional[ModelABTestingFramework] = None


async def get_ab_testing_framework() -> ModelABTestingFramework:
    """Get or create A/B testing framework."""
    global _ab_testing_framework
    if _ab_testing_framework is None:
        config = ABTestConfig()
        _ab_testing_framework = ModelABTestingFramework(config)
    return _ab_testing_framework