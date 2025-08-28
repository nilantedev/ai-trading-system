#!/usr/bin/env python3
"""
Ensemble Intelligence Coordinator - Orchestrates Multiple AI Models for Superior Performance
Combines predictions from multiple models using advanced ensemble techniques
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import json
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of models in the ensemble"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    PATTERN = "pattern"
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "ml"
    DEEP_LEARNING = "dl"
    REINFORCEMENT = "rl"
    QUANTUM = "quantum"


class EnsembleMethod(Enum):
    """Ensemble combination methods"""
    VOTING = "voting"
    AVERAGING = "averaging"
    WEIGHTED_AVERAGE = "weighted_average"
    STACKING = "stacking"
    BLENDING = "blending"
    BAYESIAN = "bayesian"
    DYNAMIC = "dynamic"
    HIERARCHICAL = "hierarchical"


@dataclass
class ModelPrediction:
    """Individual model prediction"""
    model_id: str
    model_type: ModelType
    prediction: str  # BUY, SELL, HOLD
    confidence: float
    probability_distribution: Dict[str, float]
    features_used: List[str]
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsemblePrediction:
    """Aggregated ensemble prediction"""
    final_prediction: str
    confidence: float
    consensus_level: float
    model_contributions: Dict[str, float]
    probability_distribution: Dict[str, float]
    dissenting_models: List[str]
    ensemble_method: EnsembleMethod
    risk_assessment: Dict[str, float]
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnsembleIntelligenceCoordinator:
    """
    Coordinates multiple AI models to generate superior trading decisions
    """
    
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        self.ensemble_methods = {}
        self.meta_learner = None
        self.initialize_ensemble()
    
    def initialize_ensemble(self):
        """Initialize ensemble components"""
        # Default model weights (will be dynamically adjusted)
        self.model_weights = {
            ModelType.TECHNICAL: 0.20,
            ModelType.FUNDAMENTAL: 0.15,
            ModelType.SENTIMENT: 0.10,
            ModelType.PATTERN: 0.15,
            ModelType.STATISTICAL: 0.15,
            ModelType.MACHINE_LEARNING: 0.20,
            ModelType.DEEP_LEARNING: 0.05
        }
        
        # Performance tracking
        self.model_performance = defaultdict(lambda: {
            'accuracy': 0.5,
            'precision': 0.5,
            'recall': 0.5,
            'f1_score': 0.5,
            'sharpe_ratio': 0.0,
            'total_predictions': 0
        })
        
        # Dynamic weight adjustment parameters
        self.weight_learning_rate = 0.01
        self.performance_window = 100
        self.min_weight = 0.01
        self.max_weight = 0.40
    
    async def register_model(
        self,
        model_id: str,
        model_type: ModelType,
        model_func: Callable,
        initial_weight: float = None
    ):
        """Register a model in the ensemble"""
        self.models[model_id] = {
            'type': model_type,
            'function': model_func,
            'registered_at': datetime.utcnow(),
            'total_predictions': 0,
            'performance': {'accuracy': 0.5}
        }
        
        if initial_weight is not None:
            self.model_weights[model_type] = initial_weight
        
        logger.info(f"Registered model {model_id} of type {model_type}")
    
    async def generate_ensemble_prediction(
        self,
        market_data: Dict,
        ensemble_method: EnsembleMethod = EnsembleMethod.DYNAMIC
    ) -> EnsemblePrediction:
        """Generate ensemble prediction from all models"""
        
        # Collect predictions from all models
        model_predictions = await self._collect_model_predictions(market_data)
        
        if not model_predictions:
            return self._create_default_prediction()
        
        # Apply ensemble method
        if ensemble_method == EnsembleMethod.VOTING:
            ensemble_result = await self._ensemble_voting(model_predictions)
        elif ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE:
            ensemble_result = await self._ensemble_weighted_average(model_predictions)
        elif ensemble_method == EnsembleMethod.STACKING:
            ensemble_result = await self._ensemble_stacking(model_predictions, market_data)
        elif ensemble_method == EnsembleMethod.BAYESIAN:
            ensemble_result = await self._ensemble_bayesian(model_predictions)
        elif ensemble_method == EnsembleMethod.DYNAMIC:
            ensemble_result = await self._ensemble_dynamic(model_predictions, market_data)
        else:
            ensemble_result = await self._ensemble_weighted_average(model_predictions)
        
        # Assess risk and confidence
        risk_assessment = await self._assess_ensemble_risk(model_predictions, ensemble_result)
        
        # Calculate performance metrics
        performance_metrics = await self._calculate_performance_metrics(model_predictions)
        
        return EnsemblePrediction(
            final_prediction=ensemble_result['prediction'],
            confidence=ensemble_result['confidence'],
            consensus_level=ensemble_result['consensus'],
            model_contributions=ensemble_result['contributions'],
            probability_distribution=ensemble_result['probabilities'],
            dissenting_models=ensemble_result['dissenters'],
            ensemble_method=ensemble_method,
            risk_assessment=risk_assessment,
            performance_metrics=performance_metrics,
            metadata={
                'models_used': len(model_predictions),
                'timestamp': datetime.utcnow().isoformat(),
                'market_conditions': market_data.get('regime', 'unknown')
            }
        )
    
    async def _collect_model_predictions(
        self,
        market_data: Dict
    ) -> List[ModelPrediction]:
        """Collect predictions from all registered models"""
        predictions = []
        
        # Run models in parallel
        tasks = []
        for model_id, model_info in self.models.items():
            task = self._get_model_prediction(model_id, model_info, market_data)
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, ModelPrediction):
                    predictions.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Model prediction failed: {result}")
        
        return predictions
    
    async def _get_model_prediction(
        self,
        model_id: str,
        model_info: Dict,
        market_data: Dict
    ) -> ModelPrediction:
        """Get prediction from a single model"""
        try:
            start_time = datetime.utcnow()
            
            # Call model function
            model_func = model_info['function']
            if asyncio.iscoroutinefunction(model_func):
                result = await model_func(market_data)
            else:
                result = model_func(market_data)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Parse result
            if isinstance(result, dict):
                prediction = result.get('prediction', 'HOLD')
                confidence = result.get('confidence', 0.5)
                probabilities = result.get('probabilities', {
                    'BUY': 0.33,
                    'SELL': 0.33,
                    'HOLD': 0.34
                })
            else:
                prediction = str(result)
                confidence = 0.5
                probabilities = {'BUY': 0.33, 'SELL': 0.33, 'HOLD': 0.34}
            
            return ModelPrediction(
                model_id=model_id,
                model_type=model_info['type'],
                prediction=prediction,
                confidence=confidence,
                probability_distribution=probabilities,
                features_used=market_data.get('features', []),
                execution_time_ms=execution_time,
                metadata={'model_version': model_info.get('version', '1.0')}
            )
            
        except Exception as e:
            logger.error(f"Model {model_id} prediction failed: {e}")
            raise
    
    async def _ensemble_voting(
        self,
        predictions: List[ModelPrediction]
    ) -> Dict:
        """Simple majority voting ensemble"""
        votes = defaultdict(int)
        
        for pred in predictions:
            votes[pred.prediction] += 1
        
        # Find winner
        total_votes = len(predictions)
        winner = max(votes.items(), key=lambda x: x[1])
        
        # Calculate consensus
        consensus = winner[1] / total_votes if total_votes > 0 else 0
        
        # Calculate probabilities
        probabilities = {
            action: count / total_votes
            for action, count in votes.items()
        }
        
        # Identify dissenters
        dissenters = [
            p.model_id for p in predictions
            if p.prediction != winner[0]
        ]
        
        return {
            'prediction': winner[0],
            'confidence': consensus,
            'consensus': consensus,
            'contributions': {p.model_id: 1/total_votes for p in predictions},
            'probabilities': probabilities,
            'dissenters': dissenters
        }
    
    async def _ensemble_weighted_average(
        self,
        predictions: List[ModelPrediction]
    ) -> Dict:
        """Weighted average ensemble based on model performance"""
        weighted_probs = defaultdict(float)
        total_weight = 0
        
        for pred in predictions:
            weight = self.model_weights.get(pred.model_type, 0.1) * pred.confidence
            total_weight += weight
            
            for action, prob in pred.probability_distribution.items():
                weighted_probs[action] += weight * prob
        
        # Normalize
        if total_weight > 0:
            for action in weighted_probs:
                weighted_probs[action] /= total_weight
        
        # Find best action
        best_action = max(weighted_probs.items(), key=lambda x: x[1])
        
        # Calculate confidence
        confidence = best_action[1]
        
        # Calculate consensus (agreement level)
        action_agreements = defaultdict(list)
        for pred in predictions:
            action_agreements[pred.prediction].append(pred.confidence)
        
        consensus = 0
        if best_action[0] in action_agreements:
            agreeing_confidences = action_agreements[best_action[0]]
            consensus = np.mean(agreeing_confidences) * len(agreeing_confidences) / len(predictions)
        
        # Model contributions
        contributions = {}
        for pred in predictions:
            weight = self.model_weights.get(pred.model_type, 0.1) * pred.confidence
            contributions[pred.model_id] = weight / total_weight if total_weight > 0 else 0
        
        # Dissenters
        dissenters = [
            p.model_id for p in predictions
            if p.prediction != best_action[0] and p.confidence > 0.7
        ]
        
        return {
            'prediction': best_action[0],
            'confidence': confidence,
            'consensus': consensus,
            'contributions': contributions,
            'probabilities': dict(weighted_probs),
            'dissenters': dissenters
        }
    
    async def _ensemble_stacking(
        self,
        predictions: List[ModelPrediction],
        market_data: Dict
    ) -> Dict:
        """Stacking ensemble with meta-learner"""
        # Create feature matrix from predictions
        features = []
        
        for pred in predictions:
            feature_vec = [
                1 if pred.prediction == 'BUY' else 0,
                1 if pred.prediction == 'SELL' else 0,
                1 if pred.prediction == 'HOLD' else 0,
                pred.confidence,
                pred.probability_distribution.get('BUY', 0),
                pred.probability_distribution.get('SELL', 0),
                pred.probability_distribution.get('HOLD', 0)
            ]
            features.extend(feature_vec)
        
        # Add market features
        features.extend([
            market_data.get('volatility', 0.2),
            market_data.get('volume_ratio', 1.0),
            market_data.get('rsi', 50) / 100,
            market_data.get('trend_strength', 0.5)
        ])
        
        # Use meta-learner if available, otherwise fall back to weighted average
        if self.meta_learner and hasattr(self.meta_learner, 'predict'):
            try:
                # Meta-learner prediction
                meta_pred = self.meta_learner.predict([features])[0]
                
                if meta_pred > 0.6:
                    prediction = 'BUY'
                elif meta_pred < -0.6:
                    prediction = 'SELL'
                else:
                    prediction = 'HOLD'
                
                confidence = abs(meta_pred)
            except:
                # Fallback to weighted average
                return await self._ensemble_weighted_average(predictions)
        else:
            return await self._ensemble_weighted_average(predictions)
        
        # Calculate other metrics
        consensus = self._calculate_consensus(predictions, prediction)
        probabilities = self._aggregate_probabilities(predictions)
        contributions = {p.model_id: 1/len(predictions) for p in predictions}
        dissenters = [p.model_id for p in predictions if p.prediction != prediction]
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'consensus': consensus,
            'contributions': contributions,
            'probabilities': probabilities,
            'dissenters': dissenters
        }
    
    async def _ensemble_bayesian(
        self,
        predictions: List[ModelPrediction]
    ) -> Dict:
        """Bayesian model averaging"""
        # Prior probabilities (uniform)
        prior = {'BUY': 0.33, 'SELL': 0.33, 'HOLD': 0.34}
        
        # Calculate posterior probabilities
        posterior = prior.copy()
        
        for pred in predictions:
            # Update posterior with each model's evidence
            model_weight = self.model_weights.get(pred.model_type, 0.1)
            
            for action in ['BUY', 'SELL', 'HOLD']:
                likelihood = pred.probability_distribution.get(action, 0.33)
                posterior[action] *= (likelihood * model_weight + (1 - model_weight) * prior[action])
        
        # Normalize
        total = sum(posterior.values())
        if total > 0:
            posterior = {k: v/total for k, v in posterior.items()}
        
        # Find MAP estimate
        best_action = max(posterior.items(), key=lambda x: x[1])
        
        # Calculate confidence and consensus
        confidence = best_action[1]
        consensus = self._calculate_consensus(predictions, best_action[0])
        
        # Model contributions (based on KL divergence from posterior)
        contributions = {}
        for pred in predictions:
            kl_div = self._calculate_kl_divergence(
                pred.probability_distribution,
                posterior
            )
            contributions[pred.model_id] = 1 / (1 + kl_div)
        
        # Normalize contributions
        total_contrib = sum(contributions.values())
        if total_contrib > 0:
            contributions = {k: v/total_contrib for k, v in contributions.items()}
        
        # Dissenters
        dissenters = [
            p.model_id for p in predictions
            if p.prediction != best_action[0] and p.confidence > 0.7
        ]
        
        return {
            'prediction': best_action[0],
            'confidence': confidence,
            'consensus': consensus,
            'contributions': contributions,
            'probabilities': posterior,
            'dissenters': dissenters
        }
    
    async def _ensemble_dynamic(
        self,
        predictions: List[ModelPrediction],
        market_data: Dict
    ) -> Dict:
        """Dynamic ensemble that adapts to market conditions"""
        
        # Determine market regime
        volatility = market_data.get('volatility', 0.2)
        trend_strength = market_data.get('trend_strength', 0.5)
        volume_unusual = market_data.get('volume_ratio', 1.0) > 1.5
        
        # Select ensemble method based on conditions
        if volatility > 0.3:
            # High volatility - use conservative voting
            result = await self._ensemble_voting(predictions)
        elif trend_strength > 0.7:
            # Strong trend - weight trend-following models higher
            self._adjust_weights_for_trend(predictions)
            result = await self._ensemble_weighted_average(predictions)
        elif volume_unusual:
            # Unusual volume - use Bayesian to account for uncertainty
            result = await self._ensemble_bayesian(predictions)
        else:
            # Normal conditions - use weighted average
            result = await self._ensemble_weighted_average(predictions)
        
        # Dynamic confidence adjustment
        result['confidence'] *= self._calculate_dynamic_confidence_multiplier(
            predictions, market_data
        )
        
        return result
    
    def _adjust_weights_for_trend(self, predictions: List[ModelPrediction]):
        """Temporarily adjust weights for trending markets"""
        trend_favorable_types = [ModelType.TECHNICAL, ModelType.PATTERN, ModelType.MACHINE_LEARNING]
        
        for pred in predictions:
            if pred.model_type in trend_favorable_types:
                # Temporarily boost weight
                original_weight = self.model_weights.get(pred.model_type, 0.1)
                self.model_weights[pred.model_type] = min(original_weight * 1.5, self.max_weight)
    
    def _calculate_dynamic_confidence_multiplier(
        self,
        predictions: List[ModelPrediction],
        market_data: Dict
    ) -> float:
        """Calculate dynamic confidence multiplier based on conditions"""
        multiplier = 1.0
        
        # Reduce confidence in high volatility
        volatility = market_data.get('volatility', 0.2)
        if volatility > 0.3:
            multiplier *= 0.8
        elif volatility < 0.1:
            multiplier *= 1.1
        
        # Increase confidence with model agreement
        agreement_rate = self._calculate_agreement_rate(predictions)
        if agreement_rate > 0.8:
            multiplier *= 1.2
        elif agreement_rate < 0.4:
            multiplier *= 0.8
        
        # Adjust for data quality
        data_quality = market_data.get('data_quality', 1.0)
        multiplier *= data_quality
        
        return np.clip(multiplier, 0.5, 1.5)
    
    def _calculate_consensus(
        self,
        predictions: List[ModelPrediction],
        final_prediction: str
    ) -> float:
        """Calculate consensus level among models"""
        if not predictions:
            return 0
        
        agreeing = [p for p in predictions if p.prediction == final_prediction]
        consensus = len(agreeing) / len(predictions)
        
        # Weight by confidence
        weighted_consensus = sum(p.confidence for p in agreeing) / sum(p.confidence for p in predictions)
        
        return (consensus + weighted_consensus) / 2
    
    def _calculate_agreement_rate(self, predictions: List[ModelPrediction]) -> float:
        """Calculate overall agreement rate among models"""
        if len(predictions) < 2:
            return 1.0
        
        # Count pairwise agreements
        agreements = 0
        total_pairs = 0
        
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                total_pairs += 1
                if predictions[i].prediction == predictions[j].prediction:
                    agreements += 1
        
        return agreements / total_pairs if total_pairs > 0 else 0
    
    def _aggregate_probabilities(
        self,
        predictions: List[ModelPrediction]
    ) -> Dict[str, float]:
        """Aggregate probability distributions from all models"""
        aggregated = defaultdict(float)
        
        for pred in predictions:
            weight = pred.confidence
            for action, prob in pred.probability_distribution.items():
                aggregated[action] += weight * prob
        
        # Normalize
        total = sum(aggregated.values())
        if total > 0:
            aggregated = {k: v/total for k, v in aggregated.items()}
        
        # Ensure all actions are present
        for action in ['BUY', 'SELL', 'HOLD']:
            if action not in aggregated:
                aggregated[action] = 0
        
        return dict(aggregated)
    
    def _calculate_kl_divergence(self, p: Dict, q: Dict) -> float:
        """Calculate KL divergence between two probability distributions"""
        kl = 0
        epsilon = 1e-10
        
        for action in ['BUY', 'SELL', 'HOLD']:
            p_val = p.get(action, epsilon)
            q_val = q.get(action, epsilon)
            
            if p_val > 0:
                kl += p_val * np.log(p_val / (q_val + epsilon))
        
        return kl
    
    async def _assess_ensemble_risk(
        self,
        predictions: List[ModelPrediction],
        ensemble_result: Dict
    ) -> Dict[str, float]:
        """Assess risk of ensemble prediction"""
        risk_assessment = {}
        
        # Disagreement risk
        consensus = ensemble_result.get('consensus', 0)
        risk_assessment['disagreement_risk'] = 1 - consensus
        
        # Confidence variance risk
        confidences = [p.confidence for p in predictions]
        risk_assessment['confidence_variance'] = np.std(confidences) if confidences else 0
        
        # Model dropout risk (what if a model fails)
        dropout_risks = []
        for i in range(len(predictions)):
            # Remove one model and recalculate
            remaining = predictions[:i] + predictions[i+1:]
            if remaining:
                temp_result = await self._ensemble_voting(remaining)
                if temp_result['prediction'] != ensemble_result['prediction']:
                    dropout_risks.append(1.0)
                else:
                    dropout_risks.append(0.0)
        
        risk_assessment['model_dropout_risk'] = np.mean(dropout_risks) if dropout_risks else 0
        
        # Execution risk
        exec_times = [p.execution_time_ms for p in predictions]
        risk_assessment['execution_risk'] = max(exec_times) / 1000 if exec_times else 0  # Convert to seconds
        
        # Overall risk score
        risk_assessment['overall_risk'] = np.mean([
            risk_assessment['disagreement_risk'],
            risk_assessment['confidence_variance'],
            risk_assessment['model_dropout_risk'],
            min(risk_assessment['execution_risk'], 1.0)
        ])
        
        return risk_assessment
    
    async def _calculate_performance_metrics(
        self,
        predictions: List[ModelPrediction]
    ) -> Dict[str, float]:
        """Calculate ensemble performance metrics"""
        metrics = {}
        
        # Diversity metrics
        unique_predictions = len(set(p.prediction for p in predictions))
        metrics['prediction_diversity'] = unique_predictions / 3  # Normalized by possible actions
        
        # Speed metrics
        exec_times = [p.execution_time_ms for p in predictions]
        metrics['avg_execution_time_ms'] = np.mean(exec_times) if exec_times else 0
        metrics['max_execution_time_ms'] = max(exec_times) if exec_times else 0
        
        # Confidence metrics
        confidences = [p.confidence for p in predictions]
        metrics['avg_confidence'] = np.mean(confidences) if confidences else 0
        metrics['min_confidence'] = min(confidences) if confidences else 0
        metrics['max_confidence'] = max(confidences) if confidences else 0
        
        # Model participation
        metrics['models_participated'] = len(predictions)
        metrics['participation_rate'] = len(predictions) / len(self.models) if self.models else 0
        
        return metrics
    
    async def update_model_performance(
        self,
        model_id: str,
        actual_outcome: str,
        predicted_outcome: str,
        return_achieved: float
    ):
        """Update model performance metrics"""
        if model_id not in self.models:
            return
        
        # Update performance history
        self.performance_history[model_id].append({
            'timestamp': datetime.utcnow(),
            'predicted': predicted_outcome,
            'actual': actual_outcome,
            'return': return_achieved,
            'correct': predicted_outcome == actual_outcome
        })
        
        # Calculate updated metrics
        history = list(self.performance_history[model_id])
        if len(history) >= 10:
            recent = history[-self.performance_window:]
            
            correct_predictions = [h['correct'] for h in recent]
            returns = [h['return'] for h in recent]
            
            self.model_performance[model_id]['accuracy'] = np.mean(correct_predictions)
            self.model_performance[model_id]['sharpe_ratio'] = (
                np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            )
            self.model_performance[model_id]['total_predictions'] += 1
            
            # Update model type weight
            model_type = self.models[model_id]['type']
            self._update_model_weight(model_type, self.model_performance[model_id]['accuracy'])
    
    def _update_model_weight(self, model_type: ModelType, performance: float):
        """Dynamically update model weights based on performance"""
        current_weight = self.model_weights.get(model_type, 0.1)
        
        # Performance-based adjustment
        if performance > 0.6:
            adjustment = self.weight_learning_rate
        elif performance < 0.4:
            adjustment = -self.weight_learning_rate
        else:
            adjustment = 0
        
        # Update weight
        new_weight = current_weight + adjustment
        new_weight = np.clip(new_weight, self.min_weight, self.max_weight)
        
        # Renormalize all weights
        self.model_weights[model_type] = new_weight
        total_weight = sum(self.model_weights.values())
        
        if total_weight > 0:
            self.model_weights = {
                k: v/total_weight for k, v in self.model_weights.items()
            }
    
    def _create_default_prediction(self) -> EnsemblePrediction:
        """Create default prediction when no models available"""
        return EnsemblePrediction(
            final_prediction='HOLD',
            confidence=0.0,
            consensus_level=0.0,
            model_contributions={},
            probability_distribution={'BUY': 0.33, 'SELL': 0.33, 'HOLD': 0.34},
            dissenting_models=[],
            ensemble_method=EnsembleMethod.VOTING,
            risk_assessment={'overall_risk': 1.0},
            performance_metrics={}
        )
    
    async def get_ensemble_diagnostics(self) -> Dict:
        """Get detailed diagnostics of the ensemble system"""
        diagnostics = {
            'total_models': len(self.models),
            'model_weights': dict(self.model_weights),
            'model_performance': dict(self.model_performance),
            'ensemble_health': 'healthy' if len(self.models) >= 3 else 'degraded',
            'recommendations': []
        }
        
        # Generate recommendations
        for model_id, perf in self.model_performance.items():
            if perf['accuracy'] < 0.4:
                diagnostics['recommendations'].append(
                    f"Consider retraining or removing model {model_id} (accuracy: {perf['accuracy']:.2%})"
                )
            elif perf['accuracy'] > 0.7:
                diagnostics['recommendations'].append(
                    f"Model {model_id} performing well (accuracy: {perf['accuracy']:.2%})"
                )
        
        # Check ensemble diversity
        if len(set(self.model_weights.values())) < 3:
            diagnostics['recommendations'].append(
                "Low weight diversity - consider adding more diverse models"
            )
        
        return diagnostics


# Global instance
ensemble_coordinator = EnsembleIntelligenceCoordinator()


async def get_ensemble_coordinator() -> EnsembleIntelligenceCoordinator:
    """Get the ensemble intelligence coordinator instance"""
    return ensemble_coordinator