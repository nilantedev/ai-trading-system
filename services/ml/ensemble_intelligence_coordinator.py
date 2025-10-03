#!/usr/bin/env python3
"""
Ensemble Intelligence Coordinator
Clean, production-safe implementation coordinating multiple model signals.

Public contract used by services/ml/main.py:
- class EnsembleIntelligenceCoordinator with methods:
  - generate_signals(symbols: List[str], strategies: Optional[List[str]]) -> Dict
  - generate_ensemble_prediction(market_data: Dict[str, Any], method: EnsembleMethod) -> EnsemblePrediction
  - register_model(model_id, model_type, model_func, initial_weight?)
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ModelType(Enum):
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
    def __init__(self) -> None:
        self.models: Dict[str, Dict[str, Any]] = {}
        self.model_weights: Dict[ModelType, float] = {}
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.model_performance: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "accuracy": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "f1_score": 0.5,
                "sharpe_ratio": 0.0,
                "total_predictions": 0,
            }
        )
        self.meta_learner = None
        self.weight_learning_rate = 0.01
        self.performance_window = 100
        self.min_weight = 0.01
        self.max_weight = 0.40
        self._initialize_ensemble()

    def _initialize_ensemble(self) -> None:
        self.model_weights = {
            ModelType.TECHNICAL: 0.20,
            ModelType.FUNDAMENTAL: 0.15,
            ModelType.SENTIMENT: 0.10,
            ModelType.PATTERN: 0.15,
            ModelType.STATISTICAL: 0.15,
            ModelType.MACHINE_LEARNING: 0.20,
            ModelType.DEEP_LEARNING: 0.05,
        }

    async def register_model(
        self,
        model_id: str,
        model_type: ModelType,
        model_func: Callable,
        initial_weight: Optional[float] = None,
    ) -> None:
        self.models[model_id] = {
            "type": model_type,
            "function": model_func,
            "registered_at": datetime.utcnow(),
            "total_predictions": 0,
            "performance": {"accuracy": 0.5},
        }
        if initial_weight is not None:
            self.model_weights[model_type] = float(initial_weight)
        logger.info("Registered model %s of type %s", model_id, model_type)

    async def generate_ensemble_prediction(
        self,
        market_data: Dict[str, Any],
        ensemble_method: EnsembleMethod = EnsembleMethod.DYNAMIC,
    ) -> EnsemblePrediction:
        predictions = await self._collect_model_predictions(market_data)
        if not predictions:
            return self._create_default_prediction()

        if ensemble_method == EnsembleMethod.VOTING:
            ensemble_result = await self._ensemble_voting(predictions)
        elif ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE:
            ensemble_result = await self._ensemble_weighted_average(predictions)
        elif ensemble_method == EnsembleMethod.BAYESIAN:
            ensemble_result = await self._ensemble_bayesian(predictions)
        elif ensemble_method == EnsembleMethod.DYNAMIC:
            ensemble_result = await self._ensemble_dynamic(predictions, market_data)
        else:
            ensemble_result = await self._ensemble_weighted_average(predictions)

        risk_assessment = await self._assess_ensemble_risk(predictions, ensemble_result)
        performance_metrics = await self._calculate_performance_metrics(predictions)

        return EnsemblePrediction(
            final_prediction=ensemble_result["prediction"],
            confidence=ensemble_result["confidence"],
            consensus_level=ensemble_result["consensus"],
            model_contributions=ensemble_result["contributions"],
            probability_distribution=ensemble_result["probabilities"],
            dissenting_models=ensemble_result["dissenters"],
            ensemble_method=ensemble_method,
            risk_assessment=risk_assessment,
            performance_metrics=performance_metrics,
            metadata={
                "models_used": len(predictions),
                "timestamp": datetime.utcnow().isoformat(),
                "market_conditions": market_data.get("regime", "unknown"),
            },
        )

    async def _collect_model_predictions(self, market_data: Dict[str, Any]) -> List[ModelPrediction]:
        predictions: List[ModelPrediction] = []
        tasks: List[asyncio.Task] = []
        for model_id, model_info in self.models.items():
            tasks.append(asyncio.create_task(self._get_model_prediction(model_id, model_info, market_data)))
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, ModelPrediction):
                    predictions.append(result)
                elif isinstance(result, Exception):
                    logger.error("Model prediction failed: %s", result)
        return predictions

    async def _get_model_prediction(self, model_id: str, model_info: Dict[str, Any], market_data: Dict[str, Any]) -> ModelPrediction:
        start_time = datetime.utcnow()
        try:
            model_func = model_info["function"]
            if asyncio.iscoroutinefunction(model_func):
                result = await model_func(market_data)
            else:
                result = model_func(market_data)
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            if isinstance(result, dict):
                prediction = result.get("prediction", "HOLD")
                confidence = float(result.get("confidence", 0.5))
                probabilities = result.get("probabilities", {"BUY": 0.33, "SELL": 0.33, "HOLD": 0.34})
            else:
                prediction = str(result)
                confidence = 0.5
                probabilities = {"BUY": 0.33, "SELL": 0.33, "HOLD": 0.34}
            return ModelPrediction(
                model_id=model_id,
                model_type=model_info["type"],
                prediction=prediction,
                confidence=confidence,
                probability_distribution=probabilities,
                features_used=market_data.get("features", []),
                execution_time_ms=execution_time,
                metadata={"model_version": model_info.get("version", "1.0")},
            )
        except Exception:
            logger.exception("Model %s prediction failed", model_id)
            raise

    async def _ensemble_voting(self, predictions: List[ModelPrediction]) -> Dict[str, Any]:
        votes: Dict[str, int] = defaultdict(int)
        for pred in predictions:
            votes[pred.prediction] += 1
        total_votes = len(predictions)
        winner = max(votes.items(), key=lambda x: x[1]) if votes else ("HOLD", 0)
        consensus = winner[1] / total_votes if total_votes > 0 else 0
        probabilities = {action: count / total_votes for action, count in votes.items()} if total_votes else {"BUY": 0.33, "SELL": 0.33, "HOLD": 0.34}
        dissenters = [p.model_id for p in predictions if p.prediction != winner[0]]
        return {
            "prediction": winner[0],
            "confidence": consensus,
            "consensus": consensus,
            "contributions": {p.model_id: 1 / total_votes for p in predictions} if total_votes else {},
            "probabilities": probabilities,
            "dissenters": dissenters,
        }

    async def _ensemble_weighted_average(self, predictions: List[ModelPrediction]) -> Dict[str, Any]:
        weighted_probs: Dict[str, float] = defaultdict(float)
        total_weight = 0.0
        for pred in predictions:
            weight = self.model_weights.get(pred.model_type, 0.1) * float(pred.confidence)
            total_weight += weight
            for action, prob in pred.probability_distribution.items():
                weighted_probs[action] += weight * float(prob)
        if total_weight > 0:
            for action in list(weighted_probs.keys()):
                weighted_probs[action] /= total_weight
        else:
            weighted_probs = {"BUY": 0.33, "SELL": 0.33, "HOLD": 0.34}
        best_action = max(weighted_probs.items(), key=lambda x: x[1])
        confidence = float(best_action[1])
        action_agreements: Dict[str, List[float]] = defaultdict(list)
        for pred in predictions:
            action_agreements[pred.prediction].append(float(pred.confidence))
        consensus = 0.0
        if best_action[0] in action_agreements and len(predictions) > 0:
            agreeing_confidences = action_agreements[best_action[0]]
            consensus = float(np.mean(agreeing_confidences) * len(agreeing_confidences) / len(predictions))
        contributions = {}
        for pred in predictions:
            weight = self.model_weights.get(pred.model_type, 0.1) * float(pred.confidence)
            contributions[pred.model_id] = weight / total_weight if total_weight > 0 else 0.0
        dissenters = [p.model_id for p in predictions if p.prediction != best_action[0] and float(p.confidence) > 0.7]
        return {
            "prediction": best_action[0],
            "confidence": confidence,
            "consensus": consensus,
            "contributions": contributions,
            "probabilities": dict(weighted_probs),
            "dissenters": dissenters,
        }

    async def _ensemble_bayesian(self, predictions: List[ModelPrediction]) -> Dict[str, Any]:
        prior = {"BUY": 0.33, "SELL": 0.33, "HOLD": 0.34}
        posterior = prior.copy()
        for pred in predictions:
            model_weight = self.model_weights.get(pred.model_type, 0.1)
            for action in ["BUY", "SELL", "HOLD"]:
                likelihood = float(pred.probability_distribution.get(action, 0.33))
                posterior[action] *= (likelihood * model_weight + (1 - model_weight) * prior[action])
        total = sum(posterior.values())
        if total > 0:
            posterior = {k: v / total for k, v in posterior.items()}
        best_action = max(posterior.items(), key=lambda x: x[1])
        confidence = float(best_action[1])
        consensus = self._calculate_consensus(predictions, best_action[0])
        contributions: Dict[str, float] = {}
        for pred in predictions:
            kl_div = self._calculate_kl_divergence(pred.probability_distribution, posterior)
            contributions[pred.model_id] = 1 / (1 + kl_div)
        total_contrib = sum(contributions.values())
        if total_contrib > 0:
            contributions = {k: v / total_contrib for k, v in contributions.items()}
        dissenters = [p.model_id for p in predictions if p.prediction != best_action[0] and float(p.confidence) > 0.7]
        return {
            "prediction": best_action[0],
            "confidence": confidence,
            "consensus": consensus,
            "contributions": contributions,
            "probabilities": posterior,
            "dissenters": dissenters,
        }

    async def _ensemble_dynamic(self, predictions: List[ModelPrediction], market_data: Dict[str, Any]) -> Dict[str, Any]:
        volatility = float(market_data.get("volatility", 0.2))
        trend_strength = float(market_data.get("trend_strength", 0.5))
        volume_unusual = float(market_data.get("volume_ratio", 1.0)) > 1.5
        if volatility > 0.3:
            result = await self._ensemble_voting(predictions)
        elif trend_strength > 0.7:
            self._adjust_weights_for_trend(predictions)
            result = await self._ensemble_weighted_average(predictions)
        elif volume_unusual:
            result = await self._ensemble_bayesian(predictions)
        else:
            result = await self._ensemble_weighted_average(predictions)
        result["confidence"] *= float(self._calculate_dynamic_confidence_multiplier(predictions, market_data))
        return result

    def _adjust_weights_for_trend(self, predictions: List[ModelPrediction]) -> None:
        trend_favorable = {ModelType.TECHNICAL, ModelType.PATTERN, ModelType.MACHINE_LEARNING}
        for pred in predictions:
            if pred.model_type in trend_favorable:
                original = self.model_weights.get(pred.model_type, 0.1)
                self.model_weights[pred.model_type] = min(original * 1.5, self.max_weight)

    def _calculate_dynamic_confidence_multiplier(self, predictions: List[ModelPrediction], market_data: Dict[str, Any]) -> float:
        multiplier = 1.0
        volatility = float(market_data.get("volatility", 0.2))
        if volatility > 0.3:
            multiplier *= 0.8
        elif volatility < 0.1:
            multiplier *= 1.1
        agreement_rate = self._calculate_agreement_rate(predictions)
        if agreement_rate > 0.8:
            multiplier *= 1.2
        elif agreement_rate < 0.4:
            multiplier *= 0.8
        data_quality = float(market_data.get("data_quality", 1.0))
        multiplier *= data_quality
        return float(np.clip(multiplier, 0.5, 1.5))

    def _calculate_consensus(self, predictions: List[ModelPrediction], final_prediction: str) -> float:
        if not predictions:
            return 0.0
        agreeing = [p for p in predictions if p.prediction == final_prediction]
        consensus = len(agreeing) / len(predictions)
        if sum(p.confidence for p in predictions) > 0:
            weighted = sum(p.confidence for p in agreeing) / sum(p.confidence for p in predictions)
        else:
            weighted = consensus
        return float((consensus + weighted) / 2)

    def _calculate_agreement_rate(self, predictions: List[ModelPrediction]) -> float:
        n = len(predictions)
        if n < 2:
            return 1.0
        agreements = 0
        total_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                if predictions[i].prediction == predictions[j].prediction:
                    agreements += 1
        return float(agreements / total_pairs) if total_pairs else 0.0

    def _aggregate_probabilities(self, predictions: List[ModelPrediction]) -> Dict[str, float]:
        aggregated: Dict[str, float] = defaultdict(float)
        for pred in predictions:
            weight = float(pred.confidence)
            for action, prob in pred.probability_distribution.items():
                aggregated[action] += weight * float(prob)
        total = sum(aggregated.values())
        if total > 0:
            aggregated = {k: v / total for k, v in aggregated.items()}
        for action in ["BUY", "SELL", "HOLD"]:
            aggregated.setdefault(action, 0.0)
        return dict(aggregated)

    def _calculate_kl_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        kl = 0.0
        epsilon = 1e-10
        for action in ["BUY", "SELL", "HOLD"]:
            p_val = float(p.get(action, epsilon))
            q_val = float(q.get(action, epsilon))
            if p_val > 0:
                kl += p_val * np.log(p_val / (q_val + epsilon))
        return float(kl)

    async def _assess_ensemble_risk(self, predictions: List[ModelPrediction], ensemble_result: Dict[str, Any]) -> Dict[str, float]:
        risk: Dict[str, float] = {}
        consensus = float(ensemble_result.get("consensus", 0.0))
        risk["disagreement_risk"] = 1.0 - consensus
        confidences = [float(p.confidence) for p in predictions]
        risk["confidence_variance"] = float(np.std(confidences)) if confidences else 0.0
        dropout = []
        for i in range(len(predictions)):
            remaining = predictions[:i] + predictions[i + 1 :]
            if remaining:
                temp = await self._ensemble_voting(remaining)
                dropout.append(1.0 if temp["prediction"] != ensemble_result["prediction"] else 0.0)
        risk["model_dropout_risk"] = float(np.mean(dropout)) if dropout else 0.0
        exec_times = [float(p.execution_time_ms) for p in predictions]
        risk["execution_risk"] = (max(exec_times) / 1000.0) if exec_times else 0.0
        risk["overall_risk"] = float(
            np.mean(
                [
                    risk["disagreement_risk"],
                    risk["confidence_variance"],
                    risk["model_dropout_risk"],
                    min(risk["execution_risk"], 1.0),
                ]
            )
        )
        return risk

    async def _calculate_performance_metrics(self, predictions: List[ModelPrediction]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        unique_predictions = len(set(p.prediction for p in predictions))
        metrics["prediction_diversity"] = unique_predictions / 3.0
        exec_times = [float(p.execution_time_ms) for p in predictions]
        metrics["avg_execution_time_ms"] = float(np.mean(exec_times)) if exec_times else 0.0
        metrics["max_execution_time_ms"] = float(max(exec_times)) if exec_times else 0.0
        confidences = [float(p.confidence) for p in predictions]
        metrics["avg_confidence"] = float(np.mean(confidences)) if confidences else 0.0
        metrics["min_confidence"] = float(min(confidences)) if confidences else 0.0
        metrics["max_confidence"] = float(max(confidences)) if confidences else 0.0
        metrics["models_participated"] = float(len(predictions))
        metrics["participation_rate"] = float(len(predictions) / len(self.models)) if self.models else 0.0
        return metrics

    async def update_model_performance(self, model_id: str, actual_outcome: str, predicted_outcome: str, return_achieved: float) -> None:
        if model_id not in self.models:
            return
        self.performance_history[model_id].append(
            {
                "timestamp": datetime.utcnow(),
                "predicted": predicted_outcome,
                "actual": actual_outcome,
                "return": float(return_achieved),
                "correct": predicted_outcome == actual_outcome,
            }
        )
        history = list(self.performance_history[model_id])
        if len(history) >= 10:
            recent = history[-self.performance_window :]
            correct_predictions = [1.0 if h["correct"] else 0.0 for h in recent]
            returns = [float(h["return"]) for h in recent]
            self.model_performance[model_id]["accuracy"] = float(np.mean(correct_predictions))
            self.model_performance[model_id]["sharpe_ratio"] = float(
                (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(252)
            )
            self.model_performance[model_id]["total_predictions"] += 1
            model_type = self.models[model_id]["type"]
            self._update_model_weight(model_type, self.model_performance[model_id]["accuracy"])

    def _update_model_weight(self, model_type: ModelType, performance: float) -> None:
        current_weight = float(self.model_weights.get(model_type, 0.1))
        if performance > 0.6:
            adjustment = self.weight_learning_rate
        elif performance < 0.4:
            adjustment = -self.weight_learning_rate
        else:
            adjustment = 0.0
        new_weight = float(np.clip(current_weight + adjustment, self.min_weight, self.max_weight))
        self.model_weights[model_type] = new_weight
        total_weight = float(sum(self.model_weights.values()))
        if total_weight > 0:
            self.model_weights = {k: v / total_weight for k, v in self.model_weights.items()}

    def _create_default_prediction(self) -> EnsemblePrediction:
        return EnsemblePrediction(
            final_prediction="HOLD",
            confidence=0.0,
            consensus_level=0.0,
            model_contributions={},
            probability_distribution={"BUY": 0.33, "SELL": 0.33, "HOLD": 0.34},
            dissenting_models=[],
            ensemble_method=EnsembleMethod.VOTING,
            risk_assessment={"overall_risk": 1.0},
            performance_metrics={},
        )

    async def get_ensemble_diagnostics(self) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {
            "total_models": len(self.models),
            "model_weights": {k.value: v for k, v in self.model_weights.items()},
            "model_performance": dict(self.model_performance),
            "ensemble_health": "healthy" if len(self.models) >= 3 else "degraded",
            "recommendations": [],
        }
        for model_id, perf in self.model_performance.items():
            acc = perf.get("accuracy", 0.0)
            if acc < 0.4:
                diagnostics["recommendations"].append(
                    f"Consider retraining or removing model {model_id} (accuracy: {acc:.2%})"
                )
            elif acc > 0.7:
                diagnostics["recommendations"].append(
                    f"Model {model_id} performing well (accuracy: {acc:.2%})"
                )
        if len(set(self.model_weights.values())) < 3:
            diagnostics["recommendations"].append(
                "Low weight diversity - consider adding more diverse models"
            )
        return diagnostics

    async def generate_signals(self, symbols: List[str], strategies: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        base_context: Dict[str, Any] = {
            "regime": "unknown",
            "volatility": 0.2,
            "volume_ratio": 1.0,
            "rsi": 50,
            "trend_strength": 0.5,
            "data_quality": 1.0,
        }
        for sym in symbols:
            md = dict(base_context)
            md["symbol"] = sym
            pred = await self.generate_ensemble_prediction(md, EnsembleMethod.DYNAMIC)
            results[sym] = {
                "prediction": pred.final_prediction,
                "confidence": pred.confidence,
                "consensus": pred.consensus_level,
                "probabilities": pred.probability_distribution,
                "risk": pred.risk_assessment,
                "method": pred.ensemble_method.value,
                "timestamp": datetime.utcnow().isoformat(),
            }
        return results


# Global instance used by main.py
ensemble_coordinator = EnsembleIntelligenceCoordinator()


async def get_ensemble_coordinator() -> EnsembleIntelligenceCoordinator:
    return ensemble_coordinator


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
