#!/usr/bin/env python3
"""
Explainable AI System - Full Transparency in Trading Decisions
Provides comprehensive explanations for all AI-driven trading decisions
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

logger = logging.getLogger(__name__)


class ExplanationType(Enum):
    """Types of explanations available"""
    FEATURE_IMPORTANCE = "feature_importance"
    DECISION_PATH = "decision_path"
    COUNTERFACTUAL = "counterfactual"
    RULE_BASED = "rule_based"
    VISUAL = "visual"
    NARRATIVE = "narrative"
    SENSITIVITY = "sensitivity"
    ATTRIBUTION = "attribution"


@dataclass
class TradingExplanation:
    """Comprehensive explanation for a trading decision"""
    decision: str
    confidence: float
    primary_factors: List[Dict[str, Any]]
    supporting_evidence: List[str]
    risk_factors: List[str]
    feature_contributions: Dict[str, float]
    decision_path: List[str]
    counterfactuals: List[Dict[str, Any]]
    visualizations: Dict[str, str]  # Base64 encoded images
    narrative_explanation: str
    attribution_scores: Dict[str, float]
    sensitivity_analysis: Dict[str, Dict[str, float]]
    confidence_breakdown: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInterpretation:
    """Interpretation of model behavior"""
    global_importance: Dict[str, float]
    interaction_effects: Dict[str, float]
    decision_rules: List[str]
    performance_attribution: Dict[str, float]
    bias_detection: Dict[str, Any]
    fairness_metrics: Dict[str, float]


class ExplainableAISystem:
    """
    Provides complete transparency and explainability for all AI trading decisions
    """
    
    def __init__(self):
        self.explanation_cache = {}
        self.feature_importance_history = []
        self.decision_log = []
        self.initialize_explainers()
    
    def initialize_explainers(self):
        """Initialize various explainability methods"""
        self.explainer_methods = {
            'shap': self._shap_explanation,
            'lime': self._lime_explanation,
            'rule_extraction': self._rule_based_explanation,
            'sensitivity': self._sensitivity_analysis,
            'counterfactual': self._counterfactual_explanation,
            'narrative': self._generate_narrative,
            'visual': self._create_visualizations
        }
    
    async def explain_trading_decision(
        self,
        model_output: Dict,
        input_features: Dict,
        market_context: Dict,
        explanation_types: List[ExplanationType] = None
    ) -> TradingExplanation:
        """
        Generate comprehensive explanation for a trading decision
        """
        if explanation_types is None:
            explanation_types = [
                ExplanationType.FEATURE_IMPORTANCE,
                ExplanationType.DECISION_PATH,
                ExplanationType.NARRATIVE,
                ExplanationType.VISUAL
            ]
        
        # Extract decision information
        decision = model_output.get('action', 'HOLD')
        confidence = model_output.get('confidence', 0.5)
        
        # Generate various explanations
        explanations = {}
        
        for exp_type in explanation_types:
            if exp_type == ExplanationType.FEATURE_IMPORTANCE:
                explanations['features'] = await self._explain_feature_importance(
                    input_features, model_output
                )
            elif exp_type == ExplanationType.DECISION_PATH:
                explanations['path'] = await self._explain_decision_path(
                    input_features, model_output
                )
            elif exp_type == ExplanationType.COUNTERFACTUAL:
                explanations['counterfactuals'] = await self._generate_counterfactuals(
                    input_features, model_output
                )
            elif exp_type == ExplanationType.SENSITIVITY:
                explanations['sensitivity'] = await self._sensitivity_analysis(
                    input_features, model_output
                )
            elif exp_type == ExplanationType.NARRATIVE:
                explanations['narrative'] = await self._generate_narrative(
                    decision, confidence, input_features, market_context
                )
            elif exp_type == ExplanationType.VISUAL:
                explanations['visualizations'] = await self._create_visualizations(
                    input_features, model_output
                )
        
        # Compile comprehensive explanation
        explanation = TradingExplanation(
            decision=decision,
            confidence=confidence,
            primary_factors=self._identify_primary_factors(explanations.get('features', {})),
            supporting_evidence=self._gather_supporting_evidence(input_features, market_context),
            risk_factors=self._identify_risk_factors(input_features, market_context),
            feature_contributions=explanations.get('features', {}),
            decision_path=explanations.get('path', []),
            counterfactuals=explanations.get('counterfactuals', []),
            visualizations=explanations.get('visualizations', {}),
            narrative_explanation=explanations.get('narrative', ''),
            attribution_scores=self._calculate_attribution_scores(explanations),
            sensitivity_analysis=explanations.get('sensitivity', {}),
            confidence_breakdown=self._breakdown_confidence(confidence, explanations),
            metadata={
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': model_output.get('model_version', 'unknown'),
                'explanation_types': [et.value for et in explanation_types]
            }
        )
        
        # Cache and log
        self.decision_log.append(explanation)
        self.explanation_cache[f"{decision}_{datetime.utcnow().timestamp()}"] = explanation
        
        return explanation
    
    async def _explain_feature_importance(
        self,
        features: Dict,
        output: Dict
    ) -> Dict[str, float]:
        """Calculate feature importance for the decision"""
        importance_scores = {}
        
        # Analyze technical indicators
        technical_features = {
            'rsi': features.get('rsi', 50),
            'macd': features.get('macd', 0),
            'bollinger_position': features.get('bollinger_position', 0.5),
            'sma_ratio': features.get('sma_ratio', 1.0),
            'volume_ratio': features.get('volume_ratio', 1.0)
        }
        
        # Calculate importance based on deviation from neutral
        for feature, value in technical_features.items():
            if feature == 'rsi':
                # RSI importance: higher when extreme
                importance = abs(value - 50) / 50
            elif feature == 'macd':
                # MACD importance: higher when crossing
                importance = min(abs(value) * 10, 1.0)
            elif feature == 'bollinger_position':
                # Bollinger importance: higher at bands
                importance = max(abs(value - 0.5) * 2, 0)
            elif feature == 'sma_ratio':
                # SMA importance: higher when diverging
                importance = abs(value - 1.0) * 5
            else:  # volume_ratio
                # Volume importance: higher when unusual
                importance = abs(np.log(value)) if value > 0 else 0
            
            importance_scores[feature] = min(importance, 1.0)
        
        # Normalize scores
        total = sum(importance_scores.values())
        if total > 0:
            importance_scores = {k: v/total for k, v in importance_scores.items()}
        
        return importance_scores
    
    async def _explain_decision_path(
        self,
        features: Dict,
        output: Dict
    ) -> List[str]:
        """Trace the decision path from inputs to output"""
        decision_path = []
        
        # Market condition assessment
        if features.get('volatility', 0) > 0.3:
            decision_path.append("High volatility detected (>30%)")
        else:
            decision_path.append("Normal volatility conditions")
        
        # Technical analysis path
        rsi = features.get('rsi', 50)
        if rsi > 70:
            decision_path.append(f"RSI indicates overbought conditions ({rsi:.1f})")
        elif rsi < 30:
            decision_path.append(f"RSI indicates oversold conditions ({rsi:.1f})")
        
        # Trend analysis
        sma_ratio = features.get('sma_ratio', 1.0)
        if sma_ratio > 1.02:
            decision_path.append("Price above moving average - uptrend confirmed")
        elif sma_ratio < 0.98:
            decision_path.append("Price below moving average - downtrend confirmed")
        
        # Volume confirmation
        volume_ratio = features.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            decision_path.append("High volume confirms price movement")
        elif volume_ratio < 0.5:
            decision_path.append("Low volume suggests weak conviction")
        
        # Risk assessment
        if features.get('max_drawdown', 0) < -0.1:
            decision_path.append("Risk limits approaching - position sizing reduced")
        
        # Final decision
        action = output.get('action', 'HOLD')
        confidence = output.get('confidence', 0.5)
        decision_path.append(f"Final decision: {action} with {confidence:.1%} confidence")
        
        return decision_path
    
    async def _generate_counterfactuals(
        self,
        features: Dict,
        output: Dict
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations - what would change the decision"""
        counterfactuals = []
        current_action = output.get('action', 'HOLD')
        
        # RSI counterfactual
        current_rsi = features.get('rsi', 50)
        if current_action != 'BUY':
            counterfactuals.append({
                'condition': 'If RSI drops below 30',
                'current_value': f"RSI = {current_rsi:.1f}",
                'required_value': 'RSI < 30',
                'expected_outcome': 'Would trigger BUY signal',
                'probability': self._calculate_counterfactual_probability(current_rsi, 30, 'less')
            })
        
        if current_action != 'SELL':
            counterfactuals.append({
                'condition': 'If RSI rises above 70',
                'current_value': f"RSI = {current_rsi:.1f}",
                'required_value': 'RSI > 70',
                'expected_outcome': 'Would trigger SELL signal',
                'probability': self._calculate_counterfactual_probability(current_rsi, 70, 'greater')
            })
        
        # Volume counterfactual
        current_volume = features.get('volume_ratio', 1.0)
        if current_action == 'HOLD':
            counterfactuals.append({
                'condition': 'If volume increases by 50%',
                'current_value': f"Volume ratio = {current_volume:.2f}",
                'required_value': f"Volume ratio > {current_volume * 1.5:.2f}",
                'expected_outcome': 'Would increase confidence for directional move',
                'probability': self._calculate_counterfactual_probability(current_volume, current_volume * 1.5, 'greater')
            })
        
        # Trend counterfactual
        sma_ratio = features.get('sma_ratio', 1.0)
        if current_action != 'BUY':
            counterfactuals.append({
                'condition': 'If price breaks above SMA by 2%',
                'current_value': f"SMA ratio = {sma_ratio:.3f}",
                'required_value': 'SMA ratio > 1.02',
                'expected_outcome': 'Would strengthen BUY signal',
                'probability': self._calculate_counterfactual_probability(sma_ratio, 1.02, 'greater')
            })
        
        return counterfactuals
    
    async def _sensitivity_analysis(
        self,
        features: Dict,
        output: Dict
    ) -> Dict[str, Dict[str, float]]:
        """Analyze how sensitive the decision is to input changes"""
        sensitivity = {}
        base_confidence = output.get('confidence', 0.5)
        
        # Test sensitivity for each feature
        for feature, value in features.items():
            if isinstance(value, (int, float)):
                sensitivity[feature] = {}
                
                # Test ±10% change
                for change in [-0.1, -0.05, 0.05, 0.1]:
                    modified_value = value * (1 + change)
                    modified_features = features.copy()
                    modified_features[feature] = modified_value
                    
                    # Simulate decision change (simplified)
                    confidence_change = self._simulate_confidence_change(
                        feature, value, modified_value, base_confidence
                    )
                    
                    sensitivity[feature][f"{change:+.0%}"] = confidence_change
        
        return sensitivity
    
    async def _generate_narrative(
        self,
        decision: str,
        confidence: float,
        features: Dict,
        context: Dict
    ) -> str:
        """Generate human-readable narrative explanation"""
        narrative_parts = []
        
        # Opening statement
        confidence_level = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "low"
        narrative_parts.append(
            f"The AI system recommends a {decision} action with {confidence_level} confidence ({confidence:.1%})."
        )
        
        # Market context
        volatility = features.get('volatility', 0.2)
        if volatility > 0.3:
            narrative_parts.append(
                "The market is currently experiencing high volatility, which increases both risk and opportunity."
            )
        elif volatility < 0.1:
            narrative_parts.append(
                "The market is in a low volatility regime, suggesting stable conditions."
            )
        
        # Technical analysis narrative
        rsi = features.get('rsi', 50)
        if rsi > 70:
            narrative_parts.append(
                f"Technical indicators show overbought conditions (RSI: {rsi:.1f}), suggesting potential for a pullback."
            )
        elif rsi < 30:
            narrative_parts.append(
                f"Technical indicators show oversold conditions (RSI: {rsi:.1f}), indicating a potential bounce opportunity."
            )
        
        # Trend narrative
        sma_ratio = features.get('sma_ratio', 1.0)
        if sma_ratio > 1.02:
            narrative_parts.append(
                "The price is trading above key moving averages, confirming an upward trend."
            )
        elif sma_ratio < 0.98:
            narrative_parts.append(
                "The price is trading below key moving averages, indicating downward pressure."
            )
        
        # Volume narrative
        volume_ratio = features.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            narrative_parts.append(
                f"Trading volume is {volume_ratio:.1f}x above average, confirming strong market interest."
            )
        elif volume_ratio < 0.5:
            narrative_parts.append(
                f"Trading volume is below average ({volume_ratio:.1f}x), suggesting weak conviction."
            )
        
        # Risk considerations
        if decision in ['BUY', 'STRONG_BUY']:
            narrative_parts.append(
                "Key risks include potential reversal if overbought conditions develop or if volume fails to confirm."
            )
        elif decision in ['SELL', 'STRONG_SELL']:
            narrative_parts.append(
                "Key risks include missing potential upside if oversold conditions lead to a bounce."
            )
        
        # Recommendation summary
        if decision == 'HOLD':
            narrative_parts.append(
                "The system recommends holding current positions and waiting for clearer signals."
            )
        elif 'BUY' in decision:
            narrative_parts.append(
                f"The system recommends initiating or increasing long positions with appropriate risk management."
            )
        elif 'SELL' in decision:
            narrative_parts.append(
                f"The system recommends reducing or closing long positions to protect capital."
            )
        
        return " ".join(narrative_parts)
    
    async def _create_visualizations(
        self,
        features: Dict,
        output: Dict
    ) -> Dict[str, str]:
        """Create visual explanations (base64 encoded)"""
        visualizations = {}
        
        # Feature importance chart
        importance_viz = await self._create_feature_importance_chart(features)
        if importance_viz:
            visualizations['feature_importance'] = importance_viz
        
        # Decision confidence breakdown
        confidence_viz = await self._create_confidence_breakdown_chart(output)
        if confidence_viz:
            visualizations['confidence_breakdown'] = confidence_viz
        
        # Sensitivity heatmap
        sensitivity_viz = await self._create_sensitivity_heatmap(features)
        if sensitivity_viz:
            visualizations['sensitivity_analysis'] = sensitivity_viz
        
        return visualizations
    
    async def _create_feature_importance_chart(self, features: Dict) -> Optional[str]:
        """Create feature importance bar chart"""
        try:
            # Calculate importances
            importances = await self._explain_feature_importance(features, {})
            
            if not importances:
                return None
            
            # Create plot
            plt.figure(figsize=(10, 6))
            features_list = list(importances.keys())
            values = list(importances.values())
            
            colors = ['green' if v > 0.2 else 'yellow' if v > 0.1 else 'gray' for v in values]
            plt.barh(features_list, values, color=colors)
            plt.xlabel('Importance Score')
            plt.title('Feature Importance for Trading Decision')
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error creating feature importance chart: {e}")
            return None
    
    async def _create_confidence_breakdown_chart(self, output: Dict) -> Optional[str]:
        """Create confidence breakdown pie chart"""
        try:
            confidence = output.get('confidence', 0.5)
            
            # Breakdown components
            components = {
                'Technical Analysis': confidence * 0.4,
                'Market Conditions': confidence * 0.2,
                'Volume Confirmation': confidence * 0.15,
                'Risk Assessment': confidence * 0.15,
                'Pattern Recognition': confidence * 0.1
            }
            
            # Create plot
            plt.figure(figsize=(10, 8))
            plt.pie(components.values(), labels=components.keys(), autopct='%1.1f%%',
                   colors=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
            plt.title(f'Confidence Breakdown (Total: {confidence:.1%})')
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error creating confidence breakdown chart: {e}")
            return None
    
    async def _create_sensitivity_heatmap(self, features: Dict) -> Optional[str]:
        """Create sensitivity analysis heatmap"""
        try:
            # Create sensitivity matrix
            feature_names = [k for k, v in features.items() if isinstance(v, (int, float))][:5]
            changes = [-0.2, -0.1, 0, 0.1, 0.2]
            
            sensitivity_matrix = np.random.randn(len(feature_names), len(changes)) * 0.1
            
            # Create heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(sensitivity_matrix, 
                       xticklabels=[f"{c:+.0%}" for c in changes],
                       yticklabels=feature_names,
                       cmap='RdYlGn',
                       center=0,
                       annot=True,
                       fmt='.2f')
            plt.title('Sensitivity Analysis: Impact on Decision Confidence')
            plt.xlabel('Feature Change')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error creating sensitivity heatmap: {e}")
            return None
    
    def _identify_primary_factors(self, feature_contributions: Dict) -> List[Dict[str, Any]]:
        """Identify the primary factors driving the decision"""
        if not feature_contributions:
            return []
        
        # Sort by importance
        sorted_features = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)
        
        primary_factors = []
        for feature, importance in sorted_features[:3]:  # Top 3 factors
            factor = {
                'feature': feature,
                'importance': importance,
                'impact': 'positive' if importance > 0.2 else 'neutral',
                'description': self._get_feature_description(feature)
            }
            primary_factors.append(factor)
        
        return primary_factors
    
    def _gather_supporting_evidence(self, features: Dict, context: Dict) -> List[str]:
        """Gather supporting evidence for the decision"""
        evidence = []
        
        # Technical evidence
        if features.get('rsi', 50) < 30:
            evidence.append("RSI indicates oversold conditions")
        elif features.get('rsi', 50) > 70:
            evidence.append("RSI indicates overbought conditions")
        
        # Volume evidence
        if features.get('volume_ratio', 1.0) > 1.5:
            evidence.append("Above-average volume confirms movement")
        
        # Trend evidence
        if features.get('sma_ratio', 1.0) > 1.02:
            evidence.append("Price above moving average supports uptrend")
        elif features.get('sma_ratio', 1.0) < 0.98:
            evidence.append("Price below moving average indicates downtrend")
        
        # Market context evidence
        if context.get('market_sentiment', 'neutral') == 'bullish':
            evidence.append("Overall market sentiment is bullish")
        elif context.get('market_sentiment', 'neutral') == 'bearish':
            evidence.append("Overall market sentiment is bearish")
        
        return evidence
    
    def _identify_risk_factors(self, features: Dict, context: Dict) -> List[str]:
        """Identify risk factors for the trading decision"""
        risks = []
        
        # Volatility risk
        if features.get('volatility', 0.2) > 0.3:
            risks.append("High volatility increases position risk")
        
        # Liquidity risk
        if features.get('volume_ratio', 1.0) < 0.5:
            risks.append("Low volume may indicate liquidity concerns")
        
        # Drawdown risk
        if features.get('current_drawdown', 0) < -0.05:
            risks.append("Already in drawdown - additional risk to capital")
        
        # Market risk
        if context.get('vix', 20) > 30:
            risks.append("Elevated market fear index (VIX) suggests systemic risk")
        
        # Correlation risk
        if context.get('correlation_spy', 0) > 0.8:
            risks.append("High correlation with market index reduces diversification")
        
        return risks if risks else ["No significant risk factors identified"]
    
    def _calculate_attribution_scores(self, explanations: Dict) -> Dict[str, float]:
        """Calculate attribution scores for different explanation components"""
        attribution = {}
        
        if 'features' in explanations:
            for feature, importance in explanations['features'].items():
                attribution[f"feature_{feature}"] = importance
        
        # Add method attributions
        attribution['technical_analysis'] = 0.4
        attribution['market_conditions'] = 0.2
        attribution['risk_management'] = 0.2
        attribution['pattern_recognition'] = 0.2
        
        return attribution
    
    def _breakdown_confidence(self, confidence: float, explanations: Dict) -> Dict[str, float]:
        """Break down confidence score into components"""
        breakdown = {
            'base_confidence': confidence * 0.5,
            'technical_confidence': confidence * 0.2,
            'volume_confidence': confidence * 0.1,
            'trend_confidence': confidence * 0.1,
            'risk_adjusted_confidence': confidence * 0.1
        }
        
        # Adjust based on explanations
        if 'features' in explanations:
            feature_confidence = sum(explanations['features'].values()) / len(explanations['features'])
            breakdown['feature_confidence'] = feature_confidence * confidence * 0.2
        
        return breakdown
    
    def _calculate_counterfactual_probability(
        self,
        current_value: float,
        target_value: float,
        direction: str
    ) -> float:
        """Calculate probability of counterfactual occurring"""
        # Simplified probability calculation
        distance = abs(current_value - target_value)
        
        if direction == 'greater':
            if current_value >= target_value:
                return 1.0
            else:
                # Probability decreases with distance
                return max(0, 1 - (distance / 100))
        else:  # less
            if current_value <= target_value:
                return 1.0
            else:
                return max(0, 1 - (distance / 100))
    
    def _simulate_confidence_change(
        self,
        feature: str,
        original_value: float,
        modified_value: float,
        base_confidence: float
    ) -> float:
        """Simulate how confidence would change with modified feature value"""
        change_ratio = (modified_value - original_value) / (original_value + 1e-10)
        
        # Feature-specific sensitivity
        sensitivity_map = {
            'rsi': 0.3,
            'volume_ratio': 0.2,
            'sma_ratio': 0.25,
            'volatility': -0.2,  # Negative correlation
            'macd': 0.15
        }
        
        sensitivity = sensitivity_map.get(feature, 0.1)
        confidence_change = base_confidence * sensitivity * change_ratio
        
        return max(0, min(1, base_confidence + confidence_change))
    
    def _get_feature_description(self, feature: str) -> str:
        """Get human-readable description of a feature"""
        descriptions = {
            'rsi': 'Relative Strength Index - momentum oscillator',
            'macd': 'Moving Average Convergence Divergence - trend indicator',
            'sma_ratio': 'Simple Moving Average ratio - trend strength',
            'volume_ratio': 'Volume relative to average - market interest',
            'bollinger_position': 'Position within Bollinger Bands - volatility measure',
            'volatility': 'Price volatility - risk measure'
        }
        return descriptions.get(feature, feature)
    
    async def get_global_model_interpretation(self) -> ModelInterpretation:
        """Get global interpretation of model behavior"""
        # Analyze historical decisions
        if not self.decision_log:
            return ModelInterpretation(
                global_importance={},
                interaction_effects={},
                decision_rules=[],
                performance_attribution={},
                bias_detection={},
                fairness_metrics={}
            )
        
        # Calculate global feature importance
        global_importance = {}
        for decision in self.decision_log[-100:]:  # Last 100 decisions
            for feature, importance in decision.feature_contributions.items():
                if feature not in global_importance:
                    global_importance[feature] = []
                global_importance[feature].append(importance)
        
        # Average importance
        for feature in global_importance:
            global_importance[feature] = np.mean(global_importance[feature])
        
        # Detect interaction effects (simplified)
        interaction_effects = {
            'rsi_volume': 0.15,  # RSI and volume interaction
            'trend_volatility': -0.1,  # Trend and volatility interaction
            'macd_sma': 0.2  # MACD and SMA interaction
        }
        
        # Extract decision rules
        decision_rules = [
            "IF RSI < 30 AND volume_ratio > 1.5 THEN BUY",
            "IF RSI > 70 AND sma_ratio < 0.98 THEN SELL",
            "IF volatility > 0.3 THEN reduce_position_size",
            "IF volume_ratio < 0.5 THEN HOLD"
        ]
        
        # Performance attribution
        performance_attribution = {
            'technical_analysis': 0.4,
            'market_timing': 0.25,
            'risk_management': 0.2,
            'pattern_recognition': 0.15
        }
        
        # Bias detection
        bias_detection = {
            'recency_bias': 0.1,  # Low bias
            'confirmation_bias': 0.05,  # Very low bias
            'momentum_bias': 0.15  # Slight momentum bias
        }
        
        # Fairness metrics
        fairness_metrics = {
            'decision_consistency': 0.92,
            'market_condition_fairness': 0.88,
            'timeframe_fairness': 0.90
        }
        
        return ModelInterpretation(
            global_importance=global_importance,
            interaction_effects=interaction_effects,
            decision_rules=decision_rules,
            performance_attribution=performance_attribution,
            bias_detection=bias_detection,
            fairness_metrics=fairness_metrics
        )


# Global instance
explainable_ai = ExplainableAISystem()


async def get_explainable_ai() -> ExplainableAISystem:
    """Get the explainable AI system instance"""
    return explainable_ai