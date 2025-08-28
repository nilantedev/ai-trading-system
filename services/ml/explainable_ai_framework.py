#!/usr/bin/env python3
"""
Explainable AI Framework for Trading Decisions
Provides comprehensive interpretability for all AI/ML models with
SHAP, LIME, attention visualization, and decision trees.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import logging
import shap
import lime
import lime.lime_tabular
from sklearn.tree import DecisionTreeRegressor, export_text
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import (
    IntegratedGradients, LayerConductance, NeuronConductance,
    DeepLift, GradientShap, InputXGradient, Saliency
)
import json

from trading_common import get_logger, get_settings

logger = get_logger(__name__)
settings = get_settings()


class ExplainableAIFramework:
    """
    Comprehensive explainability framework for all trading models.
    """
    
    def __init__(self):
        self.shap_explainers = {}
        self.lime_explainers = {}
        self.feature_importance_cache = {}
        self.explanation_history = []
        
    async def explain_prediction(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str],
        model_type: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a single prediction.
        """
        explanation = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'prediction': None,
            'confidence': None,
            'feature_contributions': {},
            'top_factors': [],
            'decision_path': None,
            'counterfactuals': [],
            'visual_explanation': None
        }
        
        # Get prediction
        if isinstance(model, nn.Module):
            prediction = self._get_nn_prediction(model, features)
            explanation['prediction'] = float(prediction[0])
            explanation['confidence'] = float(prediction[1]) if len(prediction) > 1 else 0.9
            
            # Neural network explanations
            nn_explanations = await self._explain_neural_network(
                model, features, feature_names
            )
            explanation.update(nn_explanations)
            
        else:
            # Tree-based or linear model
            prediction = model.predict(features.reshape(1, -1))[0]
            explanation['prediction'] = float(prediction)
            
            # SHAP explanations
            shap_values = await self._get_shap_explanation(
                model, features, feature_names
            )
            explanation['feature_contributions'] = shap_values
            
            # LIME explanations
            lime_exp = await self._get_lime_explanation(
                model, features, feature_names
            )
            explanation['lime_explanation'] = lime_exp
        
        # Get top contributing factors
        explanation['top_factors'] = self._get_top_factors(
            explanation['feature_contributions'], n=5
        )
        
        # Generate natural language explanation
        explanation['natural_language'] = self._generate_natural_language_explanation(
            explanation
        )
        
        # Generate counterfactuals
        explanation['counterfactuals'] = await self._generate_counterfactuals(
            model, features, feature_names, prediction
        )
        
        # Decision rules
        explanation['decision_rules'] = self._extract_decision_rules(
            model, features, feature_names
        )
        
        # Store in history
        self.explanation_history.append(explanation)
        
        return explanation
    
    async def _explain_neural_network(
        self,
        model: nn.Module,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Explain neural network predictions using Captum.
        """
        model.eval()
        inputs = torch.FloatTensor(features).unsqueeze(0)
        
        # Integrated Gradients
        ig = IntegratedGradients(model)
        attributions = ig.attribute(inputs, n_steps=50)
        
        # Convert to feature contributions
        feature_contributions = {}
        attr_values = attributions.squeeze().detach().numpy()
        for i, name in enumerate(feature_names):
            feature_contributions[name] = float(attr_values[i])
        
        # Get attention weights if model has attention
        attention_weights = self._get_attention_weights(model, inputs)
        
        # Layer-wise relevance propagation
        lrp_scores = await self._layer_wise_relevance_propagation(
            model, inputs, feature_names
        )
        
        return {
            'feature_contributions': feature_contributions,
            'attention_weights': attention_weights,
            'layer_relevance': lrp_scores,
            'gradient_based': self._gradient_based_explanation(model, inputs, feature_names)
        }
    
    def _get_attention_weights(
        self,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract attention weights from model if available."""
        attention_weights = {}
        
        # Hook to capture attention weights
        def hook_fn(module, input, output):
            if hasattr(module, 'attention_weights'):
                attention_weights['attention'] = output[1].detach().numpy()
        
        # Register hooks
        hooks = []
        for module in model.modules():
            if 'Attention' in module.__class__.__name__:
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        with torch.no_grad():
            _ = model(inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_weights if attention_weights else None
    
    async def _get_shap_explanation(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Get SHAP values for feature importance.
        """
        model_id = id(model)
        
        # Create or retrieve explainer
        if model_id not in self.shap_explainers:
            # Use appropriate explainer based on model type
            if hasattr(model, 'predict_proba'):
                self.shap_explainers[model_id] = shap.TreeExplainer(model)
            else:
                # Use KernelExplainer for black-box models
                self.shap_explainers[model_id] = shap.KernelExplainer(
                    model.predict,
                    shap.kmeans(features.reshape(-1, len(feature_names)), 10)
                )
        
        explainer = self.shap_explainers[model_id]
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(features.reshape(1, -1))
        
        # Convert to dictionary
        feature_contributions = {}
        for i, name in enumerate(feature_names):
            feature_contributions[name] = float(shap_values[0][i])
        
        return feature_contributions
    
    async def _get_lime_explanation(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Get LIME explanation for local interpretability.
        """
        model_id = id(model)
        
        # Create or retrieve explainer
        if model_id not in self.lime_explainers:
            self.lime_explainers[model_id] = lime.lime_tabular.LimeTabularExplainer(
                features.reshape(-1, len(feature_names)),
                feature_names=feature_names,
                mode='regression'
            )
        
        explainer = self.lime_explainers[model_id]
        
        # Get explanation
        exp = explainer.explain_instance(
            features.flatten(),
            model.predict,
            num_features=len(feature_names)
        )
        
        # Convert to dictionary
        lime_explanation = {}
        for feature_idx, weight in exp.as_list():
            lime_explanation[feature_names[feature_idx]] = weight
        
        return lime_explanation
    
    async def _generate_counterfactuals(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str],
        original_prediction: float
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactual explanations.
        """
        counterfactuals = []
        
        # For each feature, find minimal change needed to flip prediction
        for i, feature_name in enumerate(feature_names):
            # Create copy of features
            modified_features = features.copy()
            
            # Binary search for minimal change
            low, high = features[i] * 0.5, features[i] * 1.5
            threshold = None
            
            for _ in range(10):  # Binary search iterations
                mid = (low + high) / 2
                modified_features[i] = mid
                
                # Get new prediction
                if isinstance(model, nn.Module):
                    new_pred = self._get_nn_prediction(model, modified_features)[0]
                else:
                    new_pred = model.predict(modified_features.reshape(1, -1))[0]
                
                # Check if prediction changed significantly
                if abs(new_pred - original_prediction) > abs(original_prediction) * 0.2:
                    threshold = mid
                    high = mid
                else:
                    low = mid
            
            if threshold is not None:
                change_pct = (threshold - features[i]) / (features[i] + 1e-8) * 100
                counterfactuals.append({
                    'feature': feature_name,
                    'original_value': float(features[i]),
                    'counterfactual_value': float(threshold),
                    'change_percent': float(change_pct),
                    'impact': 'Would change prediction significantly'
                })
        
        # Sort by smallest change needed
        counterfactuals.sort(key=lambda x: abs(x['change_percent']))
        
        return counterfactuals[:5]  # Return top 5
    
    def _extract_decision_rules(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str]
    ) -> List[str]:
        """
        Extract human-readable decision rules.
        """
        rules = []
        
        # Fit a simple decision tree to approximate the model
        if not isinstance(model, DecisionTreeRegressor):
            # Create surrogate tree
            surrogate = DecisionTreeRegressor(max_depth=4)
            
            # Generate synthetic data around the instance
            synthetic_X = np.random.randn(1000, len(features)) * 0.1 + features
            
            # Get predictions from original model
            if isinstance(model, nn.Module):
                synthetic_y = np.array([
                    self._get_nn_prediction(model, x)[0] for x in synthetic_X
                ])
            else:
                synthetic_y = model.predict(synthetic_X)
            
            # Fit surrogate
            surrogate.fit(synthetic_X, synthetic_y)
            
            # Extract rules
            tree_rules = export_text(surrogate, feature_names=feature_names)
            rules = tree_rules.split('\n')[:10]  # Top 10 rules
        
        # Add statistical rules
        for i, name in enumerate(feature_names):
            value = features[i]
            if value > np.percentile(features, 90):
                rules.append(f"{name} is very high ({value:.2f})")
            elif value < np.percentile(features, 10):
                rules.append(f"{name} is very low ({value:.2f})")
        
        return rules
    
    def _generate_natural_language_explanation(
        self,
        explanation: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable explanation.
        """
        prediction = explanation['prediction']
        confidence = explanation.get('confidence', 0.5)
        top_factors = explanation['top_factors']
        
        # Build explanation
        nl_explanation = []
        
        # Prediction summary
        if prediction > 0:
            nl_explanation.append(
                f"The model predicts a POSITIVE return of {prediction:.2%} "
                f"with {confidence:.1%} confidence."
            )
        else:
            nl_explanation.append(
                f"The model predicts a NEGATIVE return of {prediction:.2%} "
                f"with {confidence:.1%} confidence."
            )
        
        # Top factors
        nl_explanation.append("\nKey factors influencing this prediction:")
        
        for factor in top_factors[:3]:
            feature = factor['feature']
            contribution = factor['contribution']
            
            if contribution > 0:
                nl_explanation.append(
                    f"• {feature} has a POSITIVE impact ({contribution:.3f})"
                )
            else:
                nl_explanation.append(
                    f"• {feature} has a NEGATIVE impact ({contribution:.3f})"
                )
        
        # Counterfactuals
        if explanation.get('counterfactuals'):
            cf = explanation['counterfactuals'][0]
            nl_explanation.append(
                f"\nSmallest change for different outcome: "
                f"Adjust {cf['feature']} by {cf['change_percent']:.1f}%"
            )
        
        return '\n'.join(nl_explanation)
    
    def _get_top_factors(
        self,
        feature_contributions: Dict[str, float],
        n: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top contributing factors."""
        if not feature_contributions:
            return []
        
        # Sort by absolute contribution
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return [
            {
                'feature': name,
                'contribution': value,
                'impact': 'positive' if value > 0 else 'negative'
            }
            for name, value in sorted_features[:n]
        ]
    
    def _get_nn_prediction(
        self,
        model: nn.Module,
        features: np.ndarray
    ) -> Tuple[float, float]:
        """Get prediction from neural network."""
        model.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(features).unsqueeze(0)
            outputs = model(inputs)
            
            if isinstance(outputs, tuple):
                prediction = outputs[0].item()
                confidence = outputs[1].item() if len(outputs) > 1 else 0.9
            else:
                prediction = outputs.item()
                confidence = 0.9
        
        return prediction, confidence
    
    async def _layer_wise_relevance_propagation(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Layer-wise relevance propagation for deep networks.
        """
        relevance_scores = {}
        
        # Forward pass to get activations
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output.detach())
        
        hooks = []
        for layer in model.children():
            hooks.append(layer.register_forward_hook(hook_fn))
        
        output = model(inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Backward pass for relevance
        relevance = output
        
        for i in range(len(activations) - 1, 0, -1):
            # Propagate relevance backward
            activation = activations[i]
            relevance = relevance * activation / (activation.sum() + 1e-10)
        
        # Map to input features
        input_relevance = relevance.squeeze().detach().numpy()
        
        for i, name in enumerate(feature_names[:len(input_relevance)]):
            relevance_scores[name] = float(input_relevance[i])
        
        return relevance_scores
    
    def _gradient_based_explanation(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Gradient-based feature importance.
        """
        inputs.requires_grad = True
        
        # Forward pass
        output = model(inputs)
        if isinstance(output, tuple):
            output = output[0]
        
        # Backward pass
        output.backward()
        
        # Get gradients
        gradients = inputs.grad.squeeze().detach().numpy()
        
        # Calculate importance as gradient * input
        importance = gradients * inputs.squeeze().detach().numpy()
        
        gradient_importance = {}
        for i, name in enumerate(feature_names):
            gradient_importance[name] = float(importance[i])
        
        return gradient_importance
    
    async def generate_global_explanation(
        self,
        model: Any,
        data: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Generate global model explanation.
        """
        logger.info("Generating global model explanation")
        
        # Global SHAP values
        if hasattr(model, 'predict'):
            explainer = shap.Explainer(model.predict, data[feature_names])
            shap_values = explainer(data[feature_names])
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(shap_values.values).mean(axis=0)
            }).sort_values('importance', ascending=False)
        else:
            feature_importance = pd.DataFrame()
        
        # Partial dependence plots data
        pdp_data = {}
        for feature in feature_names[:5]:  # Top 5 features
            pdp_data[feature] = self._calculate_partial_dependence(
                model, data, feature, feature_names
            )
        
        # Feature interactions
        interactions = self._calculate_feature_interactions(
            model, data, feature_names
        )
        
        return {
            'feature_importance': feature_importance.to_dict('records'),
            'partial_dependence': pdp_data,
            'feature_interactions': interactions,
            'model_complexity': self._estimate_model_complexity(model),
            'explanation_summary': self._generate_global_summary(
                feature_importance, interactions
            )
        }
    
    def _calculate_partial_dependence(
        self,
        model: Any,
        data: pd.DataFrame,
        feature: str,
        feature_names: List[str]
    ) -> Dict[str, List[float]]:
        """Calculate partial dependence for a feature."""
        # Create grid
        feature_values = np.linspace(
            data[feature].min(),
            data[feature].max(),
            50
        )
        
        predictions = []
        for value in feature_values:
            # Set feature to fixed value
            X_modified = data[feature_names].copy()
            X_modified[feature] = value
            
            # Get predictions
            if isinstance(model, nn.Module):
                preds = [
                    self._get_nn_prediction(model, row.values)[0]
                    for _, row in X_modified.iterrows()
                ]
            else:
                preds = model.predict(X_modified)
            
            predictions.append(np.mean(preds))
        
        return {
            'values': feature_values.tolist(),
            'predictions': predictions
        }
    
    def _calculate_feature_interactions(
        self,
        model: Any,
        data: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Calculate feature interaction strengths."""
        interactions = {}
        
        # Calculate H-statistic for top feature pairs
        for i, f1 in enumerate(feature_names[:5]):
            for f2 in feature_names[i+1:6]:
                interaction_strength = self._h_statistic(
                    model, data, f1, f2, feature_names
                )
                interactions[f"{f1}_x_{f2}"] = interaction_strength
        
        return interactions
    
    def _h_statistic(
        self,
        model: Any,
        data: pd.DataFrame,
        feature1: str,
        feature2: str,
        feature_names: List[str]
    ) -> float:
        """Calculate H-statistic for feature interaction."""
        # Simplified H-statistic calculation
        n_samples = min(100, len(data))
        sample_data = data.sample(n_samples)
        
        # Get predictions with both features
        if isinstance(model, nn.Module):
            full_preds = [
                self._get_nn_prediction(model, row[feature_names].values)[0]
                for _, row in sample_data.iterrows()
            ]
        else:
            full_preds = model.predict(sample_data[feature_names])
        
        # Get predictions with feature1 permuted
        data_perm1 = sample_data.copy()
        data_perm1[feature1] = np.random.permutation(data_perm1[feature1])
        
        if isinstance(model, nn.Module):
            perm1_preds = [
                self._get_nn_prediction(model, row[feature_names].values)[0]
                for _, row in data_perm1.iterrows()
            ]
        else:
            perm1_preds = model.predict(data_perm1[feature_names])
        
        # Calculate interaction strength
        interaction = np.var(full_preds - perm1_preds)
        
        return float(interaction)
    
    def _estimate_model_complexity(self, model: Any) -> Dict[str, Any]:
        """Estimate model complexity metrics."""
        complexity = {}
        
        if isinstance(model, nn.Module):
            # Neural network complexity
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            complexity = {
                'type': 'neural_network',
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'layers': len(list(model.children()))
            }
        elif hasattr(model, 'n_estimators'):
            # Ensemble model
            complexity = {
                'type': 'ensemble',
                'n_estimators': model.n_estimators,
                'max_depth': getattr(model, 'max_depth', None)
            }
        else:
            complexity = {
                'type': 'unknown',
                'complexity': 'medium'
            }
        
        return complexity
    
    def _generate_global_summary(
        self,
        feature_importance: pd.DataFrame,
        interactions: Dict[str, float]
    ) -> str:
        """Generate summary of global explanation."""
        summary = []
        
        # Top features
        if not feature_importance.empty:
            top_features = feature_importance.head(3)['feature'].tolist()
            summary.append(
                f"The model primarily relies on: {', '.join(top_features)}"
            )
        
        # Strong interactions
        if interactions:
            strong_interactions = [
                k for k, v in interactions.items()
                if v > np.mean(list(interactions.values()))
            ]
            if strong_interactions:
                summary.append(
                    f"Strong feature interactions detected: {', '.join(strong_interactions[:3])}"
                )
        
        return '\n'.join(summary)


# Global explainability framework
_explainable_ai: Optional[ExplainableAIFramework] = None


async def get_explainable_ai() -> ExplainableAIFramework:
    """Get or create explainable AI framework."""
    global _explainable_ai
    if _explainable_ai is None:
        _explainable_ai = ExplainableAIFramework()
    return _explainable_ai