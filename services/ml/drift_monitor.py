#!/usr/bin/env python3
"""
Model Drift Monitor - Detects when ML models are drifting from their baseline performance
Critical for maintaining model quality in production
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from scipy import stats
import json

logger = logging.getLogger(__name__)


class DriftSeverity(Enum):
    """Severity levels of model drift"""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Alert for detected drift"""
    model_id: str
    feature_name: str
    drift_type: str
    severity: DriftSeverity
    metric_value: float
    threshold: float
    timestamp: datetime
    action_required: str


class DriftMonitor:
    """
    Monitors ML models for various types of drift:
    - Data drift: Input distribution changes
    - Concept drift: Relationship between features and target changes
    - Prediction drift: Output distribution changes
    """
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.baseline_stats = {}
        self.baseline_predictions = []
        self.baseline_performance = {}
        self.monitoring_window = []
        self.alerts = []
        
        # Drift detection thresholds
        self.THRESHOLDS = {
            "ks_test_pvalue": 0.01,      # Kolmogorov-Smirnov test
            "psi_threshold": 0.2,         # Population Stability Index
            "accuracy_drop": 0.05,        # 5% accuracy drop
            "prediction_shift": 0.1,      # 10% shift in predictions
            "feature_importance_change": 0.3  # 30% change in feature importance
        }
        
    async def initialize_baseline(self, 
                                 features: Dict[str, np.ndarray],
                                 predictions: np.ndarray,
                                 actuals: Optional[np.ndarray] = None):
        """
        Initialize baseline statistics for drift detection
        """
        logger.info(f"Initializing baseline for model {self.model_id}")
        
        # Store baseline feature statistics
        for feature_name, values in features.items():
            self.baseline_stats[feature_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "quantiles": [float(q) for q in np.percentile(values, [25, 50, 75])],
                "distribution": values.tolist()[:1000]  # Store sample for KS test
            }
        
        # Store baseline predictions
        self.baseline_predictions = predictions.tolist()[:1000]
        
        # Calculate baseline performance if actuals provided
        if actuals is not None:
            self.baseline_performance = self._calculate_performance(predictions, actuals)
        
        logger.info(f"Baseline initialized with {len(features)} features")
    
    async def detect_drift(self, 
                          features: Dict[str, np.ndarray],
                          predictions: np.ndarray,
                          actuals: Optional[np.ndarray] = None) -> List[DriftAlert]:
        """
        Detect various types of drift in model inputs and outputs
        """
        alerts = []
        
        # Check data drift for each feature
        for feature_name, values in features.items():
            if feature_name in self.baseline_stats:
                drift_result = self._detect_feature_drift(feature_name, values)
                if drift_result:
                    alerts.append(drift_result)
        
        # Check prediction drift
        prediction_drift = self._detect_prediction_drift(predictions)
        if prediction_drift:
            alerts.append(prediction_drift)
        
        # Check performance drift if we have actuals
        if actuals is not None and self.baseline_performance:
            perf_drift = self._detect_performance_drift(predictions, actuals)
            if perf_drift:
                alerts.append(perf_drift)
        
        # Store alerts
        self.alerts.extend(alerts)
        
        # Take action on critical drift
        for alert in alerts:
            if alert.severity == DriftSeverity.CRITICAL:
                await self._handle_critical_drift(alert)
        
        return alerts
    
    def _detect_feature_drift(self, feature_name: str, current_values: np.ndarray) -> Optional[DriftAlert]:
        """
        Detect drift in a single feature using KS test and PSI
        """
        baseline = self.baseline_stats[feature_name]
        baseline_dist = np.array(baseline["distribution"])
        
        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(baseline_dist, current_values[:1000])
        
        # Check for significant drift
        if p_value < self.THRESHOLDS["ks_test_pvalue"]:
            # Calculate PSI for severity assessment
            psi = self._calculate_psi(baseline_dist, current_values)
            severity = self._assess_drift_severity(psi)
            
            return DriftAlert(
                model_id=self.model_id,
                feature_name=feature_name,
                drift_type="data_drift",
                severity=severity,
                metric_value=p_value,
                threshold=self.THRESHOLDS["ks_test_pvalue"],
                timestamp=datetime.now(),
                action_required=self._get_action_for_severity(severity)
            )
        
        return None
    
    def _detect_prediction_drift(self, current_predictions: np.ndarray) -> Optional[DriftAlert]:
        """
        Detect drift in model predictions
        """
        if not self.baseline_predictions:
            return None
        
        baseline_preds = np.array(self.baseline_predictions)
        current_preds = current_predictions[:1000]
        
        # Compare prediction distributions
        ks_stat, p_value = stats.ks_2samp(baseline_preds, current_preds)
        
        if p_value < self.THRESHOLDS["ks_test_pvalue"]:
            # Calculate shift in mean predictions
            baseline_mean = np.mean(baseline_preds)
            current_mean = np.mean(current_preds)
            shift = abs(current_mean - baseline_mean) / (baseline_mean + 1e-10)
            
            severity = DriftSeverity.HIGH if shift > 0.2 else DriftSeverity.MODERATE
            
            return DriftAlert(
                model_id=self.model_id,
                feature_name="predictions",
                drift_type="prediction_drift",
                severity=severity,
                metric_value=shift,
                threshold=self.THRESHOLDS["prediction_shift"],
                timestamp=datetime.now(),
                action_required=self._get_action_for_severity(severity)
            )
        
        return None
    
    def _detect_performance_drift(self, predictions: np.ndarray, actuals: np.ndarray) -> Optional[DriftAlert]:
        """
        Detect drift in model performance metrics
        """
        current_performance = self._calculate_performance(predictions, actuals)
        
        # Compare with baseline
        baseline_acc = self.baseline_performance.get("accuracy", 1.0)
        current_acc = current_performance.get("accuracy", 0.0)
        
        accuracy_drop = baseline_acc - current_acc
        
        if accuracy_drop > self.THRESHOLDS["accuracy_drop"]:
            severity = DriftSeverity.CRITICAL if accuracy_drop > 0.1 else DriftSeverity.HIGH
            
            return DriftAlert(
                model_id=self.model_id,
                feature_name="model_accuracy",
                drift_type="performance_drift",
                severity=severity,
                metric_value=accuracy_drop,
                threshold=self.THRESHOLDS["accuracy_drop"],
                timestamp=datetime.now(),
                action_required="Model retraining required" if severity == DriftSeverity.CRITICAL else "Monitor closely"
            )
        
        return None
    
    def _calculate_psi(self, baseline: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)
        """
        # Create buckets based on baseline
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        bins = np.linspace(min_val, max_val, buckets + 1)
        
        # Calculate distributions
        baseline_hist, _ = np.histogram(baseline, bins=bins)
        current_hist, _ = np.histogram(current, bins=bins)
        
        # Normalize
        baseline_hist = baseline_hist / len(baseline)
        current_hist = current_hist / len(current)
        
        # Calculate PSI
        psi = 0
        for b, c in zip(baseline_hist, current_hist):
            if b > 0 and c > 0:
                psi += (c - b) * np.log(c / b)
        
        return psi
    
    def _calculate_performance(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics
        """
        # For regression
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        
        # For classification (if predictions are probabilities)
        if np.all((predictions >= 0) & (predictions <= 1)):
            binary_preds = (predictions > 0.5).astype(int)
            binary_actuals = actuals.astype(int)
            accuracy = np.mean(binary_preds == binary_actuals)
        else:
            accuracy = 1.0 - (mae / (np.max(actuals) - np.min(actuals) + 1e-10))
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "accuracy": float(accuracy)
        }
    
    def _assess_drift_severity(self, psi: float) -> DriftSeverity:
        """
        Assess drift severity based on PSI value
        """
        if psi < 0.1:
            return DriftSeverity.LOW
        elif psi < 0.2:
            return DriftSeverity.MODERATE
        elif psi < 0.3:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL
    
    def _get_action_for_severity(self, severity: DriftSeverity) -> str:
        """
        Get recommended action based on drift severity
        """
        actions = {
            DriftSeverity.NONE: "No action required",
            DriftSeverity.LOW: "Continue monitoring",
            DriftSeverity.MODERATE: "Increase monitoring frequency",
            DriftSeverity.HIGH: "Consider model retraining",
            DriftSeverity.CRITICAL: "Immediate retraining required - consider fallback model"
        }
        return actions.get(severity, "Monitor closely")
    
    async def _handle_critical_drift(self, alert: DriftAlert):
        """
        Handle critical drift detection
        """
        logger.error(f"CRITICAL DRIFT DETECTED for model {self.model_id}: {alert.drift_type}")
        
        # Log to audit trail
        audit_entry = {
            "event": "critical_drift_detected",
            "model_id": self.model_id,
            "drift_type": alert.drift_type,
            "feature": alert.feature_name,
            "metric": alert.metric_value,
            "timestamp": alert.timestamp.isoformat()
        }
        
        # In production, this would:
        # 1. Switch to fallback model
        # 2. Trigger retraining pipeline
        # 3. Alert operations team
        # 4. Log to monitoring system
        
        logger.info(f"Drift alert logged: {json.dumps(audit_entry)}")
    
    def get_drift_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive drift report
        """
        recent_alerts = [a for a in self.alerts if a.timestamp > datetime.now() - timedelta(days=1)]
        
        severity_counts = {}
        for severity in DriftSeverity:
            severity_counts[severity.value] = sum(1 for a in recent_alerts if a.severity == severity)
        
        return {
            "model_id": self.model_id,
            "total_alerts_24h": len(recent_alerts),
            "severity_distribution": severity_counts,
            "critical_alerts": [
                {
                    "feature": a.feature_name,
                    "type": a.drift_type,
                    "metric": a.metric_value,
                    "action": a.action_required
                }
                for a in recent_alerts if a.severity == DriftSeverity.CRITICAL
            ],
            "baseline_features": list(self.baseline_stats.keys()),
            "monitoring_status": "active" if self.baseline_stats else "not_initialized"
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get drift monitor status for health checks."""
        recent_alerts = [a for a in self.alerts if a.timestamp > datetime.now() - timedelta(days=1)]
        has_critical = any(a.severity == DriftSeverity.CRITICAL for a in recent_alerts)
        
        return {
            "model_id": self.model_id,
            "initialized": bool(self.baseline_stats),
            "alerts_24h": len(recent_alerts),
            "has_critical_alerts": has_critical,
            "monitoring_active": bool(self.baseline_stats)
        }
    
    async def check_all_models(self):
        """Check all models for drift (stub for compatibility)."""
        # This would check multiple models in a real implementation
        return {
            "drift_detected": len(self.alerts) > 0,
            "affected_models": [self.model_id] if self.alerts else [],
            "severity": "critical" if any(a.severity == DriftSeverity.CRITICAL for a in self.alerts) else "normal",
            "recommendations": ["Review model performance"] if self.alerts else []
        }