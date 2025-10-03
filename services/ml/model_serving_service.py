#!/usr/bin/env python3
"""
Model Serving Service - Production model serving with versioning, shadow mode, and monitoring
Provides real-time inference for trained ML models with comprehensive monitoring and drift detection.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib
import uuid

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))

from trading_common import get_settings, get_logger, MarketData
from trading_common.cache import get_trading_cache
from trading_common.database import get_database
from trading_common.feature_store import get_feature_store, FeatureVector
from trading_common.ml_pipeline import get_ml_pipeline, BaseMLModel
from trading_common.resilience import with_circuit_breaker, with_retry

logger = get_logger(__name__)
settings = get_settings()


class ModelStatus(str, Enum):
    """Model serving status."""
    ACTIVE = "active"           # Actively serving predictions
    SHADOW = "shadow"          # Running in shadow mode (logging but not serving)
    CANDIDATE = "candidate"    # Candidate for deployment
    RETIRED = "retired"        # No longer serving


class DriftType(str, Enum):
    """Types of model drift."""
    FEATURE_DRIFT = "feature_drift"         # Input feature distribution changed
    PREDICTION_DRIFT = "prediction_drift"   # Prediction distribution changed
    PERFORMANCE_DRIFT = "performance_drift" # Model performance degraded


@dataclass
class ModelVersion:
    """Model version metadata."""
    model_name: str
    version: str
    status: ModelStatus
    model_path: str
    created_at: datetime
    deployed_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    
    # Performance tracking
    total_predictions: int = 0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    
    # Drift monitoring
    drift_score: float = 0.0
    last_drift_check: Optional[datetime] = None
    
    def get_model_id(self) -> str:
        """Get unique model identifier."""
        return f"{self.model_name}:{self.version}"


@dataclass
class PredictionRequest:
    """Request for model prediction."""
    model_name: str
    features: Dict[str, Any]
    entity_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
    
    def __post_init__(self):
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())


@dataclass
class PredictionResponse:
    """Response from model prediction."""
    request_id: str
    model_name: str
    model_version: str
    prediction: Any
    confidence: Optional[float] = None
    
    # Metadata
    prediction_time: datetime = field(default_factory=datetime.utcnow)
    inference_latency_ms: float = 0.0
    feature_count: int = 0
    
    # Shadow mode tracking
    is_shadow: bool = False
    shadow_predictions: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DriftReport:
    """Report on detected model drift."""
    model_name: str
    drift_type: DriftType
    drift_score: float
    threshold: float
    detected_at: datetime
    
    # Drift details
    affected_features: List[str] = field(default_factory=list)
    statistical_tests: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def is_significant(self) -> bool:
        """Check if drift is significant."""
        return self.drift_score > self.threshold


class ModelServingService:
    """Production model serving with monitoring and drift detection."""
    
    def __init__(self):
        self.models: Dict[str, BaseMLModel] = {}
        self.model_versions: Dict[str, ModelVersion] = {}
        self.prediction_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.feature_store = None
        self.ml_pipeline = None
        self.cache = None
        self.db = None
        
        # Monitoring
        self.performance_metrics = defaultdict(lambda: {
            "prediction_count": 0,
            "total_latency": 0.0,
            "error_count": 0,
            "last_prediction": None
        })
        
        # Drift detection parameters
        self.drift_check_interval = timedelta(hours=1)
        self.drift_thresholds = {
            DriftType.FEATURE_DRIFT: 0.1,
            DriftType.PREDICTION_DRIFT: 0.15,
            DriftType.PERFORMANCE_DRIFT: 0.2
        }
        
    async def start(self):
        """Initialize the model serving service."""
        logger.info("Starting Model Serving Service")
        
        # Initialize dependencies
        self.feature_store = await get_feature_store()
        self.ml_pipeline = await get_ml_pipeline()
        self.cache = await get_trading_cache()
        self.db = await get_database()
        
        # Create serving tables
        await self._create_serving_tables()
        
        # Load active models
        await self._load_active_models()
        
        # Start background monitoring
        asyncio.create_task(self._drift_monitoring_loop())
        
        logger.info("Model Serving Service started")
    
    async def _create_serving_tables(self):
        """Create model serving database tables."""
        # Model versions table
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            model_id VARCHAR(255) PRIMARY KEY,
            model_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            status VARCHAR(20) NOT NULL,
            model_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            deployed_at TIMESTAMP,
            retired_at TIMESTAMP,
            metadata JSON,
            performance_stats JSON
        )
        """)
        
        # Predictions log table
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS prediction_log (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            request_id VARCHAR(255) NOT NULL,
            model_name VARCHAR(255) NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            entity_id VARCHAR(100) NOT NULL,
            features JSON NOT NULL,
            prediction JSON NOT NULL,
            confidence FLOAT,
            inference_latency_ms FLOAT,
            prediction_time TIMESTAMP NOT NULL,
            is_shadow BOOLEAN DEFAULT FALSE
        )
        """)
        
        # Drift reports table
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS drift_reports (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            model_name VARCHAR(255) NOT NULL,
            drift_type VARCHAR(50) NOT NULL,
            drift_score FLOAT NOT NULL,
            threshold_value FLOAT NOT NULL,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            details JSON,
            recommendations JSON
        )
        """)
        
        # Create indexes
        await self.db.execute("""
        CREATE INDEX IF NOT EXISTS idx_prediction_log_model_time
        ON prediction_log(model_name, prediction_time DESC)
        """)
        
        await self.db.execute("""
        CREATE INDEX IF NOT EXISTS idx_drift_reports_model_time
        ON drift_reports(model_name, detected_at DESC)
        """)
    
    async def _load_active_models(self):
        """Load active models from registry."""
        query = """
        SELECT model_name, version, file_path FROM model_registry
        WHERE is_active = TRUE
        ORDER BY created_at DESC
        """
        
        rows = await self.db.fetch_all(query)
        
        for row in rows:
            try:
                model = await self.ml_pipeline.get_model(row['model_name'], row['version'])
                if model:
                    model_version = ModelVersion(
                        model_name=row['model_name'],
                        version=row['version'],
                        status=ModelStatus.ACTIVE,
                        model_path=row['file_path'],
                        created_at=datetime.utcnow(),
                        deployed_at=datetime.utcnow()
                    )
                    
                    self.models[row['model_name']] = model
                    self.model_versions[model_version.get_model_id()] = model_version
                    
                    logger.info(f"Loaded model: {row['model_name']}:{row['version']}")
            except Exception as e:
                logger.error(f"Failed to load model {row['model_name']}: {e}")
    
    async def deploy_model(self, model_name: str, version: str, 
                          deploy_mode: ModelStatus = ModelStatus.SHADOW) -> bool:
        """Deploy a model version."""
        try:
            # Load model from ML pipeline
            model = await self.ml_pipeline.get_model(model_name, version)
            if not model:
                logger.error(f"Model not found: {model_name}:{version}")
                return False
            
            # Create model version record
            model_version = ModelVersion(
                model_name=model_name,
                version=version,
                status=deploy_mode,
                model_path=f"data/models/{model_name}_{version}.pkl",
                created_at=datetime.utcnow(),
                deployed_at=datetime.utcnow() if deploy_mode == ModelStatus.ACTIVE else None
            )
            
            # Update database
            await self.db.execute("""
            INSERT INTO model_versions 
            (model_id, model_name, version, status, model_path, deployed_at, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_id) DO UPDATE SET
                status = EXCLUDED.status,
                deployed_at = EXCLUDED.deployed_at
            """, [
                model_version.get_model_id(),
                model_name,
                version,
                deploy_mode.value,
                model_version.model_path,
                model_version.deployed_at,
                json.dumps({})
            ])
            
            # If deploying as active, retire current active version
            if deploy_mode == ModelStatus.ACTIVE:
                await self._retire_active_model(model_name)
            
            # Store in memory
            self.models[model_name] = model
            self.model_versions[model_version.get_model_id()] = model_version
            
            logger.info(f"Model deployed: {model_name}:{version} ({deploy_mode.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy model {model_name}:{version}: {e}")
            return False
    
    async def _retire_active_model(self, model_name: str):
        """Retire currently active model version."""
        for model_id, version_info in self.model_versions.items():
            if (version_info.model_name == model_name and 
                version_info.status == ModelStatus.ACTIVE):
                version_info.status = ModelStatus.RETIRED
                version_info.retired_at = datetime.utcnow()
                
                await self.db.execute("""
                UPDATE model_versions SET status = %s, retired_at = %s
                WHERE model_id = %s
                """, [ModelStatus.RETIRED.value, version_info.retired_at, model_id])
    
    @with_circuit_breaker("model_prediction")
    @with_retry(max_attempts=2)
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make prediction using active model."""
        start_time = time.time()
        
        if request.model_name not in self.models:
            raise ValueError(f"Model not found: {request.model_name}")
        
        model = self.models[request.model_name]
        
        # Get model version info
        model_version = None
        for version_info in self.model_versions.values():
            if (version_info.model_name == request.model_name and 
                version_info.status == ModelStatus.ACTIVE):
                model_version = version_info
                break
        
        if not model_version:
            raise ValueError(f"No active version for model: {request.model_name}")
        
        try:
            # Prepare features for prediction
            feature_df = pd.DataFrame([request.features])
            
            # Make prediction
            prediction = await model.predict(feature_df)
            prediction_value = prediction[0] if isinstance(prediction, np.ndarray) else prediction
            
            # Calculate confidence (if available)
            confidence = None
            if hasattr(model.model, 'predict_proba'):
                try:
                    probas = model.model.predict_proba(await model.preprocess_features(feature_df, fit=False))
                    confidence = float(np.max(probas[0]))
                except:
                    pass
            
            # Calculate inference latency
            inference_latency = (time.time() - start_time) * 1000
            
            # Create response
            response = PredictionResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                model_version=model_version.version,
                prediction=float(prediction_value) if isinstance(prediction_value, np.number) else prediction_value,
                confidence=confidence,
                inference_latency_ms=inference_latency,
                feature_count=len(request.features)
            )
            
            # Update performance metrics
            await self._update_performance_metrics(request.model_name, inference_latency, success=True)
            
            # Store prediction for monitoring
            await self._log_prediction(request, response)
            
            # Add to history for drift monitoring
            self.prediction_history[request.model_name].append({
                'timestamp': response.prediction_time,
                'features': request.features,
                'prediction': response.prediction,
                'entity_id': request.entity_id
            })
            
            return response
            
        except Exception as e:
            await self._update_performance_metrics(request.model_name, 
                                                 (time.time() - start_time) * 1000, 
                                                 success=False)
            logger.error(f"Prediction failed for {request.model_name}: {e}")
            raise
    
    async def predict_with_shadow(self, request: PredictionRequest) -> PredictionResponse:
        """Make prediction with shadow model comparison."""
        # Get active model prediction
        primary_response = await self.predict(request)
        
        # Get shadow model predictions
        shadow_predictions = {}
        for model_id, version_info in self.model_versions.items():
            if (version_info.model_name == request.model_name and 
                version_info.status == ModelStatus.SHADOW):
                
                try:
                    shadow_model = self.models.get(version_info.model_name)
                    if shadow_model:
                        feature_df = pd.DataFrame([request.features])
                        shadow_pred = await shadow_model.predict(feature_df)
                        shadow_predictions[version_info.version] = shadow_pred[0] if isinstance(shadow_pred, np.ndarray) else shadow_pred
                except Exception as e:
                    logger.warning(f"Shadow prediction failed for {model_id}: {e}")
        
        # Add shadow predictions to response
        if shadow_predictions:
            primary_response.shadow_predictions = shadow_predictions
        
        return primary_response
    
    async def _update_performance_metrics(self, model_name: str, latency_ms: float, success: bool):
        """Update model performance metrics."""
        metrics = self.performance_metrics[model_name]
        metrics["prediction_count"] += 1
        metrics["total_latency"] += latency_ms
        metrics["last_prediction"] = datetime.utcnow()
        
        if not success:
            metrics["error_count"] += 1
        
        # Update model version performance
        for version_info in self.model_versions.values():
            if (version_info.model_name == model_name and 
                version_info.status == ModelStatus.ACTIVE):
                version_info.total_predictions += 1
                version_info.avg_latency_ms = metrics["total_latency"] / metrics["prediction_count"]
                version_info.error_rate = metrics["error_count"] / metrics["prediction_count"]
                break
    
    async def _log_prediction(self, request: PredictionRequest, response: PredictionResponse):
        """Log prediction for audit and monitoring."""
        await self.db.execute("""
        INSERT INTO prediction_log 
        (request_id, model_name, model_version, entity_id, features, prediction, 
         confidence, inference_latency_ms, prediction_time, is_shadow)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, [
            response.request_id,
            response.model_name,
            response.model_version,
            request.entity_id,
            json.dumps(request.features),
            json.dumps(response.prediction),
            response.confidence,
            response.inference_latency_ms,
            response.prediction_time,
            response.is_shadow
        ])
    
    async def _drift_monitoring_loop(self):
        """Background task to monitor for model drift."""
        while True:
            try:
                await asyncio.sleep(self.drift_check_interval.total_seconds())
                await self._check_model_drift()
            except Exception as e:
                logger.error(f"Drift monitoring error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _check_model_drift(self):
        """Check for model drift across all active models."""
        for model_name in self.models.keys():
            try:
                await self._check_model_drift_single(model_name)
            except Exception as e:
                logger.error(f"Drift check failed for {model_name}: {e}")
    
    async def _check_model_drift_single(self, model_name: str):
        """Check drift for a single model."""
        if model_name not in self.prediction_history:
            return
        
        history = list(self.prediction_history[model_name])
        if len(history) < 100:  # Need minimum samples
            return
        
        # Feature drift detection
        await self._detect_feature_drift(model_name, history)
        
        # Prediction drift detection  
        await self._detect_prediction_drift(model_name, history)
    
    async def _detect_feature_drift(self, model_name: str, history: List[Dict]):
        """Detect feature distribution drift."""
        if len(history) < 200:
            return
        
        # Split history into reference (older) and current (recent) 
        split_idx = len(history) // 2
        reference_data = history[:split_idx]
        current_data = history[split_idx:]
        
        # Extract features
        ref_features = pd.DataFrame([h['features'] for h in reference_data])
        cur_features = pd.DataFrame([h['features'] for h in current_data])
        
        # Calculate drift score using KL divergence (simplified)
        drift_scores = []
        affected_features = []
        
        for feature in ref_features.columns:
            if feature in cur_features.columns:
                # Simple statistical test - would use more sophisticated methods in production
                ref_mean = ref_features[feature].mean()
                cur_mean = cur_features[feature].mean()
                ref_std = ref_features[feature].std()
                
                if ref_std > 0:
                    drift_score = abs(cur_mean - ref_mean) / ref_std
                    drift_scores.append(drift_score)
                    
                    if drift_score > self.drift_thresholds[DriftType.FEATURE_DRIFT]:
                        affected_features.append(feature)
        
        if drift_scores:
            max_drift = max(drift_scores)
            
            if max_drift > self.drift_thresholds[DriftType.FEATURE_DRIFT]:
                # Create drift report
                drift_report = DriftReport(
                    model_name=model_name,
                    drift_type=DriftType.FEATURE_DRIFT,
                    drift_score=max_drift,
                    threshold=self.drift_thresholds[DriftType.FEATURE_DRIFT],
                    detected_at=datetime.utcnow(),
                    affected_features=affected_features,
                    recommendations=[
                        "Consider retraining the model with recent data",
                        "Review feature engineering pipeline",
                        "Check data quality upstream"
                    ]
                )
                
                await self._handle_drift_detection(drift_report)
    
    async def _detect_prediction_drift(self, model_name: str, history: List[Dict]):
        """Detect prediction distribution drift."""
        if len(history) < 200:
            return
        
        # Split history
        split_idx = len(history) // 2
        reference_preds = [h['prediction'] for h in history[:split_idx]]
        current_preds = [h['prediction'] for h in history[split_idx:]]
        
        # Calculate drift in prediction distribution
        ref_mean = np.mean(reference_preds)
        cur_mean = np.mean(current_preds)
        ref_std = np.std(reference_preds)
        
        if ref_std > 0:
            drift_score = abs(cur_mean - ref_mean) / ref_std
            
            if drift_score > self.drift_thresholds[DriftType.PREDICTION_DRIFT]:
                drift_report = DriftReport(
                    model_name=model_name,
                    drift_type=DriftType.PREDICTION_DRIFT,
                    drift_score=drift_score,
                    threshold=self.drift_thresholds[DriftType.PREDICTION_DRIFT],
                    detected_at=datetime.utcnow(),
                    recommendations=[
                        "Model predictions have shifted significantly",
                        "Evaluate model performance on recent data",
                        "Consider model retraining or replacement"
                    ]
                )
                
                await self._handle_drift_detection(drift_report)
    
    async def _handle_drift_detection(self, drift_report: DriftReport):
        """Handle detected drift."""
        logger.warning(f"Drift detected for {drift_report.model_name}: {drift_report.drift_type.value} "
                      f"(score: {drift_report.drift_score:.3f}, threshold: {drift_report.threshold:.3f})")
        
        # Store drift report
        await self.db.execute("""
        INSERT INTO drift_reports 
        (model_name, drift_type, drift_score, threshold_value, details, recommendations)
        VALUES (%s, %s, %s, %s, %s, %s)
        """, [
            drift_report.model_name,
            drift_report.drift_type.value,
            drift_report.drift_score,
            drift_report.threshold,
            json.dumps({
                "affected_features": drift_report.affected_features,
                "statistical_tests": drift_report.statistical_tests
            }),
            json.dumps(drift_report.recommendations)
        ])
        
        # Update model version drift score
        for version_info in self.model_versions.values():
            if (version_info.model_name == drift_report.model_name and 
                version_info.status == ModelStatus.ACTIVE):
                version_info.drift_score = max(version_info.drift_score, drift_report.drift_score)
                version_info.last_drift_check = drift_report.detected_at
                break
        
        # TODO: Trigger alerts/notifications for significant drift
    
    async def get_model_health(self, model_name: str) -> Dict[str, Any]:
        """Get comprehensive model health status."""
        if model_name not in self.models:
            return {"error": "Model not found"}
        
        # Get model version info
        model_version = None
        for version_info in self.model_versions.values():
            if (version_info.model_name == model_name and 
                version_info.status == ModelStatus.ACTIVE):
                model_version = version_info
                break
        
        if not model_version:
            return {"error": "No active version found"}
        
        # Get performance metrics
        metrics = self.performance_metrics[model_name]
        
        # Get recent drift reports
        recent_drift_query = """
        SELECT drift_type, drift_score, detected_at 
        FROM drift_reports 
        WHERE model_name = %s AND detected_at > %s
        ORDER BY detected_at DESC LIMIT 10
        """
        
        drift_reports = await self.db.fetch_all(
            recent_drift_query, 
            [model_name, datetime.utcnow() - timedelta(days=7)]
        )
        
        return {
            "model_name": model_name,
            "version": model_version.version,
            "status": model_version.status.value,
            "deployed_at": model_version.deployed_at.isoformat() if model_version.deployed_at else None,
            "performance": {
                "total_predictions": model_version.total_predictions,
                "avg_latency_ms": model_version.avg_latency_ms,
                "error_rate": model_version.error_rate,
                "last_prediction": metrics["last_prediction"].isoformat() if metrics["last_prediction"] else None
            },
            "drift_monitoring": {
                "current_drift_score": model_version.drift_score,
                "last_drift_check": model_version.last_drift_check.isoformat() if model_version.last_drift_check else None,
                "recent_drift_reports": [
                    {
                        "type": report['drift_type'],
                        "score": report['drift_score'],
                        "detected_at": report['detected_at'].isoformat()
                    }
                    for report in drift_reports
                ]
            }
        }
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get overall service health."""
        total_models = len(self.models)
        active_models = sum(1 for v in self.model_versions.values() if v.status == ModelStatus.ACTIVE)
        
        # Calculate aggregate metrics
        total_predictions = sum(v.total_predictions for v in self.model_versions.values())
        avg_latency = np.mean([v.avg_latency_ms for v in self.model_versions.values() if v.avg_latency_ms > 0])
        avg_error_rate = np.mean([v.error_rate for v in self.model_versions.values() if v.total_predictions > 0])
        
        return {
            "service": "model_serving",
            "status": "healthy" if active_models > 0 else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "models": {
                "total": total_models,
                "active": active_models,
                "shadow": sum(1 for v in self.model_versions.values() if v.status == ModelStatus.SHADOW)
            },
            "performance": {
                "total_predictions": total_predictions,
                "avg_latency_ms": float(avg_latency) if not np.isnan(avg_latency) else 0.0,
                "avg_error_rate": float(avg_error_rate) if not np.isnan(avg_error_rate) else 0.0
            }
        }


# Global model serving service instance
_model_serving_service: Optional[ModelServingService] = None


async def get_model_serving_service() -> ModelServingService:
    """Get global model serving service instance."""
    global _model_serving_service
    if _model_serving_service is None:
        _model_serving_service = ModelServingService()
        await _model_serving_service.start()
    return _model_serving_service