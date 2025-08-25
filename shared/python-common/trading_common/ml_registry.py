#!/usr/bin/env python3
"""
ML Model Registry for AI Trading System.
Provides model versioning, metadata management, and deployment tracking.
"""

import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import os
import asyncio
from pathlib import Path
import logging

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    import mlflow.tensorflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from .logging import get_logger
from .cache import get_trading_cache

logger = get_logger(__name__)


class ModelStatus(str, Enum):
    """Model deployment status."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"


class ModelType(str, Enum):
    """Types of trading models."""
    PRICE_PREDICTION = "price_prediction"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    ANOMALY_DETECTION = "anomaly_detection"
    REGIME_DETECTION = "regime_detection"


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    model_id: str
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    author: str
    description: str
    tags: Dict[str, str]
    
    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_score: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    
    # Model configuration
    hyperparameters: Dict[str, Any] = None
    features: List[str] = None
    training_data_hash: Optional[str] = None
    training_duration_seconds: Optional[float] = None
    
    # Deployment info
    deployment_config: Dict[str, Any] = None
    resource_requirements: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.hyperparameters is None:
            self.hyperparameters = {}
        if self.features is None:
            self.features = []
        if self.deployment_config is None:
            self.deployment_config = {}
        if self.resource_requirements is None:
            self.resource_requirements = {}


@dataclass
class ModelPrediction:
    """Record of a model prediction."""
    prediction_id: str
    model_id: str
    model_version: str
    input_data: Dict[str, Any]
    prediction: Union[float, List[float], Dict[str, Any]]
    confidence: Optional[float]
    timestamp: datetime
    execution_time_ms: float
    symbol: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}


class ModelRegistry:
    """Registry for managing ML models and their lifecycle."""
    
    def __init__(self, registry_path: str = "models/registry"):
        """Initialize model registry."""
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(exist_ok=True)
        
        self.metadata_path = self.registry_path / "metadata"
        self.metadata_path.mkdir(exist_ok=True)
        
        self.predictions_path = self.registry_path / "predictions"
        self.predictions_path.mkdir(exist_ok=True)
        
        # In-memory cache
        self._metadata_cache: Dict[str, ModelMetadata] = {}
        self._loaded_models: Dict[str, Any] = {}
        
        # Initialize MLflow if available
        if MLFLOW_AVAILABLE:
            self._init_mlflow()
    
    def _init_mlflow(self):
        """Initialize MLflow tracking."""
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///models/mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        experiment_name = "trading_models"
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Failed to set MLflow experiment: {e}")
    
    def register_model(
        self,
        model: Any,
        metadata: ModelMetadata,
        overwrite: bool = False
    ) -> str:
        """Register a new model version."""
        model_key = f"{metadata.name}_v{metadata.version}"
        model_file_path = self.models_path / f"{model_key}.pkl"
        metadata_file_path = self.metadata_path / f"{model_key}.json"
        
        # Check if model already exists
        if model_file_path.exists() and not overwrite:
            raise ValueError(f"Model {model_key} already exists. Use overwrite=True to replace.")
        
        try:
            # Save model
            if JOBLIB_AVAILABLE:
                joblib.dump(model, model_file_path)
            else:
                with open(model_file_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Generate model hash for integrity checking
            model_hash = self._calculate_model_hash(model_file_path)
            metadata.tags = metadata.tags or {}
            metadata.tags['model_hash'] = model_hash
            
            # Save metadata
            with open(metadata_file_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)
            
            # Cache metadata
            self._metadata_cache[model_key] = metadata
            
            # Log to MLflow if available
            if MLFLOW_AVAILABLE:
                self._log_to_mlflow(model, metadata)
            
            logger.info(f"Registered model {model_key}")
            return model_key
            
        except Exception as e:
            logger.error(f"Failed to register model {model_key}: {e}")
            # Clean up partial files
            for path in [model_file_path, metadata_file_path]:
                if path.exists():
                    path.unlink()
            raise
    
    def load_model(self, model_name: str, version: Optional[str] = None) -> Tuple[Any, ModelMetadata]:
        """Load a model and its metadata."""
        if version is None:
            version = self.get_latest_version(model_name)
        
        model_key = f"{model_name}_v{version}"
        
        # Check cache first
        if model_key in self._loaded_models:
            model = self._loaded_models[model_key]
            metadata = self._metadata_cache[model_key]
            return model, metadata
        
        model_file_path = self.models_path / f"{model_key}.pkl"
        metadata_file_path = self.metadata_path / f"{model_key}.json"
        
        if not model_file_path.exists() or not metadata_file_path.exists():
            raise ValueError(f"Model {model_key} not found in registry")
        
        try:
            # Load metadata
            with open(metadata_file_path, 'r') as f:
                metadata_dict = json.load(f)
                metadata = ModelMetadata(**metadata_dict)
            
            # Verify model integrity
            current_hash = self._calculate_model_hash(model_file_path)
            stored_hash = metadata.tags.get('model_hash')
            if stored_hash and current_hash != stored_hash:
                logger.warning(f"Model {model_key} hash mismatch. Model may be corrupted.")
            
            # Load model
            if JOBLIB_AVAILABLE:
                model = joblib.load(model_file_path)
            else:
                with open(model_file_path, 'rb') as f:
                    model = pickle.load(f)
            
            # Cache the loaded model
            self._loaded_models[model_key] = model
            self._metadata_cache[model_key] = metadata
            
            logger.info(f"Loaded model {model_key}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            raise
    
    def list_models(self, model_type: Optional[ModelType] = None) -> List[ModelMetadata]:
        """List all registered models."""
        models = []
        
        for metadata_file in self.metadata_path.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    metadata = ModelMetadata(**metadata_dict)
                    
                    if model_type is None or metadata.model_type == model_type:
                        models.append(metadata)
                        
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
        
        return sorted(models, key=lambda x: x.created_at, reverse=True)
    
    def get_model_metadata(self, model_name: str, version: Optional[str] = None) -> ModelMetadata:
        """Get metadata for a specific model version."""
        if version is None:
            version = self.get_latest_version(model_name)
        
        model_key = f"{model_name}_v{version}"
        
        # Check cache first
        if model_key in self._metadata_cache:
            return self._metadata_cache[model_key]
        
        metadata_file_path = self.metadata_path / f"{model_key}.json"
        
        if not metadata_file_path.exists():
            raise ValueError(f"Model {model_key} not found in registry")
        
        with open(metadata_file_path, 'r') as f:
            metadata_dict = json.load(f)
            metadata = ModelMetadata(**metadata_dict)
            self._metadata_cache[model_key] = metadata
            return metadata
    
    def get_latest_version(self, model_name: str) -> str:
        """Get the latest version of a model."""
        versions = []
        
        for metadata_file in self.metadata_path.glob(f"{model_name}_v*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    versions.append(metadata_dict['version'])
            except Exception:
                continue
        
        if not versions:
            raise ValueError(f"No versions found for model {model_name}")
        
        # Sort versions (assuming semantic versioning or numeric)
        try:
            versions.sort(key=lambda x: [int(i) for i in x.split('.')])
        except ValueError:
            versions.sort()  # Fallback to string sort
        
        return versions[-1]
    
    def update_model_status(self, model_name: str, version: str, status: ModelStatus):
        """Update the status of a model."""
        model_key = f"{model_name}_v{version}"
        metadata_file_path = self.metadata_path / f"{model_key}.json"
        
        if not metadata_file_path.exists():
            raise ValueError(f"Model {model_key} not found in registry")
        
        # Load current metadata
        with open(metadata_file_path, 'r') as f:
            metadata_dict = json.load(f)
        
        # Update status and timestamp
        metadata_dict['status'] = status.value
        metadata_dict['updated_at'] = datetime.utcnow().isoformat()
        
        # Save updated metadata
        with open(metadata_file_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
        
        # Update cache
        if model_key in self._metadata_cache:
            self._metadata_cache[model_key].status = status
            self._metadata_cache[model_key].updated_at = datetime.utcnow()
        
        logger.info(f"Updated {model_key} status to {status.value}")
    
    def record_prediction(self, prediction: ModelPrediction):
        """Record a model prediction for tracking and analysis."""
        prediction_date = prediction.timestamp.strftime("%Y-%m-%d")
        predictions_dir = self.predictions_path / prediction_date
        predictions_dir.mkdir(exist_ok=True)
        
        prediction_file = predictions_dir / f"{prediction.prediction_id}.json"
        
        with open(prediction_file, 'w') as f:
            json.dump(asdict(prediction), f, indent=2, default=str)
    
    def get_model_performance_history(
        self, 
        model_name: str, 
        version: str,
        days: int = 30
    ) -> List[ModelPrediction]:
        """Get prediction history for model performance analysis."""
        predictions = []
        model_key = f"{model_name}_v{version}"
        
        # Search through prediction files for the last N days
        for i in range(days):
            date = datetime.utcnow() - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            predictions_dir = self.predictions_path / date_str
            
            if not predictions_dir.exists():
                continue
            
            for prediction_file in predictions_dir.glob("*.json"):
                try:
                    with open(prediction_file, 'r') as f:
                        prediction_dict = json.load(f)
                        
                    if prediction_dict.get('model_id') == model_key:
                        prediction = ModelPrediction(**prediction_dict)
                        predictions.append(prediction)
                        
                except Exception as e:
                    logger.warning(f"Failed to load prediction from {prediction_file}: {e}")
        
        return sorted(predictions, key=lambda x: x.timestamp, reverse=True)
    
    def cleanup_old_predictions(self, retention_days: int = 90):
        """Clean up old prediction records."""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        for date_dir in self.predictions_path.iterdir():
            if not date_dir.is_dir():
                continue
            
            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                if dir_date < cutoff_date:
                    import shutil
                    shutil.rmtree(date_dir)
                    logger.info(f"Cleaned up predictions for {date_dir.name}")
            except ValueError:
                continue
    
    def _calculate_model_hash(self, model_file_path: Path) -> str:
        """Calculate hash of model file for integrity checking."""
        hash_algo = hashlib.sha256()
        with open(model_file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_algo.update(chunk)
        return hash_algo.hexdigest()
    
    def _log_to_mlflow(self, model: Any, metadata: ModelMetadata):
        """Log model to MLflow tracking server."""
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            with mlflow.start_run(run_name=f"{metadata.name}_v{metadata.version}"):
                # Log parameters
                mlflow.log_params(metadata.hyperparameters)
                
                # Log metrics
                metrics = {
                    'accuracy': metadata.accuracy,
                    'precision': metadata.precision,
                    'recall': metadata.recall,
                    'f1_score': metadata.f1_score,
                    'auc_score': metadata.auc_score,
                    'sharpe_ratio': metadata.sharpe_ratio,
                    'max_drawdown': metadata.max_drawdown
                }
                metrics = {k: v for k, v in metrics.items() if v is not None}
                mlflow.log_metrics(metrics)
                
                # Log model
                model_type = type(model).__name__
                if 'sklearn' in model_type.lower() or hasattr(model, 'predict'):
                    mlflow.sklearn.log_model(model, "model")
                else:
                    mlflow.log_artifact(str(self.models_path / f"{metadata.name}_v{metadata.version}.pkl"))
                
                # Log tags
                mlflow.set_tags({
                    'model_type': metadata.model_type.value,
                    'status': metadata.status.value,
                    'author': metadata.author
                })
                
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")


# Global registry instance
_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get or create global model registry instance."""
    global _model_registry
    if _model_registry is None:
        registry_path = os.getenv("MODEL_REGISTRY_PATH", "models/registry")
        _model_registry = ModelRegistry(registry_path)
    return _model_registry


def register_model(
    model: Any,
    name: str,
    version: str,
    model_type: ModelType,
    author: str,
    description: str = "",
    **kwargs
) -> str:
    """Convenience function to register a model."""
    registry = get_model_registry()
    
    metadata = ModelMetadata(
        model_id=f"{name}_v{version}",
        name=name,
        version=version,
        model_type=model_type,
        status=ModelStatus.DEVELOPMENT,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        author=author,
        description=description,
        tags=kwargs.get('tags', {}),
        **{k: v for k, v in kwargs.items() if k in ModelMetadata.__annotations__}
    )
    
    return registry.register_model(model, metadata)


def load_production_model(model_name: str) -> Tuple[Any, ModelMetadata]:
    """Load the production version of a model."""
    registry = get_model_registry()
    
    # Find production version
    models = registry.list_models()
    production_models = [
        m for m in models 
        if m.name == model_name and m.status == ModelStatus.PRODUCTION
    ]
    
    if not production_models:
        raise ValueError(f"No production model found for {model_name}")
    
    # Get latest production version
    latest_production = max(production_models, key=lambda x: x.updated_at)
    
    return registry.load_model(model_name, latest_production.version)