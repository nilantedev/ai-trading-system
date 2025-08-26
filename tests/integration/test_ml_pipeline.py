"""
Comprehensive integration tests for ML pipeline with quality gates and governance.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import hashlib
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from services.ml.model_training_service import ModelTrainingService
from services.ml.feature_engineering_service import FeatureEngineeringService
from services.ml.model_serving_service import ModelServingService
from services.ml.drift_monitoring_service import DriftMonitor
from services.ml.performance_analytics_service import PerformanceAnalytics
from services.ml.model_governance import ModelGovernance, PolicyEnforcement
from services.ml.model_registry import ModelRegistry, ModelArtifact
from services.ml.explainability import ModelExplainer


class TestFeatureEngineering:
    """Test feature engineering pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample market data."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1h')
        data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'AAPL',
            'open': np.random.uniform(150, 160, len(dates)),
            'high': np.random.uniform(155, 165, len(dates)),
            'low': np.random.uniform(145, 155, len(dates)),
            'close': np.random.uniform(150, 160, len(dates)),
            'volume': np.random.uniform(1e6, 1e7, len(dates))
        })
        return data
    
    @pytest.fixture
    def feature_service(self):
        """Create feature engineering service."""
        return FeatureEngineeringService()
    
    def test_technical_indicators(self, feature_service, sample_data):
        """Test technical indicator calculation."""
        # Act
        features = feature_service.calculate_technical_indicators(sample_data)
        
        # Assert
        assert 'rsi' in features.columns
        assert 'macd' in features.columns
        assert 'macd_signal' in features.columns
        assert 'bb_upper' in features.columns
        assert 'bb_lower' in features.columns
        assert 'atr' in features.columns
        assert not features['rsi'].isna().all()
        assert features['rsi'].between(0, 100).all()
    
    def test_market_microstructure_features(self, feature_service, sample_data):
        """Test market microstructure feature generation."""
        # Act
        features = feature_service.calculate_microstructure_features(sample_data)
        
        # Assert
        assert 'spread' in features.columns
        assert 'mid_price' in features.columns
        assert 'log_return' in features.columns
        assert 'realized_volatility' in features.columns
        assert 'volume_imbalance' in features.columns
    
    def test_feature_validation(self, feature_service, sample_data):
        """Test feature validation and quality checks."""
        # Arrange
        features = feature_service.calculate_all_features(sample_data)
        
        # Act
        validation_result = feature_service.validate_features(features)
        
        # Assert
        assert validation_result['is_valid']
        assert validation_result['missing_ratio'] < 0.1
        assert validation_result['constant_features'] == []
        assert validation_result['highly_correlated_pairs'] == []
    
    def test_feature_normalization(self, feature_service, sample_data):
        """Test feature normalization."""
        # Arrange
        features = feature_service.calculate_all_features(sample_data)
        
        # Act
        normalized = feature_service.normalize_features(features)
        
        # Assert
        numeric_cols = normalized.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert normalized[col].mean() == pytest.approx(0, abs=0.1)
            assert normalized[col].std() == pytest.approx(1, abs=0.1)
    
    def test_feature_schema_generation(self, feature_service, sample_data):
        """Test feature schema generation for versioning."""
        # Arrange
        features = feature_service.calculate_all_features(sample_data)
        
        # Act
        schema = feature_service.generate_feature_schema(features)
        
        # Assert
        assert 'version' in schema
        assert 'features' in schema
        assert 'statistics' in schema
        assert len(schema['features']) == len(features.columns)
        for feature in schema['features']:
            assert 'name' in feature
            assert 'dtype' in feature
            assert 'nullable' in feature


class TestModelTraining:
    """Test model training pipeline."""
    
    @pytest.fixture
    def training_data(self):
        """Generate training data."""
        n_samples = 1000
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        return pd.DataFrame(X), pd.Series(y)
    
    @pytest.fixture
    def training_service(self):
        """Create model training service."""
        return ModelTrainingService()
    
    def test_model_training_with_validation(self, training_service, training_data):
        """Test model training with validation split."""
        X, y = training_data
        
        # Act
        model, metrics = training_service.train_model(
            X, y,
            model_type='random_forest',
            validation_split=0.2
        )
        
        # Assert
        assert model is not None
        assert 'train_rmse' in metrics
        assert 'val_rmse' in metrics
        assert 'train_mae' in metrics
        assert 'val_mae' in metrics
        assert metrics['val_rmse'] > 0
    
    def test_hyperparameter_optimization(self, training_service, training_data):
        """Test hyperparameter optimization."""
        X, y = training_data
        
        # Act
        best_params, best_score = training_service.optimize_hyperparameters(
            X, y,
            model_type='xgboost',
            n_trials=10
        )
        
        # Assert
        assert best_params is not None
        assert best_score < float('inf')
        assert 'learning_rate' in best_params
        assert 'max_depth' in best_params
    
    def test_ensemble_model_training(self, training_service, training_data):
        """Test ensemble model training."""
        X, y = training_data
        
        # Act
        ensemble, metrics = training_service.train_ensemble(
            X, y,
            base_models=['random_forest', 'xgboost', 'lightgbm'],
            meta_model='linear_regression'
        )
        
        # Assert
        assert ensemble is not None
        assert len(ensemble.base_models) == 3
        assert ensemble.meta_model is not None
        assert metrics['ensemble_rmse'] > 0
    
    def test_model_artifact_generation(self, training_service, training_data):
        """Test model artifact generation with metadata."""
        X, y = training_data
        
        # Act
        model, _ = training_service.train_model(X, y)
        artifact = training_service.create_model_artifact(
            model=model,
            feature_names=X.columns.tolist(),
            model_type='random_forest',
            metrics={'rmse': 0.5}
        )
        
        # Assert
        assert artifact['model_hash'] is not None
        assert artifact['feature_schema'] is not None
        assert artifact['training_timestamp'] is not None
        assert artifact['model_version'] is not None
        assert artifact['metrics'] == {'rmse': 0.5}


class TestModelGovernance:
    """Test ML governance and policy enforcement."""
    
    @pytest.fixture
    def governance(self):
        """Create governance service."""
        return ModelGovernance()
    
    @pytest.fixture
    def policy_enforcer(self):
        """Create policy enforcement service."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            policy = {
                'promotion_rules': {
                    'min_sharpe_ratio': 1.5,
                    'max_drawdown': 0.15,
                    'min_samples': 1000,
                    'required_metrics': ['sharpe', 'drawdown', 'accuracy']
                },
                'data_quality': {
                    'max_missing_ratio': 0.05,
                    'min_feature_importance': 0.01
                },
                'model_constraints': {
                    'max_prediction_time_ms': 100,
                    'max_model_size_mb': 500
                }
            }
            yaml.dump(policy, f)
            return PolicyEnforcement(f.name)
    
    def test_model_promotion_approval(self, policy_enforcer):
        """Test model promotion with policy compliance."""
        # Arrange
        metrics = {
            'sharpe': 1.8,
            'drawdown': 0.12,
            'accuracy': 0.65,
            'samples': 2000
        }
        
        # Act
        result = policy_enforcer.evaluate_promotion(metrics)
        
        # Assert
        assert result['approved'] is True
        assert result['violations'] == []
    
    def test_model_promotion_rejection(self, policy_enforcer):
        """Test model promotion rejection for policy violation."""
        # Arrange
        metrics = {
            'sharpe': 1.2,  # Below threshold
            'drawdown': 0.18,  # Above threshold
            'accuracy': 0.65,
            'samples': 2000
        }
        
        # Act
        result = policy_enforcer.evaluate_promotion(metrics)
        
        # Assert
        assert result['approved'] is False
        assert len(result['violations']) >= 2
        assert any('sharpe' in v for v in result['violations'])
        assert any('drawdown' in v for v in result['violations'])
    
    def test_data_quality_enforcement(self, policy_enforcer):
        """Test data quality policy enforcement."""
        # Arrange
        data_stats = {
            'missing_ratio': 0.03,
            'feature_importance': {
                'feature1': 0.3,
                'feature2': 0.05,
                'feature3': 0.005  # Below threshold
            }
        }
        
        # Act
        result = policy_enforcer.evaluate_data_quality(data_stats)
        
        # Assert
        assert result['passed'] is False
        assert 'feature3' in str(result['issues'])
    
    def test_model_lineage_tracking(self, governance):
        """Test model lineage and provenance tracking."""
        # Arrange
        model_info = {
            'model_id': 'model_123',
            'parent_model': 'model_122',
            'training_data': 'dataset_v1.5',
            'feature_version': 'features_v2.0',
            'code_version': 'git_hash_abc123'
        }
        
        # Act
        lineage = governance.track_lineage(model_info)
        
        # Assert
        assert lineage['model_id'] == 'model_123'
        assert lineage['parent_model'] == 'model_122'
        assert lineage['timestamp'] is not None
        assert lineage['hash'] is not None


class TestDriftMonitoring:
    """Test drift detection and monitoring."""
    
    @pytest.fixture
    def drift_monitor(self):
        """Create drift monitoring service."""
        return DriftMonitor()
    
    def test_data_drift_detection(self, drift_monitor):
        """Test data drift detection."""
        # Arrange
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000)
        })
        
        # No drift
        current_data_no_drift = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100)
        })
        
        # With drift
        current_data_drift = pd.DataFrame({
            'feature1': np.random.normal(2, 1, 100),  # Mean shifted
            'feature2': np.random.normal(5, 4, 100)   # Variance increased
        })
        
        # Act
        no_drift_result = drift_monitor.detect_data_drift(
            reference_data, current_data_no_drift
        )
        drift_result = drift_monitor.detect_data_drift(
            reference_data, current_data_drift
        )
        
        # Assert
        assert no_drift_result['drift_detected'] is False
        assert drift_result['drift_detected'] is True
        assert 'feature1' in drift_result['drifted_features']
    
    def test_prediction_drift_detection(self, drift_monitor):
        """Test prediction drift detection."""
        # Arrange
        reference_predictions = np.random.normal(100, 10, 1000)
        current_predictions_no_drift = np.random.normal(100, 10, 100)
        current_predictions_drift = np.random.normal(110, 15, 100)
        
        # Act
        no_drift = drift_monitor.detect_prediction_drift(
            reference_predictions, current_predictions_no_drift
        )
        drift = drift_monitor.detect_prediction_drift(
            reference_predictions, current_predictions_drift
        )
        
        # Assert
        assert no_drift['drift_detected'] is False
        assert drift['drift_detected'] is True
        assert drift['ks_statistic'] > no_drift['ks_statistic']
    
    def test_performance_degradation_detection(self, drift_monitor):
        """Test model performance degradation detection."""
        # Arrange
        historical_metrics = [
            {'rmse': 0.5, 'mae': 0.3, 'timestamp': datetime.now() - timedelta(days=i)}
            for i in range(30, 0, -1)
        ]
        
        # Current performance similar
        current_good = {'rmse': 0.52, 'mae': 0.31}
        
        # Current performance degraded
        current_bad = {'rmse': 0.75, 'mae': 0.50}
        
        # Act
        good_result = drift_monitor.detect_performance_degradation(
            historical_metrics, current_good
        )
        bad_result = drift_monitor.detect_performance_degradation(
            historical_metrics, current_bad
        )
        
        # Assert
        assert good_result['degradation_detected'] is False
        assert bad_result['degradation_detected'] is True
        assert bad_result['severity'] in ['high', 'critical']


class TestModelExplainability:
    """Test model explainability features."""
    
    @pytest.fixture
    def explainer(self):
        """Create model explainer."""
        return ModelExplainer()
    
    @pytest.fixture
    def trained_model(self):
        """Create a trained model for testing."""
        from sklearn.ensemble import RandomForestRegressor
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model = RandomForestRegressor(n_estimators=10)
        model.fit(X, y)
        return model, X, y
    
    def test_feature_importance(self, explainer, trained_model):
        """Test feature importance calculation."""
        model, X, y = trained_model
        
        # Act
        importance = explainer.calculate_feature_importance(
            model, X, 
            feature_names=[f'feature_{i}' for i in range(X.shape[1])]
        )
        
        # Assert
        assert len(importance) == X.shape[1]
        assert sum(importance.values()) == pytest.approx(1.0, rel=0.01)
        assert all(v >= 0 for v in importance.values())
    
    def test_shap_values(self, explainer, trained_model):
        """Test SHAP value calculation."""
        model, X, y = trained_model
        
        # Act
        shap_values = explainer.calculate_shap_values(model, X[:10])
        
        # Assert
        assert shap_values.shape == (10, X.shape[1])
        # SHAP values should sum to prediction difference from expected value
        base_value = y.mean()
        predictions = model.predict(X[:10])
        for i in range(10):
            assert np.abs(shap_values[i].sum() + base_value - predictions[i]) < 0.1
    
    def test_partial_dependence(self, explainer, trained_model):
        """Test partial dependence plot data."""
        model, X, y = trained_model
        
        # Act
        pd_result = explainer.calculate_partial_dependence(
            model, X, 
            feature_idx=0,
            grid_resolution=10
        )
        
        # Assert
        assert len(pd_result['grid']) == 10
        assert len(pd_result['values']) == 10
        assert pd_result['feature_name'] == 'feature_0'


class TestModelRegistry:
    """Test model registry and artifact management."""
    
    @pytest.fixture
    def registry(self):
        """Create model registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            return ModelRegistry(storage_path=tmpdir)
    
    def test_register_model(self, registry):
        """Test model registration."""
        # Arrange
        model_artifact = ModelArtifact(
            model_id='model_001',
            model_type='random_forest',
            version='1.0.0',
            metrics={'rmse': 0.5, 'mae': 0.3},
            feature_schema={'features': ['f1', 'f2']},
            model_hash='abc123'
        )
        
        # Act
        result = registry.register_model(model_artifact)
        
        # Assert
        assert result['success'] is True
        assert result['model_id'] == 'model_001'
        assert registry.get_model('model_001') is not None
    
    def test_model_versioning(self, registry):
        """Test model versioning."""
        # Register multiple versions
        for i in range(3):
            artifact = ModelArtifact(
                model_id='model_001',
                model_type='random_forest',
                version=f'1.0.{i}',
                metrics={'rmse': 0.5 - i*0.05},
                model_hash=f'hash_{i}'
            )
            registry.register_model(artifact)
        
        # Act
        versions = registry.get_model_versions('model_001')
        latest = registry.get_latest_model('model_001')
        
        # Assert
        assert len(versions) == 3
        assert latest.version == '1.0.2'
    
    def test_model_promotion_stages(self, registry):
        """Test model promotion through stages."""
        # Arrange
        artifact = ModelArtifact(
            model_id='model_001',
            version='1.0.0',
            stage='development'
        )
        registry.register_model(artifact)
        
        # Act
        registry.promote_model('model_001', '1.0.0', 'staging')
        staging_model = registry.get_model_by_stage('model_001', 'staging')
        
        registry.promote_model('model_001', '1.0.0', 'production')
        prod_model = registry.get_model_by_stage('model_001', 'production')
        
        # Assert
        assert staging_model is not None
        assert prod_model is not None
        assert prod_model.stage == 'production'


class TestEndToEndMLPipeline:
    """Test complete ML pipeline flow."""
    
    @pytest.mark.asyncio
    async def test_complete_ml_pipeline(self):
        """Test complete pipeline from data to deployment."""
        # 1. Generate synthetic data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1h')
        data = pd.DataFrame({
            'timestamp': dates,
            'price': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'volume': np.random.uniform(1e6, 1e7, len(dates))
        })
        
        # 2. Feature engineering
        feature_service = FeatureEngineeringService()
        features = feature_service.calculate_all_features(data)
        
        # 3. Data validation
        validation = feature_service.validate_features(features)
        assert validation['is_valid']
        
        # 4. Model training
        training_service = ModelTrainingService()
        X = features.iloc[:-100]
        y = data['price'].shift(-1).iloc[:-100].dropna()
        
        model, metrics = training_service.train_model(X[:len(y)], y)
        
        # 5. Model validation against governance policies
        policy_enforcer = PolicyEnforcement()
        governance_result = policy_enforcer.evaluate_model(model, metrics)
        
        # 6. Register model if approved
        if governance_result.get('approved', True):
            registry = ModelRegistry()
            artifact = ModelArtifact(
                model=model,
                metrics=metrics,
                feature_schema=feature_service.generate_feature_schema(features)
            )
            registry.register_model(artifact)
        
        # 7. Deploy model
        serving_service = ModelServingService()
        deployment_result = await serving_service.deploy_model(
            model,
            model_id=artifact.model_id,
            stage='staging'
        )
        
        # 8. Monitor drift
        drift_monitor = DriftMonitor()
        drift_monitor.start_monitoring(
            model_id=artifact.model_id,
            reference_data=X
        )
        
        # Assert final state
        assert deployment_result['success']
        assert artifact.model_id is not None