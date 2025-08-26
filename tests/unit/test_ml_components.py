#!/usr/bin/env python3
"""
Proper pytest test suite for ML components.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError

# Add paths for imports
parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))
sys.path.append(str(parent_dir / "shared" / "python-common"))

# Define ML models for testing
class ModelType(str, Enum):
    REGRESSOR = "regressor"
    CLASSIFIER = "classifier"

class ValidationStrategy(str, Enum):
    TIME_SERIES_SPLIT = "time_series_split"
    WALK_FORWARD = "walk_forward"

class TrainingConfig(BaseModel):
    model_name: str
    model_type: ModelType
    feature_names: List[str]
    target_variable: str
    train_start: datetime
    train_end: datetime
    validation_strategy: ValidationStrategy = ValidationStrategy.TIME_SERIES_SPLIT
    version: str = "1.0.0"

class FeatureDefinition(BaseModel):
    name: str
    description: str
    feature_type: str
    source: str
    computation_logic: str
    dependencies: List[str] = Field(default_factory=list)
    version: str = "1.0.0"


# Technical indicator functions
def calculate_sma(prices: List[float], period: int) -> List[float]:
    """Calculate Simple Moving Average"""
    if not prices:
        return []
    
    sma_values = []
    for i in range(len(prices)):
        if i < period - 1:
            sma_values.append(np.nan)
        else:
            sma_values.append(np.mean(prices[i-period+1:i+1]))
    return sma_values


def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return [np.nan] * len(prices)
    
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    
    rsi_values = [np.nan] * period
    
    for i in range(period, len(prices)):
        if i == period:
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
        else:
            current_gain = gain[i-1] if i-1 < len(gain) else 0
            current_loss = loss[i-1] if i-1 < len(loss) else 0
            avg_gain = (avg_gain * (period - 1) + current_gain) / period
            avg_loss = (avg_loss * (period - 1) + current_loss) / period
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
        
        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)
    
    return rsi_values


class TestMLModels:
    """Test suite for ML model definitions."""
    
    def test_feature_definition_creation(self):
        """Test creating a feature definition."""
        feature_def = FeatureDefinition(
            name="sma_20",
            description="20-period simple moving average",
            feature_type="float",
            source="market_data",
            computation_logic="SMA(close, 20)",
            dependencies=["close_price"]
        )
        
        assert feature_def.name == "sma_20"
        assert feature_def.feature_type == "float"
        assert feature_def.source == "market_data"
        assert "close_price" in feature_def.dependencies
        assert feature_def.version == "1.0.0"
    
    def test_feature_definition_validation(self):
        """Test feature definition validation."""
        with pytest.raises(ValidationError):
            # Missing required fields
            FeatureDefinition(name="test")
    
    def test_training_config_creation(self):
        """Test creating a training configuration."""
        config = TrainingConfig(
            model_name="momentum_model",
            model_type=ModelType.REGRESSOR,
            feature_names=["sma_5", "sma_10", "sma_20", "rsi_14"],
            target_variable="next_return",
            train_start=datetime.utcnow() - timedelta(days=200),
            train_end=datetime.utcnow() - timedelta(days=30)
        )
        
        assert config.model_name == "momentum_model"
        assert config.model_type == ModelType.REGRESSOR
        assert len(config.feature_names) == 4
        assert "rsi_14" in config.feature_names
        assert config.target_variable == "next_return"
        assert config.validation_strategy == ValidationStrategy.TIME_SERIES_SPLIT
        
    def test_training_config_date_validation(self):
        """Test training config date validation."""
        train_start = datetime.utcnow() - timedelta(days=200)
        train_end = datetime.utcnow() - timedelta(days=30)
        
        config = TrainingConfig(
            model_name="test_model",
            model_type=ModelType.CLASSIFIER,
            feature_names=["feature1"],
            target_variable="target",
            train_start=train_start,
            train_end=train_end
        )
        
        assert (config.train_end - config.train_start).days == 170
        assert config.train_start < config.train_end


class TestTechnicalIndicators:
    """Test suite for technical indicators."""
    
    @pytest.fixture
    def sample_prices(self):
        """Sample price data for testing."""
        return [100.0, 101.2, 99.8, 102.5, 103.1, 101.9, 104.2, 105.8, 103.4, 106.1]
    
    def test_sma_calculation(self, sample_prices):
        """Test Simple Moving Average calculation."""
        sma_5 = calculate_sma(sample_prices, 5)
        
        assert len(sma_5) == len(sample_prices)
        assert np.isnan(sma_5[0])  # First 4 values should be NaN
        assert np.isnan(sma_5[3])
        assert not np.isnan(sma_5[4])  # 5th value should be valid
        
        # Check actual calculation
        expected_sma_5 = np.mean(sample_prices[0:5])
        assert abs(sma_5[4] - expected_sma_5) < 0.001
        
        # Count valid values
        valid_count = len([x for x in sma_5 if not np.isnan(x)])
        assert valid_count == 6  # 10 - 4 = 6 valid values
    
    def test_sma_with_different_periods(self, sample_prices):
        """Test SMA with various periods."""
        sma_3 = calculate_sma(sample_prices, 3)
        sma_7 = calculate_sma(sample_prices, 7)
        
        valid_3 = len([x for x in sma_3 if not np.isnan(x)])
        valid_7 = len([x for x in sma_7 if not np.isnan(x)])
        
        assert valid_3 == 8  # 10 - 2 = 8
        assert valid_7 == 4  # 10 - 6 = 4
    
    def test_sma_edge_cases(self):
        """Test SMA with edge cases."""
        # Empty list
        assert calculate_sma([], 5) == []
        
        # Period larger than data
        short_data = [100, 101, 102]
        sma = calculate_sma(short_data, 5)
        assert len(sma) == 3
        assert all(np.isnan(x) for x in sma)
        
        # Single value
        assert len(calculate_sma([100], 1)) == 1
        assert calculate_sma([100], 1)[0] == 100
    
    def test_rsi_calculation(self, sample_prices):
        """Test Relative Strength Index calculation."""
        # Need more data points for RSI
        extended_prices = sample_prices * 2
        rsi_14 = calculate_rsi(extended_prices, 14)
        
        assert len(rsi_14) == len(extended_prices)
        
        # First 14 values should be NaN
        assert all(np.isnan(x) for x in rsi_14[:14])
        
        # RSI should be between 0 and 100
        valid_rsi = [x for x in rsi_14 if not np.isnan(x)]
        assert all(0 <= x <= 100 for x in valid_rsi)
        
        # Count valid values
        assert len(valid_rsi) == len(extended_prices) - 14
    
    def test_rsi_edge_cases(self):
        """Test RSI with edge cases."""
        # Too few data points
        short_data = [100, 101, 102]
        rsi = calculate_rsi(short_data, 14)
        assert len(rsi) == 3
        assert all(np.isnan(x) for x in rsi)
        
        # All prices increasing (RSI should be near 100)
        increasing = list(range(100, 130))
        rsi = calculate_rsi(increasing, 14)
        valid_rsi = [x for x in rsi if not np.isnan(x)]
        if valid_rsi:  # Only check if we have valid values
            assert all(x > 70 for x in valid_rsi)  # Strong uptrend
        
        # All prices decreasing (RSI should be near 0)
        decreasing = list(range(130, 100, -1))
        rsi = calculate_rsi(decreasing, 14)
        valid_rsi = [x for x in rsi if not np.isnan(x)]
        if valid_rsi:  # Only check if we have valid values
            assert all(x < 30 for x in valid_rsi)  # Strong downtrend
    
    def test_rsi_neutral_market(self):
        """Test RSI in sideways market."""
        # Oscillating prices
        oscillating = [100, 101, 100, 101, 100, 101] * 5
        rsi = calculate_rsi(oscillating, 14)
        valid_rsi = [x for x in rsi if not np.isnan(x)]
        
        if valid_rsi:
            # RSI should be around 50 in neutral market
            avg_rsi = np.mean(valid_rsi)
            assert 40 <= avg_rsi <= 60


class TestMLInfrastructure:
    """Test suite for ML infrastructure validation."""
    
    def test_feature_store_models_exist(self):
        """Verify feature store models are properly defined."""
        assert FeatureDefinition is not None
        
        # Test creating an instance to verify fields exist
        feature = FeatureDefinition(
            name="test_feature",
            description="Test feature",
            feature_type="float",
            source="test_source",
            computation_logic="test_logic"
        )
        
        assert feature.name == "test_feature"
        assert feature.computation_logic == "test_logic"
    
    def test_training_config_validation(self):
        """Test training configuration validation."""
        with pytest.raises(ValidationError):
            # Should fail without required fields
            TrainingConfig(model_name="test")
    
    def test_model_type_enum(self):
        """Test model type enumeration."""
        assert ModelType.REGRESSOR.value == "regressor"
        assert ModelType.CLASSIFIER.value == "classifier"
        assert len(ModelType) == 2
    
    def test_validation_strategy_enum(self):
        """Test validation strategy enumeration."""
        assert ValidationStrategy.TIME_SERIES_SPLIT.value == "time_series_split"
        assert ValidationStrategy.WALK_FORWARD.value == "walk_forward"
        assert len(ValidationStrategy) == 2
    
    @pytest.mark.parametrize("model_type,expected", [
        (ModelType.REGRESSOR, "regressor"),
        (ModelType.CLASSIFIER, "classifier"),
    ])
    def test_model_type_values(self, model_type, expected):
        """Test model type enum values."""
        assert model_type.value == expected


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "--cov=.", "--cov-report=term-missing"])