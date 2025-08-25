#!/usr/bin/env python3
"""
ML Training and Evaluation Pipeline
Production-grade ML pipeline with backtesting, walk-forward validation, and risk-adjusted metrics.
"""

import asyncio
import logging
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import uuid
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

from .feature_store import get_feature_store, FeatureVector
from .cache import get_trading_cache
from .database import get_database
from .models import MarketData

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Types of ML models."""
    CLASSIFIER = "classifier"    # Predict signal direction
    REGRESSOR = "regressor"      # Predict price/returns
    RANKER = "ranker"           # Rank assets
    ANOMALY_DETECTOR = "anomaly_detector"  # Detect market anomalies


class ValidationStrategy(str, Enum):
    """Model validation strategies."""
    TIME_SERIES_SPLIT = "time_series_split"
    WALK_FORWARD = "walk_forward" 
    PURGED_CV = "purged_cv"  # Purged cross-validation for overlapping samples


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_name: str
    model_type: ModelType
    feature_names: List[str]
    target_variable: str
    
    # Training parameters
    train_start: datetime
    train_end: datetime
    validation_strategy: ValidationStrategy = ValidationStrategy.TIME_SERIES_SPLIT
    test_size: float = 0.2
    n_splits: int = 5
    
    # Feature engineering
    scaling_method: str = "standard"  # "standard", "robust", "minmax", "none"
    feature_selection: bool = True
    max_features: Optional[int] = None
    
    # Model parameters
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Risk parameters
    lookback_window: int = 252  # Days for risk metrics
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.02  # Annual risk-free rate
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"


@dataclass
class ModelMetrics:
    """Comprehensive model evaluation metrics."""
    # Basic ML metrics
    train_score: float
    validation_score: float
    test_score: float
    
    # Financial metrics
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    calmar_ratio: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    
    # Model-specific metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    
    # Risk metrics
    var_95: Optional[float] = None  # Value at Risk 95%
    expected_shortfall: Optional[float] = None
    beta: Optional[float] = None
    
    # Validation results
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BacktestResult:
    """Results from backtesting a trading strategy."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Risk metrics
    var_95: float
    expected_shortfall: float
    beta: float
    
    # Daily returns and equity curve
    daily_returns: pd.Series
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    
    # Trade log
    trades: List[Dict[str, Any]] = field(default_factory=list)
    
    def summary_stats(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "Total Return": self.total_return,
            "Annualized Return": self.annualized_return,
            "Volatility": self.volatility,
            "Sharpe Ratio": self.sharpe_ratio,
            "Sortino Ratio": self.sortino_ratio,
            "Max Drawdown": self.max_drawdown,
            "Calmar Ratio": self.calmar_ratio,
            "Win Rate": self.win_rate,
            "Profit Factor": self.profit_factor,
            "Total Trades": self.total_trades
        }


class BaseMLModel(ABC):
    """Base class for ML models in the trading system."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.is_trained = False
        self.training_metrics: Optional[ModelMetrics] = None
        
    @abstractmethod
    async def create_model(self) -> Any:
        """Create the underlying ML model."""
        pass
    
    @abstractmethod
    async def train(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Train the model."""
        pass
    
    @abstractmethod
    async def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    async def preprocess_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Preprocess features (scaling, selection)."""
        X_processed = X.copy()
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.median())
        
        # Feature scaling
        if self.config.scaling_method != "none":
            if fit or self.scaler is None:
                if self.config.scaling_method == "standard":
                    self.scaler = StandardScaler()
                elif self.config.scaling_method == "robust":
                    self.scaler = RobustScaler()
                else:
                    logger.warning(f"Unknown scaling method: {self.config.scaling_method}")
                    return X_processed
                
                X_processed = pd.DataFrame(
                    self.scaler.fit_transform(X_processed),
                    columns=X_processed.columns,
                    index=X_processed.index
                )
            else:
                X_processed = pd.DataFrame(
                    self.scaler.transform(X_processed),
                    columns=X_processed.columns,
                    index=X_processed.index
                )
        
        # Feature selection (implement if needed)
        if self.config.feature_selection and self.config.max_features:
            if fit and len(X_processed.columns) > self.config.max_features:
                # Simple variance-based feature selection
                variances = X_processed.var()
                selected_features = variances.nlargest(self.config.max_features).index
                X_processed = X_processed[selected_features]
                self.selected_features = selected_features
            elif hasattr(self, 'selected_features'):
                X_processed = X_processed[self.selected_features]
        
        return X_processed
    
    async def save_model(self, filepath: str):
        """Save trained model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'training_metrics': self.training_metrics,
            'selected_features': getattr(self, 'selected_features', None),
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    async def load_model(self, filepath: str):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.training_metrics = model_data['training_metrics']
        self.is_trained = model_data['is_trained']
        
        if 'selected_features' in model_data:
            self.selected_features = model_data['selected_features']
        
        logger.info(f"Model loaded from {filepath}")


class RandomForestModel(BaseMLModel):
    """Random Forest model for trading predictions."""
    
    async def create_model(self):
        """Create Random Forest model."""
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Merge with user-provided parameters
        params = {**default_params, **self.config.model_params}
        
        if self.config.model_type == ModelType.REGRESSOR:
            self.model = RandomForestRegressor(**params)
        elif self.config.model_type == ModelType.CLASSIFIER:
            self.model = RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unsupported model type for RandomForest: {self.config.model_type}")
        
        return self.model
    
    async def train(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Train Random Forest model."""
        if self.model is None:
            await self.create_model()
        
        # Preprocess features
        X_processed = await self.preprocess_features(X, fit=True)
        
        # Time series validation
        if self.config.validation_strategy == ValidationStrategy.TIME_SERIES_SPLIT:
            tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
            cv_scores = cross_val_score(self.model, X_processed, y, cv=tscv, 
                                      scoring='neg_mean_squared_error' if self.config.model_type == ModelType.REGRESSOR else 'accuracy')
        else:
            cv_scores = []
        
        # Train on full dataset
        self.model.fit(X_processed, y)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred_train = self.model.predict(X_processed)
        
        if self.config.model_type == ModelType.REGRESSOR:
            train_score = -mean_squared_error(y, y_pred_train)
            mse = mean_squared_error(y, y_pred_train)
            mae = mean_absolute_error(y, y_pred_train)
        else:
            train_score = accuracy_score(y, y_pred_train)
            mse = None
            mae = None
        
        # Create metrics object
        metrics = ModelMetrics(
            train_score=train_score,
            validation_score=float(np.mean(cv_scores)) if cv_scores.size > 0 else train_score,
            test_score=0.0,  # Will be calculated during backtesting
            cv_scores=cv_scores.tolist() if cv_scores.size > 0 else [],
            cv_mean=float(np.mean(cv_scores)) if cv_scores.size > 0 else None,
            cv_std=float(np.std(cv_scores)) if cv_scores.size > 0 else None,
            mse=mse,
            mae=mae
        )
        
        self.training_metrics = metrics
        logger.info(f"Model training completed. CV Score: {metrics.cv_mean:.4f} (+/- {metrics.cv_std:.4f})")
        
        return metrics
    
    async def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_processed = await self.preprocess_features(X, fit=False)
        return self.model.predict(X_processed)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importances."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        feature_names = self.selected_features if hasattr(self, 'selected_features') else self.config.feature_names
        return pd.Series(self.model.feature_importances_, index=feature_names).sort_values(ascending=False)


class MLPipeline:
    """Complete ML pipeline for training and evaluation."""
    
    def __init__(self):
        self.feature_store = None
        self.db = None
        self.models: Dict[str, BaseMLModel] = {}
        
    async def initialize(self):
        """Initialize pipeline components."""
        self.feature_store = await get_feature_store()
        self.db = await get_database()
        
        # Create model registry table
        await self._create_model_registry()
        
        logger.info("ML Pipeline initialized")
    
    async def _create_model_registry(self):
        """Create model registry table."""
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            model_name VARCHAR(255) NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            version VARCHAR(50) NOT NULL,
            config JSON NOT NULL,
            training_metrics JSON,
            file_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT FALSE,
            UNIQUE(model_name, version)
        )
        """)
    
    async def prepare_training_data(self, config: TrainingConfig) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from feature store."""
        logger.info(f"Preparing training data for {config.model_name}")
        
        # Get symbols (for now, use a predefined list)
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        
        # Get feature matrix
        feature_matrix = await self.feature_store.get_feature_matrix(
            symbols, config.feature_names, config.train_start, config.train_end
        )
        
        if feature_matrix.empty:
            raise ValueError("No training data available")
        
        # Create target variable
        target = await self._create_target_variable(feature_matrix, config.target_variable)
        
        # Remove rows with missing target
        valid_idx = ~target.isna()
        X = feature_matrix[valid_idx]
        y = target[valid_idx]
        
        logger.info(f"Training data prepared: {len(X)} samples, {len(X.columns)} features")
        return X, y
    
    async def _create_target_variable(self, df: pd.DataFrame, target_variable: str) -> pd.Series:
        """Create target variable from feature data."""
        if target_variable == "next_return":
            # Predict next period return
            returns = df.groupby('entity_id')['close'].pct_change().shift(-1)
            return returns
        elif target_variable == "signal_direction":
            # Predict signal direction (1 for up, 0 for down)
            returns = df.groupby('entity_id')['close'].pct_change().shift(-1)
            return (returns > 0).astype(int)
        else:
            # Assume target variable is already in the dataframe
            return df.get(target_variable, pd.Series(dtype=float))
    
    async def train_model(self, config: TrainingConfig, model_class=RandomForestModel) -> Tuple[BaseMLModel, ModelMetrics]:
        """Train a model with given configuration."""
        logger.info(f"Training model: {config.model_name}")
        
        # Prepare training data
        X, y = await self.prepare_training_data(config)
        
        # Create and train model
        model = model_class(config)
        metrics = await model.train(X, y)
        
        # Save model
        model_dir = Path("data/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{config.model_name}_{config.version}.pkl"
        await model.save_model(str(model_path))
        
        # Register model in database
        await self._register_model(config, metrics, str(model_path))
        
        # Store in memory
        self.models[config.model_name] = model
        
        logger.info(f"Model training completed: {config.model_name}")
        return model, metrics
    
    async def _register_model(self, config: TrainingConfig, metrics: ModelMetrics, file_path: str):
        """Register model in database."""
        await self.db.execute("""
        INSERT INTO model_registry 
        (model_name, model_type, version, config, training_metrics, file_path, is_active)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (model_name, version) DO UPDATE SET
            training_metrics = EXCLUDED.training_metrics,
            file_path = EXCLUDED.file_path,
            is_active = EXCLUDED.is_active
        """, [
            config.model_name,
            config.model_type.value,
            config.version,
            json.dumps(asdict(config), default=str),
            json.dumps(metrics.to_dict()),
            file_path,
            True
        ])
    
    async def backtest_model(self, model: BaseMLModel, 
                           start_date: datetime, 
                           end_date: datetime,
                           initial_capital: float = 100000.0,
                           transaction_cost: float = 0.001) -> BacktestResult:
        """Backtest a trained model."""
        logger.info(f"Backtesting model: {model.config.model_name}")
        
        # Get test data
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        test_data = await self.feature_store.get_feature_matrix(
            symbols, model.config.feature_names, start_date, end_date
        )
        
        if test_data.empty:
            raise ValueError("No test data available for backtesting")
        
        # Generate predictions
        predictions = await model.predict(test_data[model.config.feature_names])
        test_data['predictions'] = predictions
        
        # Simulate trading
        equity_curve = []
        trades = []
        current_capital = initial_capital
        positions = {}  # symbol -> position size
        
        # Simple strategy: long when prediction > threshold
        threshold = 0.0 if model.config.model_type == ModelType.REGRESSOR else 0.5
        
        for idx, row in test_data.iterrows():
            symbol = row['entity_id']
            price = row.get('close', row.get('price', 0))
            prediction = row['predictions']
            
            if price <= 0:
                continue
            
            # Trading logic
            if prediction > threshold and symbol not in positions:
                # Buy signal
                position_size = current_capital * 0.2  # 20% position size
                shares = position_size / price
                cost = shares * price * (1 + transaction_cost)
                
                if cost <= current_capital:
                    positions[symbol] = shares
                    current_capital -= cost
                    trades.append({
                        'timestamp': idx,
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares,
                        'cost': cost
                    })
            
            elif prediction <= threshold and symbol in positions:
                # Sell signal
                shares = positions[symbol]
                proceeds = shares * price * (1 - transaction_cost)
                current_capital += proceeds
                del positions[symbol]
                trades.append({
                    'timestamp': idx,
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': price,
                    'shares': shares,
                    'proceeds': proceeds
                })
            
            # Calculate current portfolio value
            portfolio_value = current_capital
            for pos_symbol, shares in positions.items():
                if pos_symbol == symbol:
                    portfolio_value += shares * price
            
            equity_curve.append({
                'timestamp': idx,
                'portfolio_value': portfolio_value,
                'cash': current_capital,
                'positions_value': portfolio_value - current_capital
            })
        
        # Convert to pandas for analysis
        equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
        
        if equity_df.empty:
            raise ValueError("No trades executed during backtest period")
        
        # Calculate performance metrics
        returns = equity_df['portfolio_value'].pct_change().dropna()
        total_return = (equity_df['portfolio_value'].iloc[-1] / initial_capital) - 1
        
        # Risk-adjusted metrics
        annual_factor = 252  # Trading days
        annualized_return = (1 + total_return) ** (annual_factor / len(returns)) - 1
        volatility = returns.std() * np.sqrt(annual_factor)
        sharpe_ratio = (annualized_return - model.config.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Downside metrics
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * np.sqrt(annual_factor) if len(negative_returns) > 0 else volatility
        sortino_ratio = (annualized_return - model.config.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Drawdown analysis
        running_max = equity_df['portfolio_value'].expanding().max()
        drawdown = (equity_df['portfolio_value'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        winning_trades = sum(1 for t in trades if t.get('proceeds', 0) > t.get('cost', float('inf')))
        losing_trades = len(trades) - winning_trades
        win_rate = winning_trades / len(trades) if trades else 0
        
        # Create backtest result
        result = BacktestResult(
            strategy_name=model.config.model_name,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=0.0,  # Would calculate from individual trades
            avg_loss=0.0,  # Would calculate from individual trades
            profit_factor=1.0,  # Would calculate from win/loss ratios
            var_95=returns.quantile(0.05),
            expected_shortfall=returns[returns <= returns.quantile(0.05)].mean(),
            beta=0.0,  # Would calculate vs benchmark
            daily_returns=returns,
            equity_curve=equity_df['portfolio_value'],
            drawdown_curve=drawdown,
            trades=trades
        )
        
        logger.info(f"Backtest completed. Sharpe Ratio: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2%}")
        return result
    
    async def get_model(self, model_name: str, version: Optional[str] = None) -> Optional[BaseMLModel]:
        """Load model from registry."""
        if model_name in self.models:
            return self.models[model_name]
        
        # Load from database
        query = """
        SELECT file_path FROM model_registry 
        WHERE model_name = %s AND is_active = TRUE
        """
        params = [model_name]
        
        if version:
            query += " AND version = %s"
            params.append(version)
        
        query += " ORDER BY created_at DESC LIMIT 1"
        
        row = await self.db.fetch_one(query, params)
        if not row:
            return None
        
        # Load model from file
        model = RandomForestModel(TrainingConfig(model_name=model_name, model_type=ModelType.REGRESSOR, feature_names=[], target_variable=""))
        await model.load_model(row['file_path'])
        
        self.models[model_name] = model
        return model


# Global ML pipeline instance
_ml_pipeline: Optional[MLPipeline] = None


async def get_ml_pipeline() -> MLPipeline:
    """Get global ML pipeline instance."""
    global _ml_pipeline
    if _ml_pipeline is None:
        _ml_pipeline = MLPipeline()
        await _ml_pipeline.initialize()
    return _ml_pipeline