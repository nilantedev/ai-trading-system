#!/usr/bin/env python3
"""
Off-Hours Model Training Service - Intensive model training during market closures
Performs comprehensive model retraining, backtesting, and optimization during weekends and off-hours.
"""

import asyncio
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, time
from dataclasses import dataclass, asdict
import pickle
import os
from pathlib import Path
import schedule
from concurrent.futures import ThreadPoolExecutor
import warnings
import random
import hashlib
from functools import lru_cache
from prometheus_client import Counter, Histogram, Gauge
import platform
import subprocess

warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))

from trading_common import MarketData, get_settings, get_logger
try:
    # Optional MinIO storage integration (for artifact offloading)
    from shared.storage.minio_storage import (
        get_minio_client,
        ensure_bucket,
        put_bytes,
        build_model_artifact_key,
        StorageError,
    )
except Exception:  # pragma: no cover - optional dependency path
    get_minio_client = None  # type: ignore
    ensure_bucket = None  # type: ignore
    put_bytes = None  # type: ignore
    build_model_artifact_key = None  # type: ignore
    StorageError = Exception  # type: ignore
from trading_common.cache import get_trading_cache

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class TrainingDataset:
    """Training dataset with features and targets."""
    symbol: str
    features: np.ndarray  # Feature matrix
    targets: np.ndarray   # Target values (price changes, returns, etc.)
    timestamps: List[datetime]
    feature_names: List[str]
    target_names: List[str]
    train_size: int
    test_size: int
    
    def split_train_test(self, test_ratio: float = 0.2) -> Tuple['TrainingDataset', 'TrainingDataset']:
        """Split dataset into training and testing sets."""
        split_idx = int(len(self.features) * (1 - test_ratio))
        
        train_set = TrainingDataset(
            symbol=self.symbol,
            features=self.features[:split_idx],
            targets=self.targets[:split_idx],
            timestamps=self.timestamps[:split_idx],
            feature_names=self.feature_names,
            target_names=self.target_names,
            train_size=split_idx,
            test_size=0
        )
        
        test_set = TrainingDataset(
            symbol=self.symbol,
            features=self.features[split_idx:],
            targets=self.targets[split_idx:],
            timestamps=self.timestamps[split_idx:],
            feature_names=self.feature_names,
            target_names=self.target_names,
            train_size=0,
            test_size=len(self.features) - split_idx
        )
        
        return train_set, test_set


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    symbol: str
    mse: float
    mae: float
    r2_score: float
    directional_accuracy: float  # % of time model predicts direction correctly
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    avg_win: float
    avg_loss: float
    training_time: float  # seconds
    hyperparameters: Dict[str, Any]
    feature_importance: Dict[str, float]
    validation_scores: List[float]
    
    def get_composite_score(self) -> float:
        """Calculate composite performance score."""
        # Weighted combination of metrics
        score = (
            0.25 * self.directional_accuracy +     # 25% - Direction accuracy
            0.20 * max(0, self.sharpe_ratio / 3) +  # 20% - Risk-adjusted returns
            0.15 * self.win_rate +                  # 15% - Win rate
            0.15 * max(0, self.r2_score) +         # 15% - Explanatory power
            0.10 * max(0, (2 - self.avg_loss/self.avg_win) / 2) +  # 10% - Risk-reward ratio
            0.10 * max(0, self.total_return / 0.5) + # 10% - Total return
            0.05 * max(0, (0.2 - self.max_drawdown) / 0.2)  # 5% - Drawdown control
        )
        return min(score, 1.0)
    
    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'symbol': self.symbol,
            'mse': self.mse,
            'mae': self.mae,
            'r2_score': self.r2_score,
            'directional_accuracy': self.directional_accuracy,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'total_return': self.total_return,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'training_time': self.training_time,
            'hyperparameters': self.hyperparameters,
            'composite_score': self.get_composite_score(),
        }


@dataclass
class TrainingJob:
    """Training job specification."""
    job_id: str
    symbols: List[str]
    model_types: List[str]
    start_time: datetime
    priority: int  # 1=high, 2=medium, 3=low
    estimated_duration: timedelta
    data_requirements: Dict[str, Any]
    hyperparameter_tuning: bool = True
    cross_validation: bool = True
    backtesting: bool = True
    
    @property
    def is_high_priority(self) -> bool:
        return self.priority == 1


class DataPreprocessor:
    """Preprocesses market data for model training."""
    
    def __init__(self):
        self.scalers = {}
        self.feature_generators = {
            'technical': self._generate_technical_features,
            'price_action': self._generate_price_action_features,
            'volume': self._generate_volume_features,
            'volatility': self._generate_volatility_features,
            'momentum': self._generate_momentum_features,
            'seasonal': self._generate_seasonal_features
        }
    
    async def prepare_training_data(self, symbol: str, market_data: List[MarketData], 
                                  days_back: int = 252) -> Optional[TrainingDataset]:
        """Prepare comprehensive training dataset."""
        
        if len(market_data) < 100:  # Need minimum data
            logger.warning(f"Insufficient data for {symbol}: {len(market_data)} records")
            return None
        
        # Convert to DataFrame for easier manipulation
        df = self._market_data_to_dataframe(market_data)
        
        # Generate all feature types
        feature_dfs = []
        feature_names = []
        
        for feature_type, generator in self.feature_generators.items():
            try:
                feature_df, names = generator(df)
                if feature_df is not None:
                    feature_dfs.append(feature_df)
                    feature_names.extend(names)
            except Exception as e:
                logger.warning(f"Failed to generate {feature_type} features for {symbol}: {e}")
        
        if not feature_dfs:
            logger.error(f"No features generated for {symbol}")
            return None
        
        # Combine all features
        features_df = pd.concat(feature_dfs, axis=1)
        
        # Generate targets (what we want to predict)
        targets_df, target_names = self._generate_targets(df)
        
        # Align data (remove NaN rows)
        combined_df = pd.concat([features_df, targets_df], axis=1).dropna()
        
        if len(combined_df) < 50:
            logger.warning(f"Too few valid samples after preprocessing for {symbol}: {len(combined_df)}")
            return None
        
        # Split features and targets
        features = combined_df[feature_names].values
        targets = combined_df[target_names].values
        timestamps = combined_df.index.tolist()
        
        # Scale features
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        features_scaled = scaler.fit_transform(features)
        self.scalers[symbol] = scaler
        
        dataset = TrainingDataset(
            symbol=symbol,
            features=features_scaled,
            targets=targets,
            timestamps=timestamps,
            feature_names=feature_names,
            target_names=target_names,
            train_size=len(features_scaled),
            test_size=0
        )
        
        logger.info(f"Prepared dataset for {symbol}: {len(features_scaled)} samples, "
                   f"{len(feature_names)} features, {len(target_names)} targets")
        
        return dataset
    
    def _market_data_to_dataframe(self, market_data: List[MarketData]) -> pd.DataFrame:
        """Convert market data to DataFrame."""
        data = []
        for md in market_data:
            data.append({
                'timestamp': md.timestamp,
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': md.volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df
    
    def _generate_technical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Generate technical analysis features."""
        features = pd.DataFrame(index=df.index)
        names = []
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            sma_name = f'sma_{period}'
            ema_name = f'ema_{period}'
            features[sma_name] = df['close'].rolling(period).mean()
            features[ema_name] = df['close'].ewm(span=period).mean()
            names.extend([sma_name, ema_name])
            
            # Price relative to moving averages
            sma_ratio_name = f'close_sma_{period}_ratio'
            ema_ratio_name = f'close_ema_{period}_ratio'
            features[sma_ratio_name] = df['close'] / features[sma_name]
            features[ema_ratio_name] = df['close'] / features[ema_name]
            names.extend([sma_ratio_name, ema_ratio_name])
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        names.append('rsi')
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        names.extend(['macd', 'macd_signal', 'macd_histogram'])
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_sma = df['close'].rolling(bb_period).mean()
        bb_std_val = df['close'].rolling(bb_period).std()
        features['bb_upper'] = bb_sma + (bb_std * bb_std_val)
        features['bb_lower'] = bb_sma - (bb_std * bb_std_val)
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        names.extend(['bb_upper', 'bb_lower', 'bb_position'])
        
        return features, names
    
    def _generate_price_action_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Generate price action features."""
        features = pd.DataFrame(index=df.index)
        names = []
        
        # Returns over different periods
        for period in [1, 2, 3, 5, 10, 20]:
            return_name = f'return_{period}d'
            features[return_name] = df['close'].pct_change(period)
            names.append(return_name)
        
        # High-Low spread
        features['hl_spread'] = (df['high'] - df['low']) / df['close']
        names.append('hl_spread')
        
        # Open-Close gap
        features['oc_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        names.append('oc_gap')
        
        # Intraday range
        features['intraday_return'] = (df['close'] - df['open']) / df['open']
        features['intraday_range'] = (df['high'] - df['low']) / df['open']
        names.extend(['intraday_return', 'intraday_range'])
        
        # Price position within daily range
        features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        names.append('price_position')
        
        return features, names
    
    def _generate_volume_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Generate volume-based features."""
        features = pd.DataFrame(index=df.index)
        names = []
        
        # Volume moving averages
        for period in [5, 10, 20, 50]:
            vol_sma_name = f'volume_sma_{period}'
            features[vol_sma_name] = df['volume'].rolling(period).mean()
            names.append(vol_sma_name)
            
            # Volume ratio
            vol_ratio_name = f'volume_ratio_{period}'
            features[vol_ratio_name] = df['volume'] / features[vol_sma_name]
            names.append(vol_ratio_name)
        
        # On-Balance Volume
        features['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        names.append('obv')
        
        # Volume-Price Trend
        features['vpt'] = ((df['close'].diff() / df['close'].shift(1)) * df['volume']).cumsum()
        names.append('vpt')
        
        # Money Flow Index (simplified)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        features['money_flow'] = money_flow.rolling(14).sum()
        names.append('money_flow')
        
        return features, names
    
    def _generate_volatility_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Generate volatility features."""
        features = pd.DataFrame(index=df.index)
        names = []
        
        # Historical volatility over different periods
        returns = df['close'].pct_change()
        for period in [5, 10, 20, 50]:
            vol_name = f'volatility_{period}d'
            features[vol_name] = returns.rolling(period).std() * np.sqrt(252)
            names.append(vol_name)
        
        # Parkinson volatility (uses high-low)
        features['parkinson_vol'] = np.sqrt((1/(4*np.log(2))) * (np.log(df['high']/df['low']))**2)
        names.append('parkinson_vol')
        
        # Garman-Klass volatility
        features['gk_vol'] = np.sqrt(0.5 * (np.log(df['high']/df['low']))**2 - 
                                   (2*np.log(2) - 1) * (np.log(df['close']/df['open']))**2)
        names.append('gk_vol')
        
        return features, names
    
    def _generate_momentum_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Generate momentum features."""
        features = pd.DataFrame(index=df.index)
        names = []
        
        # Rate of Change over different periods
        for period in [5, 10, 20]:
            roc_name = f'roc_{period}d'
            features[roc_name] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
            names.append(roc_name)
        
        # Williams %R
        for period in [14, 20]:
            wr_name = f'williams_r_{period}d'
            highest_high = df['high'].rolling(period).max()
            lowest_low = df['low'].rolling(period).min()
            features[wr_name] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
            names.append(wr_name)
        
        # Stochastic Oscillator
        period = 14
        lowest_low = df['low'].rolling(period).min()
        highest_high = df['high'].rolling(period).max()
        features['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        names.extend(['stoch_k', 'stoch_d'])
        
        return features, names
    
    def _generate_seasonal_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Generate seasonal/time-based features."""
        features = pd.DataFrame(index=df.index)
        names = []
        
        # Day of week (1-7)
        features['day_of_week'] = df.index.dayofweek + 1
        names.append('day_of_week')
        
        # Month (1-12)
        features['month'] = df.index.month
        names.append('month')
        
        # Quarter (1-4)
        features['quarter'] = df.index.quarter
        names.append('quarter')
        
        # Day of month
        features['day_of_month'] = df.index.day
        names.append('day_of_month')
        
        # Week of year
        features['week_of_year'] = df.index.isocalendar().week
        names.append('week_of_year')
        
        return features, names
    
    def _generate_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Generate prediction targets."""
        targets = pd.DataFrame(index=df.index)
        names = []
        
        # Future returns (what we want to predict)
        for horizon in [1, 2, 3, 5]:
            target_name = f'future_return_{horizon}d'
            targets[target_name] = df['close'].pct_change(horizon).shift(-horizon)
            names.append(target_name)
        
        # Direction (up/down)
        for horizon in [1, 2, 3]:
            direction_name = f'direction_{horizon}d'
            future_return = df['close'].pct_change(horizon).shift(-horizon)
            targets[direction_name] = (future_return > 0).astype(int)
            names.append(direction_name)
        
        return targets, names


class ModelTrainer:
    """Trains and evaluates ML models."""
    
    def __init__(self):
        self.models = {
            'random_forest': self._create_random_forest,
            'gradient_boosting': self._create_gradient_boosting,
            'xgboost': self._create_xgboost,
            'lightgbm': self._create_lightgbm
        }
        
        self.trained_models = {}
        self.model_performances = {}
    
    async def train_models(self, dataset: TrainingDataset, 
                          model_types: List[str] = None) -> Dict[str, ModelPerformance]:
        """Train multiple models on dataset."""
        
        if model_types is None:
            model_types = list(self.models.keys())
        
        performances = {}
        # Split data for validation (retain original for walk-forward)
        train_set, test_set = dataset.split_train_test(test_ratio=0.2)
        walk_forward_results = {}

        for model_type in model_types:
            if model_type not in self.models:
                logger.warning(f"Unknown model type: {model_type}")
                continue
            
            try:
                logger.info(f"Training {model_type} for {dataset.symbol}")
                start_time = datetime.utcnow()
                
                # Train model
                model, best_params = await self._train_single_model(
                    model_type, train_set, test_set
                )
                
                training_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Evaluate performance (standard split)
                performance = await self._evaluate_model(
                    model, model_type, test_set, training_time, best_params
                )

                # Walk-forward evaluation (lightweight) for additional robustness signals
                try:
                    wf = self._walk_forward_evaluate(model, dataset, windows=3)
                    walk_forward_results[model_type] = wf
                except Exception as wf_e:
                    logger.debug(f"Walk-forward evaluation failed for {model_type}: {wf_e}")
                
                performances[model_type] = performance
                
                # Store trained model
                model_key = f"{dataset.symbol}_{model_type}"
                self.trained_models[model_key] = model
                
                logger.info(f"Completed {model_type} for {dataset.symbol}: "
                           f"Score={performance.get_composite_score():.3f}, "
                           f"Directional Accuracy={performance.directional_accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_type} for {dataset.symbol}: {e}")
        
        # Attach walk-forward summaries to trainer state (optional future use)
        if walk_forward_results:
            self.walk_forward_last = walk_forward_results
        return performances

    def _walk_forward_evaluate(self, model: Any, dataset: TrainingDataset, windows: int = 3) -> Dict[str, Any]:
        """Simple walk-forward evaluation (non-leakage) using sequential expanding windows.
        Returns aggregated metrics vs a naive baseline (previous return)."""
        n = len(dataset.features)
        if n < 100 or windows < 2:
            return {}
        segment_size = n // windows
        metrics = []
        baseline_hits = 0
        model_hits = 0
        total = 0
        for i in range(windows - 1):
            end_train = segment_size * (i + 1)
            train_X = dataset.features[:end_train]
            train_y = dataset.targets[:end_train, 0]
            test_X = dataset.features[end_train: end_train + segment_size]
            test_y = dataset.targets[end_train: end_train + segment_size, 0]
            if len(test_X) < 5:
                continue
            try:
                clone_model = model  # For tree-based models retraining cost moderate; could deep copy if stateful
                clone_model.fit(train_X, train_y)
                preds = clone_model.predict(test_X)
            except Exception:
                continue
            # Directional accuracy for segment
            model_dir = np.sign(preds)
            actual_dir = np.sign(test_y)
            model_hits += np.sum(model_dir == actual_dir)
            # Baseline: previous actual direction
            prev_dir = np.sign(np.concatenate([[0], test_y[:-1]]))
            baseline_hits += np.sum(prev_dir == actual_dir)
            total += len(test_y)
            seg_sharpe = 0.0
            with np.errstate(divide='ignore', invalid='ignore'):
                returns = preds * np.sign(preds)
                if returns.std() > 0:
                    seg_sharpe = returns.mean() / returns.std() * np.sqrt(252)
            metrics.append({'segment': i, 'sharpe': float(seg_sharpe)})
        if total == 0:
            return {}
        return {
            'segments': metrics,
            'directional_accuracy_walk_forward': model_hits / total,
            'baseline_directional_accuracy': baseline_hits / total,
            'uplift_vs_baseline': (model_hits - baseline_hits) / max(baseline_hits, 1),
            'model_hits': int(model_hits),
            'baseline_hits': int(baseline_hits),
            'total_samples': int(total)
        }
    
    async def _train_single_model(self, model_type: str, train_set: TrainingDataset, 
                                test_set: TrainingDataset) -> Tuple[Any, Dict[str, Any]]:
        """Train a single model with hyperparameter tuning."""
        
        # Create base model
        model_creator = self.models[model_type]
        
        # Define hyperparameter search space
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        }
        
        # Use first target (1-day return) for training
        y_train = train_set.targets[:, 0]  # First target column
        X_train = train_set.features
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Grid search with cross-validation
        base_model = model_creator()
        param_grid = param_grids.get(model_type, {})
        
        if param_grid:
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # No hyperparameter tuning
            base_model.fit(X_train, y_train)
            best_model = base_model
            best_params = {}
        
        return best_model, best_params
    
    def _create_random_forest(self):
        """Create Random Forest model."""
        return RandomForestRegressor(random_state=42, n_jobs=-1)
    
    def _create_gradient_boosting(self):
        """Create Gradient Boosting model."""
        return GradientBoostingRegressor(random_state=42)
    
    def _create_xgboost(self):
        """Create XGBoost model."""
        return xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    def _create_lightgbm(self):
        """Create LightGBM model."""
        return lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
    
    async def _evaluate_model(self, model: Any, model_type: str, test_set: TrainingDataset,
                            training_time: float, hyperparameters: Dict[str, Any]) -> ModelPerformance:
        """Evaluate model performance comprehensively."""
        
        X_test = test_set.features
        y_test = test_set.targets[:, 0]  # First target (1-day return)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Basic metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Directional accuracy
        actual_direction = np.sign(y_test)
        predicted_direction = np.sign(y_pred)
        directional_accuracy = np.mean(actual_direction == predicted_direction)
        
        # Trading simulation metrics
        returns = y_pred * np.sign(y_pred)  # Simple proxy strategy: trade in predicted direction
        cumulative_returns = np.cumprod(1 + returns) - 1
        
        # Sharpe ratio
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        
        # Max drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / (running_max + 1)
        max_drawdown = np.min(drawdown)
        
        # Win rate and avg win/loss
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
        
        # Feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = dict(zip(test_set.feature_names, importances))
        
        performance = ModelPerformance(
            model_name=model_type,
            symbol=test_set.symbol,
            mse=mse,
            mae=mae,
            r2_score=r2,
            directional_accuracy=directional_accuracy,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_return=cumulative_returns[-1] if len(cumulative_returns) > 0 else 0,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            training_time=training_time,
            hyperparameters=hyperparameters,
            feature_importance=feature_importance,
            validation_scores=[]  # Would include cross-validation scores
        )
        
        return performance


class OffHoursTrainingService:
    """Main off-hours training service."""
    
    def __init__(self):
        self.cache = None
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        
        # Training configuration
        self.training_jobs = asyncio.Queue(maxsize=100)
        self.is_running = False
        self.training_enabled = False
        
        # Training schedule
        self.training_schedule = {
            'weekends': True,      # Train on weekends
            'market_holidays': True,  # Train on market holidays
            'after_hours': True,   # Train after market close (6 PM - 4 AM)
            'pre_market': False    # Don't train during pre-market
        }
        
        # Resource management
        self.max_concurrent_jobs = 2
        self.active_jobs = set()
        
        # Performance tracking
        self.jobs_completed = 0
        self.total_training_time = 0
        self.best_models = {}  # symbol -> best performing model info
        
        # Model persistence (configurable via env VAR TRAINING_MODELS_DIR)
        models_dir_env = os.getenv("TRAINING_MODELS_DIR", "/tmp/trading_models")
        self.models_dir = Path(models_dir_env)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Metrics (Prometheus) - define once
        self._metric_jobs_total = Counter(
            'training_jobs_total',
            'Total training jobs processed',
            ['status']
        )
        self._metric_job_duration = Histogram(
            'training_job_duration_seconds',
            'Duration of training jobs in seconds',
            buckets=(5, 30, 60, 120, 300, 600, 900, 1800, 3600)
        )
        self._metric_job_active = Gauge(
            'training_active_jobs',
            'Number of active training jobs'
        )
        self._metric_job_queue_depth = Gauge(
            'training_queue_depth',
            'Training job queue depth'
        )
        self._metric_models_saved_total = Counter(
            'training_models_saved_total',
            'Total best model artifacts saved',
            ['symbol', 'model_type']
        )
        self._metric_symbol_samples = Gauge(
            'training_symbol_samples',
            'Number of samples used for symbol training',
            ['symbol']
        )
        self._metric_data_fetch = Histogram(
            'training_data_fetch_seconds',
            'Time to fetch historical data per symbol',
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)
        )
        self._metric_data_points = Gauge(
            'training_data_points',
            'Raw historical data points fetched before preprocessing',
            ['symbol']
        )
        self._metric_integrity_warnings = Counter(
            'training_integrity_warnings_total',
            'Count of integrity or lineage related warnings',
            ['type']
        )
    
    async def initialize(self):
        """Initialize the training service."""
        self.cache = get_trading_cache()
        
        # Start background tasks
        self.is_running = True
        asyncio.create_task(self._training_job_processor())
        asyncio.create_task(self._schedule_training_sessions())
        
        logger.info("Off-Hours Training Service initialized")
    
    def is_training_time(self) -> bool:
        """Check if current time is appropriate for training."""
        now = datetime.now()
        current_time = now.time()
        weekday = now.weekday()
        
        # Weekend training
        if weekday >= 5 and self.training_schedule['weekends']:
            return True
        
        # After-hours training (6 PM - 4 AM on weekdays)
        if (weekday < 5 and self.training_schedule['after_hours'] and 
            (current_time >= time(18, 0) or current_time <= time(4, 0))):
            return True
        
        # Market holidays (would check holiday calendar)
        if self.training_schedule['market_holidays']:
            # Simplified - would check actual market holiday calendar
            pass
        
        return False
    
    async def schedule_training_job(self, symbols: List[str], model_types: List[str] = None,
                                  priority: int = 2) -> str:
        """Schedule a new training job."""
        
        if model_types is None:
            model_types = ['random_forest', 'xgboost', 'lightgbm']
        
        job_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(tuple(symbols)) % 10000}"
        
        # Estimate training duration
        estimated_duration = timedelta(
            minutes=len(symbols) * len(model_types) * 10  # 10 min per symbol-model combination
        )
        
        training_job = TrainingJob(
            job_id=job_id,
            symbols=symbols,
            model_types=model_types,
            start_time=datetime.utcnow(),
            priority=priority,
            estimated_duration=estimated_duration,
            data_requirements={'days_back': 252, 'min_samples': 100},
            hyperparameter_tuning=True,
            cross_validation=True,
            backtesting=True
        )
        
        try:
            await self.training_jobs.put(training_job)
            logger.info(f"Scheduled training job {job_id} for {len(symbols)} symbols")
            return job_id
        except asyncio.QueueFull:
            logger.warning(f"Training job queue full, cannot schedule job for {symbols}")
            return ""
    
    async def _training_job_processor(self):
        """Process training jobs from the queue."""
        while self.is_running:
            try:
                # Update queue depth metric
                try:
                    self._metric_job_queue_depth.set(self.training_jobs.qsize())
                except Exception:
                    pass
                # Wait for training job
                job = await asyncio.wait_for(self.training_jobs.get(), timeout=10.0)
                
                # Check if we can start training
                if not self.training_enabled or not self.is_training_time():
                    # Re-queue job for later
                    await self.training_jobs.put(job)
                    await asyncio.sleep(60)  # Wait 1 minute before checking again
                    continue
                
                # Check resource availability
                if len(self.active_jobs) >= self.max_concurrent_jobs:
                    # Re-queue job
                    await self.training_jobs.put(job)
                    await asyncio.sleep(30)  # Wait 30 seconds
                    continue
                
                # Start training job
                self.active_jobs.add(job.job_id)
                try:
                    self._metric_job_active.set(len(self.active_jobs))
                except Exception:
                    pass
                asyncio.create_task(self._execute_training_job(job))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in training job processor: {e}")
    
    async def _execute_training_job(self, job: TrainingJob):
        """Execute a training job with deterministic seeding, metrics, and dataset manifest capture."""
        start_time = datetime.utcnow()
        seed = self._derive_seed(job.job_id)
        self._apply_seeds(seed)
        job_seed_info = {'seed': seed}

        try:
            logger.info(f"Starting training job {job.job_id} with {len(job.symbols)} symbols (seed={seed})")
            job_results: Dict[str, Any] = {}

            for symbol in job.symbols:
                symbol_start = datetime.utcnow()
                try:
                    # Historical data retrieval
                    market_data = await self._get_historical_data(symbol, job.data_requirements['days_back'])
                    if not market_data or len(market_data) < job.data_requirements['min_samples']:
                        logger.warning(f"Insufficient data for {symbol}, skipping")
                        job_results[symbol] = {'error': 'insufficient_data'}
                        continue

                    # Prepare dataset
                    dataset = await self.preprocessor.prepare_training_data(symbol, market_data)
                    if not dataset:
                        job_results[symbol] = {'error': 'dataset_preparation_failed'}
                        continue

                    # Metric: samples used
                    try:
                        self._metric_symbol_samples.labels(symbol=symbol).set(len(dataset.features))
                    except Exception:
                        pass

                    # Train models
                    model_performances = await self.trainer.train_models(dataset, job.model_types)
                    if not model_performances:
                        job_results[symbol] = {'error': 'no_models_trained'}
                        continue

                    best_model_type = max(
                        model_performances.keys(),
                        key=lambda x: model_performances[x].get_composite_score()
                    )
                    best_performance = model_performances[best_model_type]

                    # Dataset manifest
                    dataset_manifest = self._build_dataset_manifest(dataset)

                    # Save artifact
                    await self._save_best_model(
                        symbol,
                        best_model_type,
                        best_performance,
                        dataset_manifest,
                        feature_names=dataset.feature_names,
                    )

                    # Baseline uplift if walk-forward existed
                    wf = getattr(self.trainer, 'walk_forward_last', {}).get(best_model_type, {})
                    composite = best_performance.get_composite_score()
                    p_value = None
                    if wf.get('total_samples') and wf.get('model_hits') is not None:
                        p_value = self._binomial_p_value(wf['model_hits'], wf['total_samples'], 0.5)
                        wf['p_value_directional'] = p_value
                    job_results[symbol] = {
                        'best_model': best_model_type,
                        'performance': best_performance.to_summary_dict(),
                        'all_performances': {k: v.to_summary_dict() for k, v in model_performances.items()},
                        'dataset_manifest': dataset_manifest,
                        'walk_forward': wf,
                        'composite_score': composite,
                    }
                    try:
                        logger.info(json.dumps({
                            'event': 'symbol_training_complete',
                            'job_id': job.job_id,
                            'symbol': symbol,
                            'best_model': best_model_type,
                            'composite': composite,
                            'wf_uplift': wf.get('uplift_vs_baseline'),
                            'wf_dir_acc': wf.get('directional_accuracy_walk_forward'),
                            'baseline_dir_acc': wf.get('baseline_directional_accuracy'),
                            'wf_p_value': p_value,
                            'samples': len(dataset.features),
                            'timestamp': datetime.utcnow().isoformat(),
                        }))
                    except Exception:
                        pass

                    elapsed_symbol = (datetime.utcnow() - symbol_start).total_seconds()
                    logger.info(
                        f"Completed training for {symbol}: best={best_model_type} "
                        f"score={best_performance.get_composite_score():.3f} "
                        f"samples={len(dataset.features)} elapsed={elapsed_symbol:.1f}s"
                    )
                except Exception as e:
                    logger.error(f"Failed to train models for {symbol}: {e}")
                    job_results[symbol] = {'error': str(e)}

            # Persist job results
            await self._save_job_results(job, {**job_results, '_job_meta': job_seed_info})

            training_duration = (datetime.utcnow() - start_time).total_seconds()
            self.jobs_completed += 1
            self.total_training_time += training_duration

            # Metrics update
            try:
                self._metric_jobs_total.labels(status='completed').inc()
                self._metric_job_duration.observe(training_duration)
            except Exception:
                pass

            success_count = len([r for r in job_results.values() if isinstance(r, dict) and 'error' not in r])
            logger.info(
                f"Completed training job {job.job_id} in {training_duration:.1f}s; "
                f"successful_symbols={success_count} total_symbols={len(job.symbols)}"
            )
        except Exception as e:
            logger.error(f"Training job {job.job_id} failed: {e}")
            try:
                self._metric_jobs_total.labels(status='failed').inc()
            except Exception:
                pass
        finally:
            self.active_jobs.discard(job.job_id)
            try:
                self._metric_job_active.set(len(self.active_jobs))
            except Exception:
                pass
    
    async def _get_historical_data(self, symbol: str, days_back: int) -> List[MarketData]:
        """Get historical market data for training."""
        start_fetch = datetime.utcnow()
        records: List[MarketData] = []
        try:
            # Primary: cached JSON array in trading cache (if ingestion service stores it)
            if self.cache:
                cache_key = f"historical_prices:{symbol}:{days_back}"  # convention
                raw = await self.cache.get_json(cache_key)
                if raw and isinstance(raw, list):
                    for item in raw:
                        # Expecting dict with needed fields
                        try:
                            records.append(
                                MarketData(
                                    timestamp=datetime.fromisoformat(item['timestamp']),
                                    open=item.get('open'),
                                    high=item.get('high'),
                                    low=item.get('low'),
                                    close=item.get('close'),
                                    volume=item.get('volume')
                                )
                            )
                        except Exception:
                            continue
            # Fallback: (placeholder) return empty list if cache miss
        except Exception as e:
            logger.warning(f"Historical data fetch failed for {symbol}: {e}")
        finally:
            elapsed = (datetime.utcnow() - start_fetch).total_seconds()
            try:
                self._metric_data_fetch.observe(elapsed)
                self._metric_data_points.labels(symbol=symbol).set(len(records))
            except Exception:
                pass
        # Basic integrity check
        if records:
            # Ensure chronological order
            if any(records[i].timestamp > records[i+1].timestamp for i in range(len(records)-1)):
                records.sort(key=lambda r: r.timestamp)
                try:
                    self._metric_integrity_warnings.labels(type='timestamp_order').inc()
                except Exception:
                    pass
        return records
    
    async def _save_best_model(
        self,
        symbol: str,
        model_type: str,
        performance: ModelPerformance,
        dataset_manifest: Dict[str, Any],
        feature_names: Optional[List[str]] = None,
    ):
        """Save the best performing model with metadata and dataset manifest."""
        
        # Update best models tracking
        current_best = self.best_models.get(symbol)
        if not current_best or performance.get_composite_score() > current_best['score']:
            self.best_models[symbol] = {
                'model_type': model_type,
                'performance': performance,
                'score': performance.get_composite_score(),
                'timestamp': datetime.utcnow()
            }
        
        # Save model to disk
        model_key = f"{symbol}_{model_type}"
        if model_key in self.trainer.trained_models:
            model_path = self.models_dir / f"{symbol}_{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"

            model_data = {
                'model': self.trainer.trained_models[model_key],
                'scaler': self.preprocessor.scalers.get(symbol),
                'performance': performance,
                'feature_names': feature_names or [],
                'dataset_manifest': dataset_manifest,
                'timestamp': datetime.utcnow(),
                'artifact_version': 1,
            }

            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            # Compute artifact hash for integrity tracking
            sha256_hex = None
            try:
                h = hashlib.sha256()
                with open(model_path, 'rb') as rf:
                    for chunk in iter(lambda: rf.read(8192), b''):
                        h.update(chunk)
                sha256_hex = h.hexdigest()
            except Exception as e:
                logger.warning(f"Artifact hashing failed for {model_path}: {e}")
                try:
                    self._metric_integrity_warnings.labels(type='artifact_hash').inc()
                except Exception:
                    pass

            repro_info = self._build_repro_manifest()
            if sha256_hex and self.cache:
                try:
                    await self.cache.set_json(
                        f"model_artifact:{symbol}:{model_type}",
                        {
                            'path': str(model_path),
                            'sha256': sha256_hex,
                            'saved_at': datetime.utcnow().isoformat(),
                            'score': performance.get_composite_score(),
                            'dataset_hash': dataset_manifest.get('sample_hash'),
                            'artifact_version': model_data['artifact_version'],
                            'repro': repro_info,
                        },
                        ttl=86400 * 30
                    )
                except Exception as e:
                    logger.warning(f"Cache store failed for artifact metadata {model_path}: {e}")

            try:
                self._metric_models_saved_total.labels(symbol=symbol, model_type=model_type).inc()
            except Exception:
                pass

            logger.info(
                f"Saved model artifact {model_key} path={model_path.name} hash={sha256_hex} "
                f"score={performance.get_composite_score():.3f}"
            )

            # Optional: Upload artifact to MinIO for centralized storage
            await self._maybe_upload_to_minio(
                symbol=symbol,
                model_type=model_type,
                local_path=model_path,
                sha256_hex=sha256_hex,
                performance=performance,
            )
        
        # Cache model performance
        if self.cache:
            cache_key = f"model_performance:{symbol}:{model_type}"
            performance_data = asdict(performance)
            await self.cache.set_json(cache_key, performance_data, ttl=86400)  # 24 hours

    async def _maybe_upload_to_minio(
        self,
        *,
        symbol: str,
        model_type: str,
        local_path: Path,
        sha256_hex: Optional[str],
        performance: ModelPerformance,
    ) -> None:
        """Upload model artifact to MinIO if configuration available.
        Safe no-op if MinIO is not configured or client unavailable.
        """
        if get_minio_client is None or put_bytes is None or build_model_artifact_key is None:
            return  # MinIO integration not available
        try:
            bucket = os.getenv("TRAINING_MODELS_BUCKET", "models")
            client = get_minio_client()
            if ensure_bucket:
                try:
                    ensure_bucket(bucket, client=client)
                except Exception:
                    pass  # non-fatal
            # Read file bytes
            data = b""
            with open(local_path, "rb") as rf:
                data = rf.read()
            version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            filename = local_path.name
            object_key = build_model_artifact_key(
                model_name=f"{symbol}_{model_type}",
                version=version,
                filename=filename,
            )
            meta = put_bytes(
                bucket=bucket,
                data=data,
                object_key=object_key,
                content_type="application/octet-stream",
                verify_integrity=True,
                client=client,
                extra_headers={
                    "x-amz-meta-symbol": symbol,
                    "x-amz-meta-model-type": model_type,
                    "x-amz-meta-sha256": sha256_hex or "",
                    "x-amz-meta-score": str(performance.get_composite_score()),
                },
            )
            logger.info(
                f"Uploaded model artifact to MinIO: s3://{bucket}/{object_key} size={meta.get('size_bytes')}"
            )
        except StorageError as se:  # pragma: no cover - network path
            logger.warning(f"MinIO upload failed: {se}")
        except Exception as e:  # pragma: no cover - network path
            logger.warning(f"MinIO upload error: {e}")

    def _derive_seed(self, job_id: str) -> int:
        """Derive a deterministic seed from job id."""
        h = hashlib.sha256(job_id.encode('utf-8')).hexdigest()[:8]
        return int(h, 16)

    def _apply_seeds(self, seed: int):
        """Apply global seeds for reproducibility."""
        try:
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
        except Exception:
            pass
        # Optional: torch seed if available
        try:
            import torch  # type: ignore
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    def _build_dataset_manifest(self, dataset: TrainingDataset) -> Dict[str, Any]:
        """Build a lightweight manifest capturing dataset lineage & hash."""
        # Hash only a sample (first 100 rows) for efficiency
        sample_size = min(100, len(dataset.features))
        features_sample = dataset.features[:sample_size].tobytes()
        targets_sample = dataset.targets[:sample_size].tobytes()
        raw = features_sample + targets_sample
        digest = hashlib.sha256(raw).hexdigest()
        manifest = {
            'symbol': dataset.symbol,
            'num_samples': len(dataset.features),
            'num_features': dataset.features.shape[1] if len(dataset.features) else 0,
            'feature_names': dataset.feature_names,
            'target_names': dataset.target_names,
            'train_size': dataset.train_size,
            'test_size': dataset.test_size,
            'start_timestamp': dataset.timestamps[0].isoformat() if dataset.timestamps else None,
            'end_timestamp': dataset.timestamps[-1].isoformat() if dataset.timestamps else None,
            'sample_hash': digest,
            'hash_sample_size': sample_size,
            'created_at': datetime.utcnow().isoformat(),
            'schema_version': 1,
        }
        return manifest

    def _binomial_p_value(self, successes: int, n: int, p: float) -> float:
        """Approximate two-tailed p-value (normal approximation)."""
        if n == 0:
            return 1.0
        mean = n * p
        var = n * p * (1 - p)
        if var == 0:
            return 1.0
        z = (successes - mean) / (var ** 0.5)
        # Two-tailed using error function approximation
        try:
            import math
            # Phi(z)
            phi = 0.5 * (1 + math.erf(z / (2 ** 0.5)))
            p_two = 2 * min(phi, 1 - phi)
            return float(max(min(p_two, 1.0), 0.0))
        except Exception:
            return 1.0

    def _build_repro_manifest(self) -> Dict[str, Any]:
        """Capture reproducibility environment snapshot (lightweight)."""
        py_ver = platform.python_version()
        uname = platform.uname()._asdict()
        git_commit = None
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            pass
        return {
            'python_version': py_ver,
            'platform': uname,
            'git_commit': git_commit,
            'created_at': datetime.utcnow().isoformat(),
        }
    
    async def _save_job_results(self, job: TrainingJob, results: Dict[str, Any]):
        """Save training job results."""
        
        job_summary = {
            'job_id': job.job_id,
            'symbols': job.symbols,
            'model_types': job.model_types,
            'start_time': job.start_time.isoformat(),
            'completion_time': datetime.utcnow().isoformat(),
            'results': results,
            'successful_symbols': len([r for r in results.values() if 'error' not in r]),
            'failed_symbols': len([r for r in results.values() if 'error' in r])
        }
        
        # Cache job results
        if self.cache:
            cache_key = f"training_job_results:{job.job_id}"
            await self.cache.set_json(cache_key, job_summary, ttl=86400 * 7)  # 7 days
    
    async def _schedule_training_sessions(self):
        """Schedule regular training sessions."""
        while self.is_running:
            try:
                # Check every hour if it's time to start training
                await asyncio.sleep(3600)  # 1 hour
                
                if not self.training_enabled:
                    continue
                
                if self.is_training_time():
                    # Get symbols that need training
                    symbols_to_train = await self._get_symbols_needing_training()
                    
                    if symbols_to_train:
                        await self.schedule_training_job(
                            symbols=symbols_to_train,
                            priority=3  # Low priority for scheduled jobs
                        )
                        logger.info(f"Scheduled training for {len(symbols_to_train)} symbols")
                
            except Exception as e:
                logger.error(f"Error in training session scheduler: {e}")
    
    async def _get_symbols_needing_training(self) -> List[str]:
        """Get symbols that need model retraining."""
        # This would check which symbols haven't been trained recently
        # or have poor performing models
        return []  # Would return actual symbols from watchlist/portfolio
    
    async def enable_training(self):
        """Enable off-hours training."""
        self.training_enabled = True
        logger.info("Off-hours training enabled")
    
    async def disable_training(self):
        """Disable off-hours training."""
        self.training_enabled = False
        logger.info("Off-hours training disabled")
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            'training_enabled': self.training_enabled,
            'is_training_time': self.is_training_time(),
            'active_jobs': len(self.active_jobs),
            'queued_jobs': self.training_jobs.qsize(),
            'jobs_completed': self.jobs_completed,
            'total_training_time': self.total_training_time,
            'avg_job_time': self.total_training_time / max(self.jobs_completed, 1),
            'best_models': {symbol: info['model_type'] for symbol, info in self.best_models.items()}
        }
    
    async def get_model_performance(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for symbol's best model."""
        if symbol in self.best_models:
            return {
                'symbol': symbol,
                'model_type': self.best_models[symbol]['model_type'],
                'performance': asdict(self.best_models[symbol]['performance']),
                'score': self.best_models[symbol]['score'],
                'last_trained': self.best_models[symbol]['timestamp'].isoformat()
            }
        return None
    
    async def stop(self):
        """Stop the training service."""
        self.is_running = False
        logger.info("Off-Hours Training Service stopped")


# Global training service instance
training_service: Optional[OffHoursTrainingService] = None


async def get_training_service() -> OffHoursTrainingService:
    """Get or create training service instance."""
    global training_service
    if training_service is None:
        training_service = OffHoursTrainingService()
        await training_service.initialize()
    return training_service