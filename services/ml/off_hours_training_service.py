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
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from trading_common import MarketData, get_settings, get_logger
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
        
        # Split data for validation
        train_set, test_set = dataset.split_train_test(test_ratio=0.2)
        
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
                
                # Evaluate performance
                performance = await self._evaluate_model(
                    model, model_type, test_set, training_time, best_params
                )
                
                performances[model_type] = performance
                
                # Store trained model
                model_key = f"{dataset.symbol}_{model_type}"
                self.trained_models[model_key] = model
                
                logger.info(f"Completed {model_type} for {dataset.symbol}: "
                           f"Score={performance.get_composite_score():.3f}, "
                           f"Directional Accuracy={performance.directional_accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_type} for {dataset.symbol}: {e}")
        
        return performances
    
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
        returns = y_pred * np.sign(y_pred)  # Simple strategy: trade in predicted direction
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
        
        # Model persistence
        self.models_dir = Path("/tmp/trading_models")
        self.models_dir.mkdir(exist_ok=True)
    
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
                asyncio.create_task(self._execute_training_job(job))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in training job processor: {e}")
    
    async def _execute_training_job(self, job: TrainingJob):
        """Execute a training job."""
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting training job {job.job_id} with {len(job.symbols)} symbols")
            
            job_results = {}
            
            for symbol in job.symbols:
                try:
                    # Get historical data for symbol
                    market_data = await self._get_historical_data(symbol, job.data_requirements['days_back'])
                    
                    if not market_data or len(market_data) < job.data_requirements['min_samples']:
                        logger.warning(f"Insufficient data for {symbol}, skipping")
                        continue
                    
                    # Prepare training dataset
                    dataset = await self.preprocessor.prepare_training_data(symbol, market_data)
                    
                    if not dataset:
                        logger.warning(f"Failed to prepare dataset for {symbol}, skipping")
                        continue
                    
                    # Train models
                    model_performances = await self.trainer.train_models(dataset, job.model_types)
                    
                    # Find best performing model
                    best_model_type = max(model_performances.keys(), 
                                        key=lambda x: model_performances[x].get_composite_score())
                    best_performance = model_performances[best_model_type]
                    
                    # Save best model
                    await self._save_best_model(symbol, best_model_type, best_performance)
                    
                    job_results[symbol] = {
                        'best_model': best_model_type,
                        'performance': best_performance,
                        'all_performances': model_performances
                    }
                    
                    logger.info(f"Completed training for {symbol}: best model = {best_model_type} "
                               f"(score: {best_performance.get_composite_score():.3f})")
                    
                except Exception as e:
                    logger.error(f"Failed to train models for {symbol}: {e}")
                    job_results[symbol] = {'error': str(e)}
            
            # Save job results
            await self._save_job_results(job, job_results)
            
            training_duration = (datetime.utcnow() - start_time).total_seconds()
            self.jobs_completed += 1
            self.total_training_time += training_duration
            
            logger.info(f"Completed training job {job.job_id} in {training_duration:.1f} seconds. "
                       f"Trained {len([r for r in job_results.values() if 'error' not in r])} symbols successfully.")
            
        except Exception as e:
            logger.error(f"Training job {job.job_id} failed: {e}")
        finally:
            self.active_jobs.discard(job.job_id)
    
    async def _get_historical_data(self, symbol: str, days_back: int) -> List[MarketData]:
        """Get historical market data for training."""
        # This would integrate with the market data service
        # For now, return empty list (would implement actual data retrieval)
        return []
    
    async def _save_best_model(self, symbol: str, model_type: str, performance: ModelPerformance):
        """Save the best performing model."""
        
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
            model_path = self.models_dir / f"{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d')}.pkl"
            
            model_data = {
                'model': self.trainer.trained_models[model_key],
                'scaler': self.preprocessor.scalers.get(symbol),
                'performance': performance,
                'feature_names': [],  # Would include actual feature names
                'timestamp': datetime.utcnow()
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Saved model {model_key} to {model_path}")
        
        # Cache model performance
        if self.cache:
            cache_key = f"model_performance:{symbol}:{model_type}"
            performance_data = asdict(performance)
            await self.cache.set_json(cache_key, performance_data, ttl=86400)  # 24 hours
    
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