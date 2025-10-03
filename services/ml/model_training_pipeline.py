#!/usr/bin/env python3
"""
Advanced Model Training and Fine-Tuning Pipeline
Implements state-of-the-art training with hyperparameter optimization,
cross-validation, and automated model selection.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import optuna
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))

from trading_common import get_logger, get_settings
from trading_common.database_manager import get_database_manager

logger = get_logger(__name__)
settings = get_settings()


def sharpe_ratio(returns, risk_free_rate=0):
    """Calculate Sharpe ratio."""
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0


def sortino_ratio(returns, risk_free_rate=0):
    """Calculate Sortino ratio."""
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
    return np.mean(excess_returns) / downside_std if downside_std > 0 else 0


def calmar_ratio(returns):
    """Calculate Calmar ratio."""
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    annual_return = returns.mean() * 252  # Assuming daily returns
    return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_type: str  # 'deep_learning', 'gradient_boosting', 'ensemble'
    task_type: str  # 'price_prediction', 'volatility_forecast', 'regime_detection'
    
    # Data parameters
    lookback_window: int = 60
    forecast_horizon: int = 5
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10
    
    # Optimization
    use_hyperopt: bool = True
    n_trials: int = 50
    cv_folds: int = 5
    
    # Advanced features
    use_mixup: bool = True
    use_label_smoothing: bool = True
    use_gradient_accumulation: bool = True
    accumulation_steps: int = 4


class AdvancedTradingModel(nn.Module):
    """
    State-of-the-art deep learning model for trading.
    Combines Transformer, LSTM, and Attention mechanisms.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Temporal Convolutional Network
        self.tcn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Bidirectional LSTM with attention
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=3,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )
        
        # Mixture of Experts layer
        self.num_experts = 4
        self.experts = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim // 2)
            for _ in range(self.num_experts)
        ])
        self.gating = nn.Linear(hidden_dim, self.num_experts)
        
        # Output layers with residual connections
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 3)  # [price_change, volatility, confidence]
        )
        
        # Uncertainty estimation
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensures positive uncertainty
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out  # Residual connection
        
        # TCN processing
        tcn_out = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        
        # LSTM with skip connection
        lstm_out, _ = self.lstm(tcn_out)
        
        # Mixture of Experts
        gates = F.softmax(self.gating(lstm_out), dim=-1)
        expert_outputs = torch.stack([
            expert(lstm_out) for expert in self.experts
        ], dim=1)
        moe_out = torch.einsum('bte,btne->btn', gates, expert_outputs).mean(dim=1)
        
        # Final predictions with uncertainty
        predictions = self.output_layers(moe_out)
        uncertainty = self.uncertainty(moe_out)
        
        return predictions, uncertainty


class ModelTrainingPipeline:
    """
    Complete model training pipeline with automated optimization.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.models = {}
        self.best_model = None
        self.training_history = []
        self.feature_importance = {}
        
        # Initialize MLflow for experiment tracking
        mlflow.set_experiment("ai_trading_models")
    
    async def train_model(
        self,
        data: pd.DataFrame,
        features: List[str],
        target: str,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Train a model with full pipeline.
        """
        logger.info(f"Starting training for {model_name}")
        
        # Prepare data
        X, y = self._prepare_data(data, features, target)
        
        # Split data temporally
        X_train, X_val, X_test, y_train, y_val, y_test = self._temporal_split(X, y)
        
        # Hyperparameter optimization
        if self.config.use_hyperopt:
            best_params = await self._optimize_hyperparameters(
                X_train, y_train, X_val, y_val
            )
        else:
            best_params = self._get_default_params()
        
        # Train final model
        model = await self._train_final_model(
            X_train, y_train, X_val, y_val, best_params
        )
        
        # Evaluate model
        metrics = self._evaluate_model(model, X_test, y_test)
        
        # Calculate feature importance
        self.feature_importance[model_name] = self._get_feature_importance(
            model, features
        )
        
        # Save model and metrics
        self._save_model(model, model_name, metrics)
        
        return {
            'model': model,
            'metrics': metrics,
            'params': best_params,
            'feature_importance': self.feature_importance[model_name]
        }

    async def run_training_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """End-to-end training using QuestDB daily data.

        Expected job keys (defaults in parentheses):
          - model_name (required)
          - model_type ('ensemble'|'gradient_boosting'|'deep_learning')
          - task_type ('price_prediction' default)
          - symbols (list[str], required)
          - start_date (YYYY-MM-DD, default 5y ago)
          - end_date (YYYY-MM-DD, default today)
          - horizon_days (int, default 1) for next-day returns target
          - features (list[str], optional; uses default feature set if missing)
        """
        model_name = str(job.get('model_name') or job.get('name') or 'model')
        symbols: List[str] = [str(s).upper() for s in (job.get('symbols') or []) if str(s).strip()]
        if not symbols:
            raise ValueError('symbols list is required in training job')
        # Configure pipeline type dynamically if provided
        mt = job.get('model_type')
        if isinstance(mt, str) and mt:
            self.config.model_type = mt
        tt = job.get('task_type')
        if isinstance(tt, str) and tt:
            self.config.task_type = tt

        # Date window
        today = datetime.utcnow().date()
        default_start = datetime(today.year - 5, today.month, min(28, today.day))
        try:
            start_dt = datetime.strptime(job.get('start_date', ''), '%Y-%m-%d') if job.get('start_date') else default_start
        except Exception:
            start_dt = default_start
        try:
            end_dt = datetime.strptime(job.get('end_date', ''), '%Y-%m-%d') if job.get('end_date') else datetime(today.year, today.month, today.day)
        except Exception:
            end_dt = datetime(today.year, today.month, today.day)
        horizon = int(job.get('horizon_days', 1) or 1)

        # Fetch data from QuestDB (daily close at 16:00)
        dbm = await get_database_manager()
        async with dbm.get_questdb() as conn:
            sql = (
                "SELECT symbol, timestamp, open, high, low, close, volume FROM market_data "
                "WHERE symbol = ANY($1::text[]) AND to_char(timestamp, 'HH24:MI:SS') = '16:00:00' "
                "AND timestamp >= $2 AND timestamp < $3 ORDER BY symbol, timestamp"
            )
            rows = await conn.fetch(sql, symbols, start_dt, end_dt)

        if not rows:
            raise ValueError('No data returned from QuestDB for given window')

        df = pd.DataFrame([dict(r) for r in rows])
        # Ensure proper dtypes
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

        # Feature engineering per symbol
        def _fe(group: pd.DataFrame) -> pd.DataFrame:
            g = group.copy()
            g['returns'] = g['close'].pct_change()
            g['rsi14'] = _calculate_rsi(g['close'], period=14)
            macd_line, macd_signal, macd_hist = _calculate_macd(g['close'])
            g['macd_line'] = macd_line
            g['macd_signal'] = macd_signal
            bb_u, bb_l = _calculate_bollinger(g['close'])
            g['bb_upper'] = bb_u
            g['bb_lower'] = bb_l
            g['realized_vol20'] = g['returns'].rolling(20).std().fillna(0.0)
            # Target: forward return over horizon
            g['target_return'] = g['close'].pct_change(periods=horizon).shift(-horizon)
            return g

        df = df.groupby('symbol', group_keys=False).apply(_fe)
        # Drop rows without target
        df = df.dropna(subset=['target_return']).reset_index(drop=True)

        default_features = [
            'rsi14','macd_line','macd_signal','bb_upper','bb_lower','realized_vol20',
            'open','high','low','close','volume'
        ]
        features: List[str] = [str(f) for f in (job.get('features') or default_features)]
        target = 'target_return'

        result = await self.train_model(df, features, target, model_name)
        return result
    
    async def _optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna with Bayesian optimization.
        """
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                'num_layers': trial.suggest_int('num_layers', 2, 5),
                'hidden_dim': trial.suggest_int('hidden_dim', 64, 512, step=64),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'batch_size': trial.suggest_int('batch_size', 16, 128, step=16)
            }
            
            # Train model with suggested params
            model = self._create_model(params)
            val_score = self._train_and_evaluate(
                model, X_train, y_train, X_val, y_val, params
            )
            
            return val_score
        
        # Create study with pruning
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Run optimization
        study.optimize(objective, n_trials=self.config.n_trials)
        
        logger.info(f"Best trial: {study.best_trial.value}")
        return study.best_params
    
    def _prepare_data(
        self,
        data: pd.DataFrame,
        features: List[str],
        target: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data with advanced feature engineering.
        """
        # Feature engineering (idempotent; will only add missing engineered columns)
        data = self._engineer_features(data)
        
        # Handle missing values
        data = self._handle_missing(data)
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(data[features])
        y = data[target].values
        
        # Create sequences for time series
        X_seq, y_seq = self._create_sequences(
            X, y, self.config.lookback_window, self.config.forecast_horizon
        )
        
        return X_seq, y_seq
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering for trading.
        """
        # Only add features if missing to support pre-engineered inputs
        series_close = data['close']
        if 'rsi' not in data.columns and 'rsi14' not in data.columns:
            data['rsi14'] = _calculate_rsi(series_close, period=14)
        if 'macd_line' not in data.columns or 'macd_signal' not in data.columns:
            macd_line, macd_signal, _ = _calculate_macd(series_close)
            data['macd_line'] = macd_line
            data['macd_signal'] = macd_signal
        if 'bb_upper' not in data.columns or 'bb_lower' not in data.columns:
            bb_u, bb_l = _calculate_bollinger(series_close)
            data['bb_upper'] = bb_u
            data['bb_lower'] = bb_l
        if 'returns' not in data.columns:
            data['returns'] = data['close'].pct_change()
        if 'realized_vol20' not in data.columns:
            data['realized_vol20'] = data['returns'].rolling(20).std()
        # Lightweight microstructure placeholders (avoid dependencies on tick data)
        if 'volume_imbalance' not in data.columns:
            data['volume_imbalance'] = _calculate_volume_imbalance(data)
        if 'order_flow' not in data.columns:
            data['order_flow'] = _calculate_order_flow(data)
        return data
    
    def _train_final_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict[str, Any]
    ) -> Any:
        """
        Train the final model with best parameters.
        """
        if self.config.model_type == 'deep_learning':
            model = self._train_deep_learning(X_train, y_train, X_val, y_val, params)
        elif self.config.model_type == 'gradient_boosting':
            model = self._train_gradient_boosting(X_train, y_train, X_val, y_val, params)
        else:
            model = self._train_ensemble(X_train, y_train, X_val, y_val, params)
        
        return model
    
    def _train_deep_learning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict[str, Any]
    ) -> nn.Module:
        """
        Train advanced deep learning model.
        """
        # Create model
        model = AdvancedTradingModel(
            input_dim=X_train.shape[-1],
            hidden_dim=params.get('hidden_dim', 256)
        )
        
        # Setup training
        optimizer = AdamW(model.parameters(), lr=params['learning_rate'])
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        criterion = nn.MSELoss()
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        
        # Training loop with advanced techniques
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            model.train()
            
            # Mixup augmentation
            if self.config.use_mixup:
                X_train_t, y_train_t = self._mixup(X_train_t, y_train_t)
            
            # Forward pass
            predictions, uncertainty = model(X_train_t)
            loss = criterion(predictions[:, 0], y_train_t)
            
            # Add uncertainty regularization
            uncertainty_loss = uncertainty.mean() * 0.01
            total_loss = loss + uncertainty_loss
            
            # Backward pass with gradient accumulation
            if self.config.use_gradient_accumulation:
                total_loss = total_loss / self.config.accumulation_steps
            
            total_loss.backward()
            
            if (epoch + 1) % self.config.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred, val_unc = model(X_val_t)
                val_loss = criterion(val_pred[:, 0], y_val_t)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"best_model_{epoch}.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return model
    
    def _train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict[str, Any]
    ) -> Any:
        """
        Train ensemble of gradient boosting models.
        """
        models = []
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=params.get('n_estimators', 1000),
            learning_rate=params.get('learning_rate', 0.01),
            max_depth=params.get('max_depth', -1),
            num_leaves=params.get('num_leaves', 31),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            objective='regression',
            metric='rmse'
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
        )
        models.append(('lightgbm', lgb_model))
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=params.get('n_estimators', 1000),
            learning_rate=params.get('learning_rate', 0.01),
            max_depth=params.get('max_depth', 6),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8)
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=False
        )
        models.append(('xgboost', xgb_model))
        
        # CatBoost
        cat_model = CatBoostRegressor(
            iterations=params.get('n_estimators', 1000),
            learning_rate=params.get('learning_rate', 0.01),
            depth=params.get('max_depth', 6),
            verbose=False
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
        models.append(('catboost', cat_model))
        
        return models
    
    def _evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation with trading metrics.
        """
        predictions = self._get_predictions(model, X_test)
        
        metrics = {
            # Standard metrics
            'mse': np.mean((predictions - y_test) ** 2),
            'mae': np.mean(np.abs(predictions - y_test)),
            'rmse': np.sqrt(np.mean((predictions - y_test) ** 2)),
            
            # Trading metrics
            'sharpe_ratio': self._calculate_sharpe(predictions, y_test),
            'sortino_ratio': self._calculate_sortino(predictions, y_test),
            'calmar_ratio': self._calculate_calmar(predictions, y_test),
            'max_drawdown': self._calculate_max_drawdown(predictions),
            
            # Directional accuracy
            'directional_accuracy': self._directional_accuracy(predictions, y_test),
            'profit_factor': self._profit_factor(predictions, y_test)
        }
        
        return metrics
    
    def _calculate_sharpe(self, returns: np.ndarray, benchmark: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - benchmark
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)
    
    def _calculate_sortino(self, returns: np.ndarray, benchmark: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - benchmark
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        return np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    def _mixup(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mixup data augmentation for better generalization.
        """
        batch_size = X.shape[0]
        lam = np.random.beta(alpha, alpha)
        
        index = torch.randperm(batch_size)
        mixed_X = lam * X + (1 - lam) * X[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_X, mixed_y
    
    def _save_model(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float]
    ):
        """
        Save model with MLflow tracking.
        """
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config.__dict__)
            
            # Log metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Log model
            if isinstance(model, nn.Module):
                mlflow.pytorch.log_model(model, model_name)
            else:
                mlflow.sklearn.log_model(model, model_name)
            
            # Log feature importance
            if model_name in self.feature_importance:
                mlflow.log_dict(
                    self.feature_importance[model_name],
                    f"feature_importance_{model_name}.json"
                )


# Initialize global training pipeline
_training_pipeline: Optional[ModelTrainingPipeline] = None


async def get_training_pipeline() -> ModelTrainingPipeline:
    """Get or create training pipeline instance."""
    global _training_pipeline
    if _training_pipeline is None:
        config = TrainingConfig(
            model_type='ensemble',
            use_hyperopt=True,
            n_trials=100
        )
        _training_pipeline = ModelTrainingPipeline(config)
    return _training_pipeline

# --- Technical Indicator Helpers (local, vectorized) ---

def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    close = close.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method='bfill').fillna(50.0)

def _calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    close = close.astype(float)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line.fillna(0.0), signal_line.fillna(0.0), hist.fillna(0.0)

def _calculate_bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    close = close.astype(float)
    ma = close.rolling(window=window, min_periods=window).mean()
    sd = close.rolling(window=window, min_periods=window).std()
    upper = (ma + num_std * sd).fillna(method='bfill')
    lower = (ma - num_std * sd).fillna(method='bfill')
    return upper, lower

def _calculate_volume_imbalance(df: pd.DataFrame) -> pd.Series:
    # Simple heuristic: (volume - rolling median) / rolling median
    vol = df['volume'].astype(float)
    med = vol.rolling(10, min_periods=3).median()
    out = (vol - med) / (med.replace(0, np.nan))
    return out.fillna(0.0)

def _calculate_order_flow(df: pd.DataFrame) -> pd.Series:
    # Proxy using candle direction and range
    rng = (df['high'].astype(float) - df['low'].astype(float)).replace(0, np.nan)
    dirn = np.sign(df['close'].astype(float) - df['open'].astype(float))
    of = (dirn * (df['close'].astype(float) - df['open'].astype(float)).abs()) / rng
    return of.replace([np.inf, -np.inf], 0.0).fillna(0.0)
