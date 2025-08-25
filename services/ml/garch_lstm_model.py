#!/usr/bin/env python3
"""
GARCH-LSTM Hybrid Model for Volatility Prediction and Price Forecasting.
Combines GARCH for volatility modeling with LSTM for sequence prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import asyncio
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from arch import arch_model
    from arch.univariate import GARCH
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from trading_common import MarketData, get_logger, get_metrics_registry
from trading_common.ml_registry import get_model_registry, ModelMetadata, ModelType, ModelStatus
from trading_common.metrics import track_model_metrics

logger = get_logger(__name__)
metrics = get_metrics_registry()


class GARCHModel:
    """GARCH model for volatility estimation."""
    
    def __init__(self, p: int = 1, q: int = 1):
        """Initialize GARCH model."""
        self.p = p  # GARCH order
        self.q = q  # ARCH order
        self.model = None
        self.fitted_model = None
        self.volatility_scaler = MinMaxScaler() if SKLEARN_AVAILABLE else None
    
    def fit(self, returns: np.ndarray) -> Dict[str, Any]:
        """Fit GARCH model to return series."""
        if not ARCH_AVAILABLE:
            raise ImportError("arch package required for GARCH modeling")
        
        try:
            # Convert returns to percentage returns
            returns_pct = returns * 100
            
            # Fit GARCH model
            self.model = arch_model(
                returns_pct, 
                vol='GARCH', 
                p=self.p, 
                q=self.q,
                dist='normal'
            )
            
            self.fitted_model = self.model.fit(disp='off', show_warning=False)
            
            # Calculate fitted values
            conditional_volatility = self.fitted_model.conditional_volatility / 100  # Convert back from %
            
            # Fit scaler on volatility
            if self.volatility_scaler:
                self.volatility_scaler.fit(conditional_volatility.values.reshape(-1, 1))
            
            return {
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'log_likelihood': self.fitted_model.loglikelihood,
                'volatility_mean': conditional_volatility.mean(),
                'volatility_std': conditional_volatility.std()
            }
            
        except Exception as e:
            logger.error(f"GARCH model fitting failed: {e}")
            raise
    
    def predict_volatility(self, horizon: int = 1) -> np.ndarray:
        """Predict future volatility."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Generate volatility forecasts
            forecasts = self.fitted_model.forecast(horizon=horizon, method='simulation')
            volatility_forecast = np.sqrt(forecasts.variance.values[-horizon:, :]) / 100
            
            # Scale volatility if scaler is available
            if self.volatility_scaler:
                volatility_forecast = self.volatility_scaler.transform(
                    volatility_forecast.reshape(-1, 1)
                ).flatten()
            
            return volatility_forecast
            
        except Exception as e:
            logger.error(f"GARCH volatility prediction failed: {e}")
            raise
    
    def get_conditional_volatility(self) -> np.ndarray:
        """Get fitted conditional volatility."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        volatility = self.fitted_model.conditional_volatility.values / 100
        
        # Scale if scaler is available
        if self.volatility_scaler:
            volatility = self.volatility_scaler.transform(volatility.reshape(-1, 1)).flatten()
        
        return volatility


class LSTMModel(nn.Module):
    """LSTM neural network for sequence prediction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """Initialize LSTM model."""
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        batch_size = x.size(0)
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention to LSTM output
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last output for prediction
        last_output = attn_output[:, -1, :]
        
        # Dropout and final prediction
        output = self.dropout(last_output)
        prediction = self.fc(output)
        
        return prediction


class GARCHLSTMHybridModel:
    """Hybrid GARCH-LSTM model for comprehensive price and volatility prediction."""
    
    def __init__(
        self,
        sequence_length: int = 60,
        garch_p: int = 1,
        garch_q: int = 1,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        """Initialize hybrid model."""
        self.sequence_length = sequence_length
        self.device = device
        self.learning_rate = learning_rate
        
        # Component models
        self.garch_model = GARCHModel(p=garch_p, q=garch_q)
        self.lstm_model = None
        
        # Data preprocessing
        self.price_scaler = MinMaxScaler() if SKLEARN_AVAILABLE else None
        self.feature_scaler = MinMaxScaler() if SKLEARN_AVAILABLE else None
        
        # Model parameters
        self.lstm_params = {
            'hidden_size': lstm_hidden_size,
            'num_layers': lstm_num_layers,
            'dropout': lstm_dropout
        }
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'garch_metrics': {}
        }
        
        # Model state
        self.is_fitted = False
        self.feature_names = []
    
    def _prepare_features(self, market_data: List[MarketData]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for LSTM including GARCH volatility."""
        if len(market_data) < self.sequence_length + 1:
            raise ValueError(f"Need at least {self.sequence_length + 1} data points")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame([
            {
                'timestamp': md.timestamp,
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': md.volume
            }
            for md in market_data
        ])
        
        # Calculate returns for GARCH
        df['returns'] = df['close'].pct_change()
        df = df.dropna()
        
        # Fit GARCH model for volatility
        try:
            garch_metrics = self.garch_model.fit(df['returns'].values)
            self.training_history['garch_metrics'] = garch_metrics
            conditional_volatility = self.garch_model.get_conditional_volatility()
        except Exception as e:
            logger.warning(f"GARCH fitting failed, using rolling volatility: {e}")
            conditional_volatility = df['returns'].rolling(window=20).std().values
        
        # Technical indicators
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['bb_upper'] = df['close'].rolling(window=20).mean() + (df['close'].rolling(window=20).std() * 2)
        df['bb_lower'] = df['close'].rolling(window=20).mean() - (df['close'].rolling(window=20).std() * 2)
        df['rsi'] = self._calculate_rsi(df['close'], window=14)
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['volume_change'] = df['volume'].pct_change()
        
        # Add GARCH volatility as feature
        df['garch_volatility'] = conditional_volatility
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour / 24.0
        df['day_of_week'] = df['timestamp'].dt.dayofweek / 7.0
        
        # Select feature columns
        feature_columns = [
            'returns', 'garch_volatility', 'sma_5', 'sma_20', 'macd',
            'rsi', 'high_low_ratio', 'volume_ratio', 'price_change',
            'volume_change', 'hour', 'day_of_week'
        ]
        
        # Clean data
        df = df.dropna()
        
        if len(df) < self.sequence_length + 1:
            raise ValueError("Insufficient data after cleaning")
        
        # Prepare features and targets
        features = df[feature_columns].values
        targets = df['close'].values
        
        self.feature_names = feature_columns
        
        return features, targets
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:(i + self.sequence_length)])
            y.append(targets[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    @track_model_metrics("garch_lstm_hybrid")
    async def fit(
        self,
        market_data: List[MarketData],
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10
    ) -> Dict[str, Any]:
        """Fit the hybrid GARCH-LSTM model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for LSTM modeling")
        
        logger.info(f"Training GARCH-LSTM hybrid model with {len(market_data)} data points")
        
        try:
            # Prepare features
            features, targets = self._prepare_features(market_data)
            
            # Scale features and targets
            if self.feature_scaler:
                features = self.feature_scaler.fit_transform(features)
            if self.price_scaler:
                targets = self.price_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
            
            # Create sequences
            X, y = self._create_sequences(features, targets)
            
            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Convert to tensors
            X_train = torch.FloatTensor(X_train).to(self.device)
            y_train = torch.FloatTensor(y_train).to(self.device)
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device)
            
            # Initialize LSTM model
            input_size = X_train.shape[2]
            self.lstm_model = LSTMModel(
                input_size=input_size,
                output_size=1,
                **self.lstm_params
            ).to(self.device)
            
            # Loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=False
            )
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                self.lstm_model.train()
                train_loss = 0.0
                
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = self.lstm_model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                self.lstm_model.eval()
                with torch.no_grad():
                    val_outputs = self.lstm_model(X_val).squeeze()
                    val_loss = criterion(val_outputs, y_val).item()
                
                # Update learning rate
                scheduler.step(val_loss)
                
                # Record metrics
                avg_train_loss = train_loss / (len(X_train) // batch_size + 1)
                self.training_history['train_loss'].append(avg_train_loss)
                self.training_history['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    torch.save(self.lstm_model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Load best model
            if os.path.exists('best_model.pth'):
                self.lstm_model.load_state_dict(torch.load('best_model.pth'))
                os.remove('best_model.pth')
            
            self.is_fitted = True
            
            # Calculate final metrics
            self.lstm_model.eval()
            with torch.no_grad():
                train_pred = self.lstm_model(X_train).squeeze().cpu().numpy()
                val_pred = self.lstm_model(X_val).squeeze().cpu().numpy()
            
            # Convert back to original scale
            if self.price_scaler:
                train_pred = self.price_scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
                val_pred = self.price_scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()
                y_train_orig = self.price_scaler.inverse_transform(y_train.cpu().numpy().reshape(-1, 1)).flatten()
                y_val_orig = self.price_scaler.inverse_transform(y_val.cpu().numpy().reshape(-1, 1)).flatten()
            else:
                y_train_orig = y_train.cpu().numpy()
                y_val_orig = y_val.cpu().numpy()
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train_orig, train_pred) if SKLEARN_AVAILABLE else 0
            val_mse = mean_squared_error(y_val_orig, val_pred) if SKLEARN_AVAILABLE else 0
            train_mae = mean_absolute_error(y_train_orig, train_pred) if SKLEARN_AVAILABLE else 0
            val_mae = mean_absolute_error(y_val_orig, val_pred) if SKLEARN_AVAILABLE else 0
            
            training_metrics = {
                'train_mse': train_mse,
                'val_mse': val_mse,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'best_val_loss': best_val_loss,
                'epochs_trained': epoch + 1,
                'garch_aic': self.training_history['garch_metrics'].get('aic', 0),
                'garch_bic': self.training_history['garch_metrics'].get('bic', 0),
                'final_lr': scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else self.learning_rate
            }
            
            logger.info(f"Model training completed. Val MSE: {val_mse:.6f}, Val MAE: {val_mae:.6f}")
            
            return training_metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    async def predict(
        self,
        market_data: List[MarketData],
        forecast_horizon: int = 1
    ) -> Dict[str, Any]:
        """Generate predictions using the hybrid model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Prepare features for the most recent sequence
            features, targets = self._prepare_features(market_data)
            
            if len(features) < self.sequence_length:
                raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
            
            # Scale features
            if self.feature_scaler:
                features = self.feature_scaler.transform(features)
            
            # Get the last sequence
            last_sequence = features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Convert to tensor
            X = torch.FloatTensor(last_sequence).to(self.device)
            
            # Generate price prediction
            self.lstm_model.eval()
            with torch.no_grad():
                price_pred = self.lstm_model(X).cpu().numpy()
            
            # Scale back price prediction
            if self.price_scaler:
                price_pred = self.price_scaler.inverse_transform(price_pred.reshape(-1, 1)).flatten()
            
            # Generate volatility prediction using GARCH
            volatility_pred = self.garch_model.predict_volatility(horizon=forecast_horizon)
            
            # Calculate confidence intervals using volatility
            confidence_95 = price_pred[-1] + np.array([-1.96, 1.96]) * volatility_pred[-1]
            confidence_68 = price_pred[-1] + np.array([-1.0, 1.0]) * volatility_pred[-1]
            
            return {
                'price_prediction': float(price_pred[-1]),
                'volatility_prediction': float(volatility_pred[-1]),
                'confidence_intervals': {
                    '95%': confidence_95.tolist(),
                    '68%': confidence_68.tolist()
                },
                'prediction_timestamp': datetime.utcnow(),
                'model_components': {
                    'garch_volatility': float(volatility_pred[-1]),
                    'lstm_price': float(price_pred[-1])
                },
                'metadata': {
                    'sequence_length': self.sequence_length,
                    'feature_count': len(self.feature_names),
                    'features_used': self.feature_names
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_type': 'GARCH-LSTM Hybrid',
            'is_fitted': self.is_fitted,
            'sequence_length': self.sequence_length,
            'lstm_params': self.lstm_params,
            'garch_params': {'p': self.garch_model.p, 'q': self.garch_model.q},
            'training_history': self.training_history,
            'feature_names': self.feature_names,
            'device': self.device
        }


# Model factory and management functions
async def create_garch_lstm_model(
    market_data: List[MarketData],
    model_name: str = "garch_lstm_hybrid",
    version: str = "1.0.0",
    **kwargs
) -> GARCHLSTMHybridModel:
    """Create and train a new GARCH-LSTM hybrid model."""
    
    # Initialize model
    model = GARCHLSTMHybridModel(**kwargs)
    
    # Train model
    training_metrics = await model.fit(market_data)
    
    # Register model in registry
    registry = get_model_registry()
    
    metadata = ModelMetadata(
        model_id=f"{model_name}_v{version}",
        name=model_name,
        version=version,
        model_type=ModelType.PRICE_PREDICTION,
        status=ModelStatus.DEVELOPMENT,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        author="garch_lstm_service",
        description="Hybrid GARCH-LSTM model for price and volatility prediction",
        tags={
            'model_class': 'GARCHLSTMHybridModel',
            'framework': 'pytorch_arch'
        },
        
        # Performance metrics from training
        accuracy=1.0 - training_metrics['val_mse'],  # Converted metric
        precision=training_metrics['val_mae'],
        
        # Model configuration
        hyperparameters={
            'sequence_length': model.sequence_length,
            'garch_p': model.garch_model.p,
            'garch_q': model.garch_model.q,
            **model.lstm_params,
            'learning_rate': model.learning_rate
        },
        features=model.feature_names,
        training_duration_seconds=training_metrics.get('training_time', 0),
        
        # Resource requirements
        resource_requirements={
            'memory_mb': 512,
            'cpu_cores': 2,
            'gpu_required': False
        }
    )
    
    # Save model to registry
    model_key = registry.register_model(model, metadata)
    logger.info(f"GARCH-LSTM model registered as {model_key}")
    
    return model