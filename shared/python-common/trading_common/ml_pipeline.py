#!/usr/bin/env python3
"""ML Training & Evaluation Pipeline (reconstructed)

Stable version after correcting indentation/scope issues and integrating:
- Extended model registry (stateful) via model_registry
- Reproducibility hashing (feature graph, dataset, config, git commit)

Follow-up tasks (not yet implemented here):
- Drift detection & reporting
- Promotion policy externalization & endpoints
- Feature graph validation on inference
- Extended test coverage
"""
from __future__ import annotations

import logging
import pickle
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings("ignore")

# Third‑party imports guarded (editor environment may lack deps during static parsing)
try:  # pragma: no cover - environment dependent
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score  # type: ignore
    from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score  # type: ignore
    from sklearn.preprocessing import StandardScaler, RobustScaler  # type: ignore
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  # type: ignore
except ImportError:  # pragma: no cover
    pd = None  # type: ignore
    np = None  # type: ignore
    TimeSeriesSplit = cross_val_score = mean_squared_error = mean_absolute_error = accuracy_score = None  # type: ignore
    StandardScaler = RobustScaler = RandomForestRegressor = RandomForestClassifier = None  # type: ignore

from .feature_store import get_feature_store
from .model_registry import get_model_registry, RegistryEntry, ModelState
from . import registry_utils
from .feature_graph import FeatureGraphBuilder, validate_feature_graph

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    RANKER = "ranker"
    ANOMALY_DETECTOR = "anomaly_detector"


class ValidationStrategy(str, Enum):
    TIME_SERIES_SPLIT = "time_series_split"
    WALK_FORWARD = "walk_forward"
    PURGED_CV = "purged_cv"


@dataclass
class TrainingConfig:
    model_name: str
    model_type: ModelType
    feature_names: List[str]
    target_variable: str
    train_start: datetime
    train_end: datetime
    validation_strategy: ValidationStrategy = ValidationStrategy.TIME_SERIES_SPLIT
    test_size: float = 0.2
    n_splits: int = 5
    scaling_method: str = "standard"
    feature_selection: bool = True
    max_features: Optional[int] = None
    model_params: Dict[str, Any] = field(default_factory=dict)
    lookback_window: int = 252
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.02
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"


@dataclass
class ModelMetrics:
    train_score: float
    validation_score: float
    test_score: float
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    calmar_ratio: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    var_95: Optional[float] = None
    expected_shortfall: Optional[float] = None
    beta: Optional[float] = None
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    strategy_name: str
    start_date: datetime
    end_date: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    var_95: float
    expected_shortfall: float
    beta: float
    # Using Any to avoid lint issues when pandas not present at analysis time
    daily_returns: Any
    equity_curve: Any
    drawdown_curve: Any
    trades: List[Dict[str, Any]] = field(default_factory=list)

    def summary_stats(self) -> Dict[str, float]:
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
            "Total Trades": self.total_trades,
        }


class BaseMLModel(ABC):
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.training_metrics: Optional[ModelMetrics] = None
        self.selected_features: Optional[List[str]] = None

    @abstractmethod
    async def create_model(self) -> Any:  # pragma: no cover
        pass

    @abstractmethod
    async def train(self, X: Any, y: Any) -> ModelMetrics:  # pragma: no cover
        pass

    @abstractmethod
    async def predict(self, X: Any) -> Any:  # pragma: no cover
        pass

    async def preprocess_features(self, X: Any, fit: bool = True) -> Any:
        Xp = X.copy()
        Xp = Xp.fillna(Xp.median())
        if self.config.scaling_method != "none" and StandardScaler is not None:
            if fit or self.scaler is None:
                if self.config.scaling_method == "standard":
                    self.scaler = StandardScaler()
                elif self.config.scaling_method == "robust":
                    self.scaler = RobustScaler()
                else:
                    logger.warning("Unknown scaling method: %s", self.config.scaling_method)
                    return Xp
                Xp = pd.DataFrame(self.scaler.fit_transform(Xp), columns=Xp.columns, index=Xp.index)
            else:
                Xp = pd.DataFrame(self.scaler.transform(Xp), columns=Xp.columns, index=Xp.index)
        if self.config.feature_selection and self.config.max_features:
            if fit and len(Xp.columns) > self.config.max_features:
                variances = Xp.var()
                sel = variances.nlargest(self.config.max_features).index
                Xp = Xp[sel]
                self.selected_features = list(sel)
            elif self.selected_features:
                Xp = Xp[self.selected_features]
        return Xp

    async def save_model(self, filepath: str):
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "config": self.config,
            "training_metrics": self.training_metrics,
            "selected_features": self.selected_features,
            "is_trained": self.is_trained,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        logger.info("Model saved to %s", filepath)

    async def load_model(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.training_metrics = data["training_metrics"]
        self.is_trained = data["is_trained"]
        if isinstance(data.get("config"), TrainingConfig):
            self.config = data["config"]
        self.selected_features = data.get("selected_features")
        logger.info("Model loaded from %s", filepath)


class RandomForestModel(BaseMLModel):
    async def create_model(self):
        if RandomForestRegressor is None:
            raise RuntimeError("scikit-learn not available in environment")
        defaults = {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1}
        params = {**defaults, **self.config.model_params}
        if self.config.model_type == ModelType.REGRESSOR:
            self.model = RandomForestRegressor(**params)
        elif self.config.model_type == ModelType.CLASSIFIER:
            self.model = RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        return self.model

    async def train(self, X: Any, y: Any) -> ModelMetrics:
        if self.model is None:
            await self.create_model()
        Xp = await self.preprocess_features(X, fit=True)
        if self.config.validation_strategy == ValidationStrategy.TIME_SERIES_SPLIT and TimeSeriesSplit is not None:
            tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
            scoring = 'neg_mean_squared_error' if self.config.model_type == ModelType.REGRESSOR else 'accuracy'
            cv_scores = cross_val_score(self.model, Xp, y, cv=tscv, scoring=scoring) if cross_val_score else []
        else:
            cv_scores = []
        self.model.fit(Xp, y)
        self.is_trained = True
        y_pred = self.model.predict(Xp)
        if self.config.model_type == ModelType.REGRESSOR and mean_squared_error is not None:
            train_score = -mean_squared_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred) if mean_absolute_error else None
        else:
            train_score = accuracy_score(y, y_pred) if accuracy_score else 0.0
            mse = None
            mae = None
        cv_array_size = getattr(cv_scores, 'size', 0)
        metrics = ModelMetrics(
            train_score=train_score,
            validation_score=float(np.mean(cv_scores)) if cv_array_size else train_score,
            test_score=0.0,
            cv_scores=cv_scores.tolist() if cv_array_size else list(cv_scores) if isinstance(cv_scores, list) else [],
            cv_mean=float(np.mean(cv_scores)) if cv_array_size else None,
            cv_std=float(np.std(cv_scores)) if cv_array_size else None,
            mse=mse,
            mae=mae,
        )
        self.training_metrics = metrics
        logger.info("Model training completed: %s score=%.4f", self.config.model_name, metrics.train_score)
        return metrics

    async def predict(self, X: Any) -> Any:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        Xp = await self.preprocess_features(X, fit=False)
        return self.model.predict(Xp)

    def get_feature_importance(self) -> Any:  # pragma: no cover
        if not self.is_trained:
            raise ValueError("Train model first")
        feature_names = self.selected_features or self.config.feature_names
        return pd.Series(self.model.feature_importances_, index=feature_names).sort_values(ascending=False)


class MLPipeline:
    def __init__(self):
        self.feature_store = None
        self.db = None
        self.registry = None
        self.models: Dict[str, BaseMLModel] = {}

    async def initialize(self):
        self.feature_store = await get_feature_store()
        self.db = None  # database not required for core training in this context
        self.registry = await get_model_registry()
        logger.info("ML Pipeline initialized (extended registry)")

    async def _create_model_registry(self):  # legacy no-op retained
        return

    async def prepare_training_data(self, config: TrainingConfig) -> Tuple[Any, Any]:
        logger.info("Preparing training data for %s", config.model_name)
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        feature_matrix = await self.feature_store.get_feature_matrix(symbols, config.feature_names, config.train_start, config.train_end)
        if feature_matrix.empty:
            raise ValueError("No training data available")
        target = await self._create_target_variable(feature_matrix, config.target_variable)
        valid_idx = ~target.isna()
        X = feature_matrix[valid_idx]
        y = target[valid_idx]
        logger.info("Training data prepared: %d samples, %d features", len(X), len(X.columns))
        return X, y

    async def _create_target_variable(self, df: Any, target_variable: str) -> Any:
        if target_variable == "next_return":
            returns = df.groupby('entity_id')['close'].pct_change().shift(-1)
            return returns
        if target_variable == "signal_direction":
            returns = df.groupby('entity_id')['close'].pct_change().shift(-1)
            return (returns > 0).astype(int)
        return df.get(target_variable, pd.Series(dtype=float))

    async def train_model(self, config: TrainingConfig, model_class=RandomForestModel) -> Tuple[BaseMLModel, ModelMetrics]:
        logger.info("Training model: %s", config.model_name)
        X, y = await self.prepare_training_data(config)
        model = model_class(config)
        metrics = await model.train(X, y)
        model_dir = Path("data/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{config.model_name}_{config.version}.pkl"
        await model.save_model(str(model_path))
        # Build feature graph hash via deterministic DAG builder
        fg_builder = FeatureGraphBuilder(self.feature_store.feature_definitions)
        try:
            feature_graph_hash = fg_builder.compute_hash_for_features(config.feature_names)
        except ValueError as e:
            raise ValueError(f"Feature graph build failed: {e}")
        training_config_hash = registry_utils.hash_training_config(asdict(config))
        dataset_combined = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True).rename('target')], axis=1)
        dataset_hash = registry_utils.hash_dataset(dataset_combined)
        git_commit = registry_utils.git_commit_hash()
        # Build reproducibility manifest and persist alongside model artifact
        repro_manifest = registry_utils.build_repro_manifest(
            model_name=config.model_name,
            version=config.version,
            model_type=config.model_type.value,
            feature_graph_hash=feature_graph_hash,
            training_config_hash=training_config_hash,
            dataset_hash=dataset_hash,
            git_commit=git_commit,
            config=asdict(config),
            metrics=metrics.to_dict(),
            artifact_path=str(model_path),
        )
        manifest_path = model_path.with_suffix('.manifest.json')
        try:
            import json
            with open(manifest_path, 'w') as mf:
                json.dump(repro_manifest, mf, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to write reproducibility manifest: %s", e)
        entry = RegistryEntry(
            model_name=config.model_name,
            model_type=config.model_type.value,
            version=config.version,
            state=ModelState.TRAINED,
            metrics=metrics.to_dict(),
            config=asdict(config),
            artifact_path=str(model_path),
            dataset_hash=dataset_hash,
            feature_graph_hash=feature_graph_hash,
            training_config_hash=training_config_hash,
            git_commit=git_commit,
        )
        await self.registry.upsert_entry(entry)
        self.models[config.model_name] = model
        logger.info("Model training completed: %s", config.model_name)
        # Emit promotion candidate event (best-effort)
        try:  # pragma: no cover
            from trading_common.event_logging import emit_event
            emit_event(
                event_type="model.promotion.candidate",
                model_name=config.model_name,
                version=config.version,
                metrics=metrics.to_dict(),
                feature_graph_hash=feature_graph_hash,
                training_config_hash=training_config_hash,
                dataset_hash=dataset_hash,
            )
        except Exception:
            pass
        return model, metrics

    async def _register_model(self, *args, **kwargs):  # legacy compatibility
        logger.warning("Legacy _register_model invoked; ignored in favor of extended registry")
        return

    async def backtest_model(self, model: BaseMLModel, start_date: datetime, end_date: datetime, initial_capital: float = 100000.0, transaction_cost: float = 0.001) -> BacktestResult:
        logger.info("Backtesting model: %s", model.config.model_name)
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        test_data = await self.feature_store.get_feature_matrix(symbols, model.config.feature_names, start_date, end_date)
        if test_data.empty:
            raise ValueError("No test data available for backtesting")
        predictions = await model.predict(test_data[model.config.feature_names])
        test_data['predictions'] = predictions
        equity_curve: List[Dict[str, Any]] = []
        trades: List[Dict[str, Any]] = []
        current_capital = initial_capital
        positions: Dict[str, float] = {}
        threshold = 0.0 if model.config.model_type == ModelType.REGRESSOR else 0.5
        for idx, row in test_data.iterrows():
            symbol = row['entity_id']
            price = row.get('close', row.get('price', 0))
            pred = row['predictions']
            if price <= 0:
                continue
            if pred > threshold and symbol not in positions:
                position_size = current_capital * 0.2
                shares = position_size / price
                cost = shares * price * (1 + transaction_cost)
                if cost <= current_capital:
                    positions[symbol] = shares
                    current_capital -= cost
                    trades.append({'timestamp': idx, 'symbol': symbol, 'action': 'BUY', 'price': price, 'shares': shares, 'cost': cost})
            elif pred <= threshold and symbol in positions:
                shares = positions[symbol]
                proceeds = shares * price * (1 - transaction_cost)
                current_capital += proceeds
                del positions[symbol]
                trades.append({'timestamp': idx, 'symbol': symbol, 'action': 'SELL', 'price': price, 'shares': shares, 'proceeds': proceeds})
            portfolio_value = current_capital + sum(sh * price for sym, sh in positions.items() if sym == symbol)
            equity_curve.append({'timestamp': idx, 'portfolio_value': portfolio_value, 'cash': current_capital, 'positions_value': portfolio_value - current_capital})
        equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
        if equity_df.empty:
            raise ValueError("No trades executed during backtest period")
        returns = equity_df['portfolio_value'].pct_change().dropna()
        total_return = (equity_df['portfolio_value'].iloc[-1] / initial_capital) - 1
        annual_factor = 252
        volatility = returns.std() * (annual_factor ** 0.5)
        annualized_return = (1 + total_return) ** (annual_factor / max(len(returns), 1)) - 1 if len(returns) > 0 else 0.0
        sharpe_ratio = (annualized_return - model.config.risk_free_rate) / volatility if volatility > 0 else 0
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * (annual_factor ** 0.5) if len(negative_returns) > 0 else volatility
        sortino_ratio = (annualized_return - model.config.risk_free_rate) / downside_std if downside_std > 0 else 0
        running_max = equity_df['portfolio_value'].expanding().max()
        drawdown = (equity_df['portfolio_value'] - running_max) / running_max
        max_drawdown = drawdown.min()
        winning_trades = sum(1 for t in trades if t.get('proceeds', 0) > t.get('cost', float('inf')))
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
            losing_trades=len(trades) - winning_trades,
            win_rate=winning_trades / len(trades) if trades else 0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=1.0,
            var_95=returns.quantile(0.05),
            expected_shortfall=returns[returns <= returns.quantile(0.05)].mean(),
            beta=0.0,
            daily_returns=returns,
            equity_curve=equity_df['portfolio_value'],
            drawdown_curve=drawdown,
            trades=trades,
        )
        logger.info("Backtest completed. Sharpe=%.2f DD=%.2f%%", result.sharpe_ratio, result.max_drawdown * 100)
        return result

    async def get_model(self, model_name: str, version: Optional[str] = None) -> Optional[BaseMLModel]:
        if model_name in self.models:
            return self.models[model_name]
        entry = await self.registry.get_entry(model_name, version) if version else await self.registry.get_entry(model_name)
        if not entry:
            return None
        with open(entry.artifact_path, 'rb') as f:
            data = pickle.load(f)
        embedded_config = data.get('config') if isinstance(data, dict) else None
        if not isinstance(embedded_config, TrainingConfig):
            embedded_config = TrainingConfig(
                model_name=model_name,
                model_type=ModelType.REGRESSOR,
                feature_names=[],
                target_variable="",
                train_start=datetime.utcnow(),
                train_end=datetime.utcnow(),
            )
        model = RandomForestModel(embedded_config)
        await model.load_model(entry.artifact_path)
        self.models[model_name] = model
        # Feature graph validation on load to detect stale definitions
        try:
            ok = await validate_feature_graph(self.feature_store, entry.feature_graph_hash, model.config.feature_names)
            if not ok:
                logger.error("Loaded model %s has feature graph mismatch; predictions may be blocked.", model_name)
        except Exception as e:
            logger.warning("Feature graph validation raised exception: %s", e)
        return model


async def get_ml_pipeline() -> MLPipeline:
    if getattr(get_ml_pipeline, "_instance", None) is None:  # type: ignore
        inst = MLPipeline()
        await inst.initialize()
        setattr(get_ml_pipeline, "_instance", inst)  # type: ignore
    return getattr(get_ml_pipeline, "_instance")  # type: ignore