#!/usr/bin/env python3
"""
ML Training Pipeline Script
Comprehensive script to train, evaluate, and deploy ML models for trading.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add parent directories to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
sys.path.append(str(parent_dir / "shared" / "python-common"))

try:
    from trading_common.feature_store import get_feature_store, compute_technical_features
    from trading_common.ml_pipeline import get_ml_pipeline, TrainingConfig, ModelType, ValidationStrategy
    from services.ml.model_serving_service import get_model_serving_service, ModelStatus
    from services.ml.performance_analytics_service import get_performance_analytics_service
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Some dependencies may be missing. Please install required packages.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLTrainingPipeline:
    """Comprehensive ML training pipeline for trading models."""
    
    def __init__(self):
        self.feature_store = None
        self.ml_pipeline = None
        self.model_serving = None
        self.performance_analytics = None
        
    async def initialize(self):
        """Initialize all pipeline components."""
        logger.info("Initializing ML Training Pipeline...")
        
        self.feature_store = await get_feature_store()
        self.ml_pipeline = await get_ml_pipeline()
        self.model_serving = await get_model_serving_service()
        self.performance_analytics = await get_performance_analytics_service()
        
        logger.info("ML Training Pipeline initialized")
    
    async def create_features(self, symbols: List[str], start_date: datetime, end_date: datetime):
        """Create and store features for training."""
        logger.info(f"Creating features for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Compute technical features
        feature_df = await compute_technical_features(symbols, start_date, end_date)
        
        logger.info(f"Created {len(feature_df)} feature records")
        return feature_df
    
    async def train_model(self, model_config: Dict[str, Any]) -> str:
        """Train a new model with given configuration."""
        logger.info(f"Training model: {model_config['model_name']}")
        
        # Create training configuration
        config = TrainingConfig(
            model_name=model_config['model_name'],
            model_type=ModelType(model_config.get('model_type', 'regressor')),
            feature_names=model_config['feature_names'],
            target_variable=model_config['target_variable'],
            train_start=datetime.fromisoformat(model_config['train_start']),
            train_end=datetime.fromisoformat(model_config['train_end']),
            validation_strategy=ValidationStrategy(model_config.get('validation_strategy', 'time_series_split')),
            test_size=model_config.get('test_size', 0.2),
            n_splits=model_config.get('n_splits', 5),
            model_params=model_config.get('model_params', {}),
            version=model_config.get('version', '1.0.0')
        )
        
        # Train model
        model, metrics = await self.ml_pipeline.train_model(config)
        
        logger.info(f"Model training completed. CV Score: {metrics.cv_mean:.4f}")
        return config.model_name
    
    async def backtest_model(self, model_name: str, start_date: datetime, end_date: datetime):
        """Backtest a trained model."""
        logger.info(f"Backtesting model: {model_name}")
        
        # Get model
        model = await self.ml_pipeline.get_model(model_name)
        if not model:
            raise ValueError(f"Model not found: {model_name}")
        
        # Run backtest
        backtest_result = await self.ml_pipeline.backtest_model(
            model, start_date, end_date, initial_capital=100000.0
        )
        
        # Analyze performance
        performance_report = await self.performance_analytics.analyze_strategy_performance(
            strategy_name=model_name,
            returns=backtest_result.daily_returns,
            benchmark_symbol="SPY"
        )
        
        logger.info(f"Backtest completed. Sharpe Ratio: {backtest_result.sharpe_ratio:.2f}")
        return backtest_result, performance_report
    
    async def deploy_model(self, model_name: str, version: str, deploy_mode: str = "shadow"):
        """Deploy a model to serving."""
        logger.info(f"Deploying model: {model_name}:{version} in {deploy_mode} mode")
        
        deploy_status = ModelStatus.SHADOW if deploy_mode == "shadow" else ModelStatus.ACTIVE
        success = await self.model_serving.deploy_model(model_name, version, deploy_status)
        
        if success:
            logger.info(f"Model deployed successfully: {model_name}:{version}")
        else:
            logger.error(f"Failed to deploy model: {model_name}:{version}")
        
        return success
    
    async def run_full_pipeline(self, config_file: str):
        """Run complete ML pipeline from configuration file."""
        logger.info(f"Running full ML pipeline with config: {config_file}")
        
        # Load configuration
        import json
        with open(config_file, 'r') as f:
            pipeline_config = json.load(f)
        
        # Step 1: Create features
        if pipeline_config.get('create_features', True):
            await self.create_features(
                symbols=pipeline_config['symbols'],
                start_date=datetime.fromisoformat(pipeline_config['feature_start_date']),
                end_date=datetime.fromisoformat(pipeline_config['feature_end_date'])
            )
        
        # Step 2: Train models
        for model_config in pipeline_config['models']:
            model_name = await self.train_model(model_config)
            
            # Step 3: Backtest model
            if pipeline_config.get('run_backtest', True):
                backtest_result, performance_report = await self.backtest_model(
                    model_name,
                    datetime.fromisoformat(pipeline_config['backtest_start_date']),
                    datetime.fromisoformat(pipeline_config['backtest_end_date'])
                )
                
                # Step 4: Deploy if performance is acceptable
                if (backtest_result.sharpe_ratio > pipeline_config.get('min_sharpe_for_deployment', 0.5) and
                    backtest_result.max_drawdown > pipeline_config.get('max_drawdown_for_deployment', -0.15)):
                    
                    await self.deploy_model(
                        model_name, 
                        model_config.get('version', '1.0.0'),
                        pipeline_config.get('deploy_mode', 'shadow')
                    )
        
        logger.info("Full ML pipeline completed")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ML Training Pipeline for AI Trading System")
    parser.add_argument("command", choices=[
        "create-features", "train", "backtest", "deploy", "full-pipeline"
    ], help="Command to execute")
    
    # Feature creation arguments
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "GOOGL", "MSFT"],
                       help="Symbols to create features for")
    parser.add_argument("--feature-start", type=str, default=(datetime.utcnow() - timedelta(days=365)).isoformat(),
                       help="Feature creation start date (ISO format)")
    parser.add_argument("--feature-end", type=str, default=datetime.utcnow().isoformat(),
                       help="Feature creation end date (ISO format)")
    
    # Training arguments
    parser.add_argument("--model-name", type=str, default="default_model",
                       help="Name of the model to train")
    parser.add_argument("--model-type", type=str, default="regressor",
                       choices=["regressor", "classifier"],
                       help="Type of model to train")
    parser.add_argument("--target-variable", type=str, default="next_return",
                       help="Target variable for training")
    parser.add_argument("--train-start", type=str, default=(datetime.utcnow() - timedelta(days=300)).isoformat(),
                       help="Training start date (ISO format)")
    parser.add_argument("--train-end", type=str, default=(datetime.utcnow() - timedelta(days=30)).isoformat(),
                       help="Training end date (ISO format)")
    
    # Backtesting arguments
    parser.add_argument("--backtest-start", type=str, default=(datetime.utcnow() - timedelta(days=90)).isoformat(),
                       help="Backtest start date (ISO format)")
    parser.add_argument("--backtest-end", type=str, default=datetime.utcnow().isoformat(),
                       help="Backtest end date (ISO format)")
    
    # Deployment arguments
    parser.add_argument("--version", type=str, default="1.0.0",
                       help="Model version")
    parser.add_argument("--deploy-mode", type=str, default="shadow",
                       choices=["shadow", "active"],
                       help="Deployment mode")
    
    # Full pipeline arguments
    parser.add_argument("--config", type=str, default="ml_pipeline_config.json",
                       help="Configuration file for full pipeline")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MLTrainingPipeline()
    await pipeline.initialize()
    
    try:
        if args.command == "create-features":
            await pipeline.create_features(
                symbols=args.symbols,
                start_date=datetime.fromisoformat(args.feature_start),
                end_date=datetime.fromisoformat(args.feature_end)
            )
        
        elif args.command == "train":
            # Create default training configuration
            model_config = {
                "model_name": args.model_name,
                "model_type": args.model_type,
                "feature_names": [
                    "sma_5", "sma_10", "sma_20", "sma_50", 
                    "rsi_14", "bb_upper_20_2", "bb_middle_20_2", "bb_lower_20_2"
                ],
                "target_variable": args.target_variable,
                "train_start": args.train_start,
                "train_end": args.train_end,
                "version": args.version
            }
            
            await pipeline.train_model(model_config)
        
        elif args.command == "backtest":
            await pipeline.backtest_model(
                model_name=args.model_name,
                start_date=datetime.fromisoformat(args.backtest_start),
                end_date=datetime.fromisoformat(args.backtest_end)
            )
        
        elif args.command == "deploy":
            await pipeline.deploy_model(
                model_name=args.model_name,
                version=args.version,
                deploy_mode=args.deploy_mode
            )
        
        elif args.command == "full-pipeline":
            # Create default config if it doesn't exist
            config_path = Path(args.config)
            if not config_path.exists():
                logger.info(f"Creating default configuration: {config_path}")
                default_config = {
                    "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
                    "feature_start_date": (datetime.utcnow() - timedelta(days=365)).isoformat(),
                    "feature_end_date": datetime.utcnow().isoformat(),
                    "create_features": True,
                    "models": [
                        {
                            "model_name": "momentum_predictor",
                            "model_type": "regressor",
                            "feature_names": [
                                "sma_5", "sma_10", "sma_20", "sma_50",
                                "rsi_14", "bb_upper_20_2", "bb_middle_20_2", "bb_lower_20_2"
                            ],
                            "target_variable": "next_return",
                            "train_start": (datetime.utcnow() - timedelta(days=300)).isoformat(),
                            "train_end": (datetime.utcnow() - timedelta(days=30)).isoformat(),
                            "version": "1.0.0",
                            "model_params": {
                                "n_estimators": 100,
                                "max_depth": 10,
                                "min_samples_split": 5
                            }
                        }
                    ],
                    "backtest_start_date": (datetime.utcnow() - timedelta(days=90)).isoformat(),
                    "backtest_end_date": datetime.utcnow().isoformat(),
                    "run_backtest": True,
                    "min_sharpe_for_deployment": 0.5,
                    "max_drawdown_for_deployment": -0.15,
                    "deploy_mode": "shadow"
                }
                
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
            
            await pipeline.run_full_pipeline(args.config)
        
        logger.info(f"Command '{args.command}' completed successfully")
        
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())