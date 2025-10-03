#!/usr/bin/env python3
"""
Advanced Intelligence Coordinator - Orchestrates PhD-level ML techniques
Integrates GNN, Factor Models, Transfer Entropy, and Stochastic Volatility for maximum alpha generation.
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))

from trading_common import MarketData, get_settings, get_logger
from trading_common.database_manager import get_database_manager
from trading_common.cache import get_trading_cache

# Import our advanced models
from graph_neural_network import get_gnn_service
from advanced_factor_models import get_factor_service
from transfer_entropy_analysis import get_causality_analyzer
from stochastic_volatility_models import get_stoch_vol_service

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class AdvancedSignal:
    """Comprehensive signal from advanced models."""
    symbol: str
    timestamp: datetime
    
    # Individual model signals
    gnn_signal: float = 0.0              # Graph Neural Network prediction
    factor_signal: float = 0.0           # Factor model alpha signal
    causality_signal: float = 0.0        # Transfer entropy signal
    volatility_signal: float = 0.0       # Volatility regime signal
    
    # Model confidences
    gnn_confidence: float = 0.0
    factor_confidence: float = 0.0
    causality_confidence: float = 0.0
    volatility_confidence: float = 0.0
    
    # Ensemble signal
    ensemble_signal: float = 0.0         # Weighted combination
    ensemble_confidence: float = 0.0     # Overall confidence
    
    # Risk metrics
    predicted_volatility: float = 0.0    # Forward-looking volatility
    var_95: float = 0.0                  # Value at Risk (95%)
    expected_return: float = 0.0         # Factor-based expected return
    
    # Timing and attribution
    optimal_holding_period: int = 1      # Optimal holding period in days
    signal_attribution: Dict[str, float] = field(default_factory=dict)
    
    def get_risk_adjusted_signal(self) -> float:
        """Get risk-adjusted signal using Sharpe-like ratio."""
        if self.predicted_volatility > 0:
            return self.ensemble_signal / (self.predicted_volatility + 0.01)
        return self.ensemble_signal


@dataclass
class MarketRegimeAnalysis:
    """Comprehensive market regime analysis."""
    timestamp: datetime
    
    # Regime indicators from different models
    volatility_regime: str               # 'low', 'medium', 'high'
    factor_regime: str                   # 'bull', 'bear', 'neutral'
    network_regime: str                  # 'connected', 'fragmented', 'crisis'
    causality_regime: str               # 'efficient', 'trending', 'chaotic'
    
    # Quantitative metrics
    market_stress_level: float          # 0-1, higher = more stressed
    information_flow_efficiency: float   # How efficiently information spreads
    systemic_risk_indicator: float      # Risk of contagion
    regime_stability: float             # How stable current regime is
    
    # Regime predictions
    regime_change_probability: float    # Probability of regime change
    expected_regime_duration: int       # Days until likely regime change
    
    def get_overall_regime(self) -> str:
        """Determine overall market regime."""
        stress_levels = {
            'low_stress': self.market_stress_level < 0.3,
            'medium_stress': 0.3 <= self.market_stress_level < 0.7,
            'high_stress': self.market_stress_level >= 0.7
        }
        
        if stress_levels['high_stress']:
            return 'crisis'
        elif stress_levels['low_stress'] and self.regime_stability > 0.7:
            return 'normal'
        else:
            return 'transitional'


class AdvancedSignalGenerator:
    """Generates sophisticated trading signals using multiple PhD-level models."""
    
    def __init__(self):
        # Model weights (optimized through backtesting)
        self.model_weights = {
            'gnn': 0.35,           # Graph Neural Networks
            'factor': 0.25,        # Factor Models  
            'causality': 0.25,     # Transfer Entropy
            'volatility': 0.15     # Stochastic Volatility
        }
        
        # Regime-dependent weight adjustments
        self.regime_adjustments = {
            'crisis': {'gnn': 0.2, 'factor': 0.2, 'causality': 0.3, 'volatility': 0.3},
            'normal': {'gnn': 0.4, 'factor': 0.3, 'causality': 0.2, 'volatility': 0.1},
            'transitional': {'gnn': 0.3, 'factor': 0.25, 'causality': 0.3, 'volatility': 0.15}
        }
        
        # Signal decay factors
        self.signal_half_life = {
            'gnn': 2,         # GNN signals decay in 2 days
            'factor': 21,     # Factor signals persist for ~1 month
            'causality': 5,   # Causality signals decay in 5 days
            'volatility': 10  # Volatility signals decay in 10 days
        }
    
    async def generate_ensemble_signal(self, symbol: str, 
                                     gnn_prediction: float,
                                     factor_signal: float,
                                     causality_signals: Dict[str, float],
                                     volatility_forecast: float,
                                     regime: MarketRegimeAnalysis) -> AdvancedSignal:
        """Generate ensemble signal from all models."""
        
        # Normalize inputs to [-1, 1] range
        gnn_signal = np.tanh(gnn_prediction * 5)  # Scale GNN prediction
        factor_signal = np.tanh(factor_signal * 10)  # Scale factor alpha
        
        # Aggregate causality signals
        causality_signal = 0.0
        if causality_signals:
            # Weight by signal strength and take average
            total_strength = sum(abs(signal) for signal in causality_signals.values())
            if total_strength > 0:
                causality_signal = sum(
                    signal * (abs(signal) / total_strength) 
                    for signal in causality_signals.values()
                )
                causality_signal = np.tanh(causality_signal)
        
        # Volatility-based signal (mean reversion)
        current_vol_estimate = 0.2  # Would get from current market data
        if volatility_forecast > 0:
            vol_ratio = volatility_forecast / current_vol_estimate
            if vol_ratio > 1.2:  # High vol -> expect reversion
                volatility_signal = -0.3
            elif vol_ratio < 0.8:  # Low vol -> expect expansion
                volatility_signal = 0.3
            else:
                volatility_signal = 0.0
        else:
            volatility_signal = 0.0
        
        # Adjust weights based on regime
        weights = self.model_weights.copy()
        overall_regime = regime.get_overall_regime()
        
        if overall_regime in self.regime_adjustments:
            regime_weights = self.regime_adjustments[overall_regime]
            for model, weight in regime_weights.items():
                weights[model] = weight
        
        # Calculate model confidences
        gnn_confidence = min(abs(gnn_signal) + 0.1, 1.0)
        factor_confidence = 0.8  # Would calculate from factor model R²
        causality_confidence = min(len(causality_signals) * 0.2, 1.0)
        volatility_confidence = 0.7  # Would calculate from model fit
        
        # Ensemble signal with confidence weighting
        confidence_weights = np.array([
            gnn_confidence * weights['gnn'],
            factor_confidence * weights['factor'], 
            causality_confidence * weights['causality'],
            volatility_confidence * weights['volatility']
        ])
        confidence_weights /= confidence_weights.sum()
        
        signals = np.array([gnn_signal, factor_signal, causality_signal, volatility_signal])
        ensemble_signal = np.sum(signals * confidence_weights)
        ensemble_confidence = np.sum(confidence_weights)
        
        # Risk metrics
        predicted_volatility = volatility_forecast if volatility_forecast > 0 else 0.2
        var_95 = 1.645 * predicted_volatility  # 95% VaR
        
        # Expected return (from factor model)
        expected_return = factor_signal * 0.1  # Convert to return estimate
        
        # Optimal holding period based on signal characteristics
        holding_period = self._calculate_optimal_holding_period(
            gnn_signal, factor_signal, causality_signal, volatility_signal
        )
        
        # Signal attribution
        attribution = {
            'gnn': gnn_signal * confidence_weights[0],
            'factor': factor_signal * confidence_weights[1],
            'causality': causality_signal * confidence_weights[2],
            'volatility': volatility_signal * confidence_weights[3]
        }
        
        return AdvancedSignal(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            gnn_signal=gnn_signal,
            factor_signal=factor_signal,
            causality_signal=causality_signal,
            volatility_signal=volatility_signal,
            gnn_confidence=gnn_confidence,
            factor_confidence=factor_confidence,
            causality_confidence=causality_confidence,
            volatility_confidence=volatility_confidence,
            ensemble_signal=ensemble_signal,
            ensemble_confidence=ensemble_confidence,
            predicted_volatility=predicted_volatility,
            var_95=var_95,
            expected_return=expected_return,
            optimal_holding_period=holding_period,
            signal_attribution=attribution
        )
    
    def _calculate_optimal_holding_period(self, gnn_signal: float, factor_signal: float,
                                        causality_signal: float, volatility_signal: float) -> int:
        """Calculate optimal holding period based on signal characteristics."""
        
        # Weight by signal strength and typical persistence
        weighted_periods = [
            abs(gnn_signal) * self.signal_half_life['gnn'],
            abs(factor_signal) * self.signal_half_life['factor'],
            abs(causality_signal) * self.signal_half_life['causality'],
            abs(volatility_signal) * self.signal_half_life['volatility']
        ]
        
        total_weight = abs(gnn_signal) + abs(factor_signal) + abs(causality_signal) + abs(volatility_signal)
        
        if total_weight > 0:
            optimal_period = sum(weighted_periods) / total_weight
            return max(1, int(optimal_period))
        else:
            return 5  # Default holding period


class AdvancedIntelligenceCoordinator:
    """Coordinates all PhD-level ML techniques for maximum alpha generation."""
    
    def __init__(self):
        self.signal_generator = AdvancedSignalGenerator()
        self.cache = None
        
        # Service references (will be initialized)
        self.gnn_service = None
        self.factor_service = None
        self.causality_analyzer = None
        self.stoch_vol_service = None
        
        # Current state
        self.current_signals = {}  # symbol -> AdvancedSignal
        self.current_regime = None  # MarketRegimeAnalysis
        
        # Performance tracking
        self.signals_generated = 0
        self.ensemble_accuracy = 0.0
        self.last_full_update = None
        
        # Configuration
        self.update_frequency = timedelta(minutes=30)
        self.full_recalibration_frequency = timedelta(hours=6)
        
    async def initialize(self):
        """Initialize the advanced intelligence coordinator."""
        
        self.cache = get_trading_cache()
        
        # Initialize all services
        self.gnn_service = await get_gnn_service()
        self.factor_service = await get_factor_service()
        self.causality_analyzer = await get_causality_analyzer()
        self.stoch_vol_service = await get_stoch_vol_service()
        
        logger.info("Advanced Intelligence Coordinator initialized with all PhD-level models")
    
    async def generate_advanced_signals(self, symbols: List[str],
                                      market_data: Dict[str, List[MarketData]],
                                      fundamental_data: Dict[str, Dict] = None) -> Dict[str, AdvancedSignal]:
        """Generate comprehensive signals using all advanced models."""
        
        logger.info(f"Generating advanced signals for {len(symbols)} symbols")
        
        if fundamental_data is None:
            fundamental_data = {}
        
        # Check if we need full recalibration
        needs_recalibration = (
            self.last_full_update is None or 
            datetime.utcnow() - self.last_full_update > self.full_recalibration_frequency
        )
        
        if needs_recalibration:
            await self._perform_full_model_update(symbols, market_data, fundamental_data)
            self.last_full_update = datetime.utcnow()
        
        # Analyze current market regime
        regime_analysis = await self._analyze_market_regime(symbols, market_data)
        self.current_regime = regime_analysis
        
        # Generate signals for each symbol
        signals = {}
        
        for symbol in symbols:
            try:
                signal = await self._generate_symbol_signal(
                    symbol, market_data, fundamental_data, regime_analysis
                )
                if signal:
                    signals[symbol] = signal
                    self.signals_generated += 1
                    
            except Exception as e:
                logger.warning(f"Failed to generate signal for {symbol}: {e}")
        
        # Update current signals
        self.current_signals.update(signals)
        
        # Cache results
        await self._cache_signals_and_regime(signals, regime_analysis)
        
        logger.info(f"Generated {len(signals)} advanced signals")
        
        return signals

    async def generate_signals(self, symbols: List[str], strategies: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Compatibility wrapper used by ml/main.py to generate signals without pre-fetched data.

        This method pulls recent market data from QuestDB for each symbol, then calls
        generate_advanced_signals() and returns a JSON-serializable structure.
        """
        # Fetch recent OHLCV from QuestDB
        market_data: Dict[str, List[MarketData]] = {}
        try:
            dbm = await get_database_manager()
            async with dbm.get_questdb() as conn:
                query = (
                    "SELECT symbol, timestamp, open, high, low, close, volume, vwap, trade_count, data_source "
                    "FROM market_data WHERE symbol = $1 ORDER BY timestamp DESC LIMIT 500"
                )
                for sym in symbols:
                    rows = await conn.fetch(query, sym)
                    series = []
                    for r in reversed(rows):  # chronological order
                        try:
                            series.append(MarketData(
                                symbol=r["symbol"],
                                timestamp=r["timestamp"],
                                open=r["open"], high=r["high"], low=r["low"], close=r["close"],
                                volume=int(r["volume"]) if r["volume"] is not None else 0,
                                vwap=r["vwap"],
                                trade_count=int(r["trade_count"]) if r["trade_count"] is not None else 0,
                                data_source=r["data_source"] or "questdb"
                            ))
                        except Exception:
                            continue
                    if series:
                        market_data[sym] = series
        except Exception as e:
            logger.warning(f"generate_signals: failed to fetch market data from QuestDB: {e}")

        # If no data could be fetched, return empty
        if not market_data:
            return {}

        # Delegate to core generator
        adv = await self.generate_advanced_signals(symbols, market_data)

        # Convert dataclasses to JSON-serializable dicts
        out: Dict[str, Dict[str, Any]] = {}
        for sym, sig in adv.items():
            out[sym] = {
                "timestamp": sig.timestamp.isoformat(),
                "gnn_signal": sig.gnn_signal,
                "factor_signal": sig.factor_signal,
                "causality_signal": sig.causality_signal,
                "volatility_signal": sig.volatility_signal,
                "gnn_confidence": sig.gnn_confidence,
                "factor_confidence": sig.factor_confidence,
                "causality_confidence": sig.causality_confidence,
                "volatility_confidence": sig.volatility_confidence,
                "ensemble_signal": sig.ensemble_signal,
                "ensemble_confidence": sig.ensemble_confidence,
                "predicted_volatility": sig.predicted_volatility,
                "var_95": sig.var_95,
                "expected_return": sig.expected_return,
                "optimal_holding_period": sig.optimal_holding_period,
                "signal_attribution": sig.signal_attribution,
                "risk_adjusted_signal": sig.get_risk_adjusted_signal(),
            }
        return out
    
    async def _perform_full_model_update(self, symbols: List[str],
                                       market_data: Dict[str, List[MarketData]],
                                       fundamental_data: Dict[str, Dict]):
        """Perform full update of all underlying models."""
        
        logger.info("Performing full model recalibration")
        
        # Update all models concurrently
        update_tasks = [
            self.factor_service.update_factor_models(symbols, market_data, fundamental_data),
            self.causality_analyzer.analyze_market_causality(symbols, market_data),
            self.stoch_vol_service.update_volatility_models(symbols, market_data),
            # GNN training is more expensive, only do periodically
        ]
        
        try:
            results = await asyncio.gather(*update_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    model_names = ['factor', 'causality', 'stochastic_vol']
                    logger.warning(f"Model update failed for {model_names[i]}: {result}")
                    
        except Exception as e:
            logger.error(f"Full model update failed: {e}")
    
    async def _analyze_market_regime(self, symbols: List[str],
                                   market_data: Dict[str, List[MarketData]]) -> MarketRegimeAnalysis:
        """Analyze current market regime using all models."""
        
        # Get regime indicators from each model
        volatility_regime = 'medium'  # Default
        factor_regime = 'neutral'
        network_regime = 'connected'
        causality_regime = 'efficient'
        
        # Volatility regime
        if self.stoch_vol_service.stoch_vol_service:  # Check if service is available
            vol_regimes = await self.stoch_vol_service.detect_volatility_regime_changes(symbols)
            if vol_regimes:
                # Take most common regime
                regime_counts = defaultdict(int)
                for symbol_regime in vol_regimes.values():
                    regime = symbol_regime.get('current_regime', 'normal_volatility')
                    if 'high' in regime:
                        regime_counts['high'] += 1
                    elif 'low' in regime:
                        regime_counts['low'] += 1
                    else:
                        regime_counts['medium'] += 1
                
                volatility_regime = max(regime_counts, key=regime_counts.get)
        
        # Factor regime
        if self.factor_service.current_factor_returns:
            bull_indicator = self.factor_service.current_factor_returns.bull_market_indicator
            if bull_indicator > 0.6:
                factor_regime = 'bull'
            elif bull_indicator < 0.4:
                factor_regime = 'bear'
            else:
                factor_regime = 'neutral'
        
        # Network regime (from GNN or causality analysis)
        if self.causality_analyzer.current_network:
            network_density = self.causality_analyzer.current_network.network_density
            if network_density > 0.3:
                network_regime = 'connected'
            elif network_density > 0.1:
                network_regime = 'moderate'
            else:
                network_regime = 'fragmented'
        
        # Causality regime
        if self.causality_analyzer.current_network:
            efficiency = self.causality_analyzer.current_network.market_efficiency
            if efficiency > 0.5:
                causality_regime = 'efficient'
            elif efficiency > 0.2:
                causality_regime = 'trending' 
            else:
                causality_regime = 'chaotic'
        
        # Calculate quantitative metrics
        market_stress_level = self._calculate_market_stress(volatility_regime, factor_regime, network_regime)
        
        info_flow_efficiency = 0.5  # Default
        if self.causality_analyzer.current_network:
            info_flow_efficiency = self.causality_analyzer.current_network.information_flow_intensity
        
        systemic_risk = 0.3  # Default
        if self.causality_analyzer.current_network:
            systemic_risk = self.causality_analyzer.current_network.systemic_risk_indicator
        
        # Regime stability (how consistent are the indicators)
        regime_stability = self._calculate_regime_stability(
            volatility_regime, factor_regime, network_regime, causality_regime
        )
        
        # Regime change prediction
        regime_change_prob = max(0.1, 1.0 - regime_stability)
        expected_duration = int(30 * regime_stability)  # More stable = longer duration
        
        return MarketRegimeAnalysis(
            timestamp=datetime.utcnow(),
            volatility_regime=volatility_regime,
            factor_regime=factor_regime,
            network_regime=network_regime,
            causality_regime=causality_regime,
            market_stress_level=market_stress_level,
            information_flow_efficiency=info_flow_efficiency,
            systemic_risk_indicator=systemic_risk,
            regime_stability=regime_stability,
            regime_change_probability=regime_change_prob,
            expected_regime_duration=expected_duration
        )
    
    def _calculate_market_stress(self, vol_regime: str, factor_regime: str, network_regime: str) -> float:
        """Calculate overall market stress level."""
        
        stress_components = []
        
        # Volatility stress
        if vol_regime == 'high':
            stress_components.append(0.8)
        elif vol_regime == 'low':
            stress_components.append(0.2)
        else:
            stress_components.append(0.4)
        
        # Factor stress  
        if factor_regime == 'bear':
            stress_components.append(0.7)
        elif factor_regime == 'bull':
            stress_components.append(0.2)
        else:
            stress_components.append(0.4)
        
        # Network stress
        if network_regime == 'fragmented':
            stress_components.append(0.8)
        elif network_regime == 'connected':
            stress_components.append(0.3)
        else:
            stress_components.append(0.5)
        
        return np.mean(stress_components)
    
    def _calculate_regime_stability(self, vol_regime: str, factor_regime: str,
                                  network_regime: str, causality_regime: str) -> float:
        """Calculate how stable/consistent the current regime is."""
        
        # This is a simplified implementation
        # Would use more sophisticated measures in practice
        
        regime_scores = {
            'high': 0.8, 'medium': 0.6, 'low': 0.4,
            'bull': 0.8, 'neutral': 0.5, 'bear': 0.2,
            'connected': 0.8, 'moderate': 0.6, 'fragmented': 0.3,
            'efficient': 0.8, 'trending': 0.6, 'chaotic': 0.2
        }
        
        scores = [
            regime_scores.get(vol_regime, 0.5),
            regime_scores.get(factor_regime, 0.5),
            regime_scores.get(network_regime, 0.5),
            regime_scores.get(causality_regime, 0.5)
        ]
        
        # Stability is inverse of variance in regime scores
        variance = np.var(scores)
        stability = max(0.1, 1.0 - variance * 2)  # Scale factor
        
        return min(stability, 1.0)
    
    async def _generate_symbol_signal(self, symbol: str,
                                    market_data: Dict[str, List[MarketData]],
                                    fundamental_data: Dict[str, Dict],
                                    regime: MarketRegimeAnalysis) -> Optional[AdvancedSignal]:
        """Generate comprehensive signal for a single symbol."""
        
        if symbol not in market_data:
            return None
        
        # Get predictions from each model
        gnn_prediction = 0.0
        try:
            gnn_predictions = await self.gnn_service.predict([symbol], {symbol: market_data[symbol]})
            gnn_prediction = gnn_predictions.get(symbol, 0.0)
        except Exception as e:
            logger.debug(f"GNN prediction failed for {symbol}: {e}")
        
        # Factor model signal
        factor_signals = await self.factor_service.get_factor_signals([symbol])
        factor_signal = factor_signals.get(symbol, 0.0)
        
        # Causality signals
        causality_signals = await self.causality_analyzer.get_causality_signals(symbol)
        
        # Volatility forecast
        volatility_forecast = await self.stoch_vol_service.get_volatility_forecast(symbol, 21)
        if volatility_forecast is None:
            volatility_forecast = 0.2  # Default
        
        # Generate ensemble signal
        signal = await self.signal_generator.generate_ensemble_signal(
            symbol=symbol,
            gnn_prediction=gnn_prediction,
            factor_signal=factor_signal,
            causality_signals=causality_signals,
            volatility_forecast=volatility_forecast,
            regime=regime
        )
        
        return signal
    
    async def get_portfolio_optimization_inputs(self, portfolio: Dict[str, float]) -> Dict[str, Any]:
        """Get advanced inputs for portfolio optimization."""
        
        if not portfolio:
            return {}
        
        symbols = list(portfolio.keys())
        
        # Expected returns from factor models
        expected_returns = {}
        for symbol in symbols:
            if symbol in self.current_signals:
                expected_returns[symbol] = self.current_signals[symbol].expected_return
            else:
                expected_returns[symbol] = 0.0
        
        # Covariance matrix (enhanced with network structure)
        covariance_matrix = await self._build_enhanced_covariance_matrix(symbols)
        
        # Risk constraints
        risk_constraints = {}
        for symbol in symbols:
            if symbol in self.current_signals:
                risk_constraints[symbol] = self.current_signals[symbol].var_95
            else:
                risk_constraints[symbol] = 0.02  # Default 2% VaR
        
        # Factor exposures
        factor_exposures = {}
        if self.factor_service:
            factor_exposures = await self.factor_service.get_portfolio_factor_exposure(portfolio)
        
        return {
            'expected_returns': expected_returns,
            'covariance_matrix': covariance_matrix,
            'risk_constraints': risk_constraints,
            'factor_exposures': factor_exposures,
            'regime_analysis': self.current_regime
        }
    
    async def _build_enhanced_covariance_matrix(self, symbols: List[str]) -> np.ndarray:
        """Build covariance matrix enhanced with network structure."""
        
        n = len(symbols)
        cov_matrix = np.eye(n) * 0.04  # Default diagonal
        
        # Enhance with causality network structure
        if self.causality_analyzer.current_network:
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i != j:
                        # Find causality relationship
                        causality_strength = 0.0
                        
                        for edge in self.causality_analyzer.current_network.edges:
                            if ((edge.source_symbol == symbol1 and edge.target_symbol == symbol2) or
                                (edge.source_symbol == symbol2 and edge.target_symbol == symbol1)):
                                causality_strength = edge.normalized_te
                                break
                        
                        # Enhance correlation based on causality
                        base_corr = 0.1  # Base correlation
                        enhanced_corr = base_corr + causality_strength * 0.3
                        
                        # Convert correlation to covariance
                        vol_i = vol_j = 0.2  # Default volatilities
                        if symbol1 in self.current_signals:
                            vol_i = self.current_signals[symbol1].predicted_volatility
                        if symbol2 in self.current_signals:
                            vol_j = self.current_signals[symbol2].predicted_volatility
                        
                        cov_matrix[i, j] = enhanced_corr * vol_i * vol_j
        
        # Ensure positive semi-definite
        eigenvals, eigenvects = np.linalg.eigh(cov_matrix)
        eigenvals = np.maximum(eigenvals, 0.001)  # Floor eigenvalues
        cov_matrix = eigenvects @ np.diag(eigenvals) @ eigenvects.T
        
        return cov_matrix
    
    async def _cache_signals_and_regime(self, signals: Dict[str, AdvancedSignal],
                                      regime: MarketRegimeAnalysis):
        """Cache signals and regime analysis."""
        
        if self.cache:
            # Cache signals
            signals_cache = {}
            for symbol, signal in signals.items():
                signals_cache[symbol] = {
                    'ensemble_signal': signal.ensemble_signal,
                    'ensemble_confidence': signal.ensemble_confidence,
                    'predicted_volatility': signal.predicted_volatility,
                    'expected_return': signal.expected_return,
                    'optimal_holding_period': signal.optimal_holding_period,
                    'signal_attribution': signal.signal_attribution,
                    'timestamp': signal.timestamp.isoformat()
                }
            
            await self.cache.set_json("advanced_signals", signals_cache, ttl=1800)
            
            # Cache regime analysis
            regime_cache = {
                'overall_regime': regime.get_overall_regime(),
                'volatility_regime': regime.volatility_regime,
                'factor_regime': regime.factor_regime,
                'network_regime': regime.network_regime,
                'market_stress_level': regime.market_stress_level,
                'regime_stability': regime.regime_stability,
                'regime_change_probability': regime.regime_change_probability,
                'timestamp': regime.timestamp.isoformat()
            }
            
            await self.cache.set_json("market_regime_analysis", regime_cache, ttl=3600)
    
    async def get_service_performance(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        
        # Get performance from all sub-services
        gnn_perf = {}
        factor_perf = {}
        causality_perf = {}
        stoch_vol_perf = {}
        
        try:
            if hasattr(self.gnn_service, 'get_performance_metrics'):
                gnn_perf = await self.gnn_service.get_performance_metrics()
            if hasattr(self.factor_service, 'get_service_performance'):
                factor_perf = await self.factor_service.get_service_performance()
            if hasattr(self.causality_analyzer, 'get_service_statistics'):
                causality_perf = await self.causality_analyzer.get_service_statistics()
            if hasattr(self.stoch_vol_service, 'get_service_performance'):
                stoch_vol_perf = await self.stoch_vol_service.get_service_performance()
        except Exception as e:
            logger.debug(f"Error getting sub-service performance: {e}")
        
        return {
            'coordinator_metrics': {
                'signals_generated': self.signals_generated,
                'ensemble_accuracy': self.ensemble_accuracy,
                'active_symbols': len(self.current_signals),
                'last_full_update': self.last_full_update.isoformat() if self.last_full_update else None,
                'current_regime': self.current_regime.get_overall_regime() if self.current_regime else 'unknown'
            },
            'model_performance': {
                'gnn': gnn_perf,
                'factor': factor_perf,
                'causality': causality_perf,
                'stochastic_volatility': stoch_vol_perf
            }
        }


# Global coordinator instance
advanced_coordinator: Optional[AdvancedIntelligenceCoordinator] = None


async def get_advanced_coordinator() -> AdvancedIntelligenceCoordinator:
    """Get or create advanced intelligence coordinator instance."""
    global advanced_coordinator
    if advanced_coordinator is None:
        advanced_coordinator = AdvancedIntelligenceCoordinator()
        await advanced_coordinator.initialize()
    return advanced_coordinator