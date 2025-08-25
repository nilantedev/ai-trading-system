#!/usr/bin/env python3
"""
Market Regime Detection System - Adaptive strategy selection based on market conditions
Uses Hidden Markov Models and volatility clustering to identify market regimes.
"""

import asyncio
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
from enum import Enum
import statistics
from sklearn.mixture import GaussianMixture
from scipy import stats

from trading_common import MarketData, get_settings, get_logger
from trading_common.cache import get_trading_cache

logger = get_logger(__name__)
settings = get_settings()


class MarketRegime(Enum):
    """Market regime types."""
    LOW_VOLATILITY_BULL = "low_vol_bull"      # Low vol, positive trend
    HIGH_VOLATILITY_BULL = "high_vol_bull"    # High vol, positive trend  
    LOW_VOLATILITY_BEAR = "low_vol_bear"      # Low vol, negative trend
    HIGH_VOLATILITY_BEAR = "high_vol_bear"    # High vol, negative trend
    SIDEWAYS_LOW_VOL = "sideways_low_vol"     # Range-bound, low vol
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"   # Range-bound, high vol
    CRISIS = "crisis"                         # Extreme volatility/correlation
    RECOVERY = "recovery"                     # Post-crisis stabilization


@dataclass
class RegimeCharacteristics:
    """Characteristics of a market regime."""
    regime: MarketRegime
    volatility_level: float  # 0-1, normalized volatility
    trend_strength: float   # -1 to 1, negative=bearish, positive=bullish
    correlation_breakdown: bool  # True if correlations are breaking down
    volume_surge: float     # Volume relative to average
    sector_rotation: bool   # True if sector rotation is occurring
    probability: float      # Confidence in regime classification (0-1)
    duration_estimate: int  # Expected duration in days
    
    def get_strategy_recommendations(self) -> Dict[str, float]:
        """Get strategy allocation recommendations for this regime."""
        if self.regime == MarketRegime.LOW_VOLATILITY_BULL:
            return {
                'momentum': 0.4,
                'breakout': 0.3, 
                'mean_reversion': 0.1,
                'ma_crossover': 0.2
            }
        elif self.regime == MarketRegime.HIGH_VOLATILITY_BULL:
            return {
                'momentum': 0.5,
                'breakout': 0.3,
                'mean_reversion': 0.0,
                'ma_crossover': 0.2
            }
        elif self.regime == MarketRegime.LOW_VOLATILITY_BEAR:
            return {
                'momentum': 0.1,
                'breakout': 0.2,
                'mean_reversion': 0.4,
                'ma_crossover': 0.3
            }
        elif self.regime == MarketRegime.HIGH_VOLATILITY_BEAR:
            return {
                'momentum': 0.0,
                'breakout': 0.1,
                'mean_reversion': 0.6,
                'ma_crossover': 0.3
            }
        elif self.regime == MarketRegime.SIDEWAYS_LOW_VOL:
            return {
                'momentum': 0.1,
                'breakout': 0.2,
                'mean_reversion': 0.5,
                'ma_crossover': 0.2
            }
        elif self.regime == MarketRegime.SIDEWAYS_HIGH_VOL:
            return {
                'momentum': 0.2,
                'breakout': 0.4,
                'mean_reversion': 0.3,
                'ma_crossover': 0.1
            }
        elif self.regime == MarketRegime.CRISIS:
            return {
                'momentum': 0.0,
                'breakout': 0.0,
                'mean_reversion': 0.8,
                'ma_crossover': 0.2
            }
        else:  # RECOVERY
            return {
                'momentum': 0.3,
                'breakout': 0.4,
                'mean_reversion': 0.2,
                'ma_crossover': 0.1
            }


@dataclass
class MarketMetrics:
    """Market-wide metrics for regime detection."""
    timestamp: datetime
    market_volatility: float      # VIX or calculated volatility
    trend_strength: float         # Overall market trend strength
    sector_dispersion: float      # How spread out sector returns are
    correlation_avg: float        # Average cross-asset correlation
    volume_profile: float         # Volume vs historical average
    breadth_indicator: float      # % of stocks above moving average
    credit_spreads: float         # Credit spread widening indicator
    currency_volatility: float    # FX volatility indicator


class VolatilityClustering:
    """Detect volatility clustering using GARCH-like approach."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.returns_history = deque(maxlen=window_size)
        self.volatility_history = deque(maxlen=window_size)
        
    def update(self, return_value: float):
        """Update with new return and calculate volatility."""
        self.returns_history.append(return_value)
        
        if len(self.returns_history) >= 5:
            # Calculate realized volatility over last 5 periods
            recent_returns = list(self.returns_history)[-5:]
            volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
            self.volatility_history.append(volatility)
    
    def detect_volatility_regime(self) -> Tuple[str, float]:
        """Detect current volatility regime."""
        if len(self.volatility_history) < 20:
            return "normal", 0.5
            
        volatilities = np.array(list(self.volatility_history))
        current_vol = volatilities[-1]
        vol_mean = np.mean(volatilities)
        vol_std = np.std(volatilities)
        
        # Classify volatility regime
        if current_vol > vol_mean + 2 * vol_std:
            return "high", min((current_vol - vol_mean) / (2 * vol_std), 1.0)
        elif current_vol < vol_mean - vol_std:
            return "low", max((vol_mean - current_vol) / vol_std, 0.0)
        else:
            return "normal", 0.5


class HiddenMarkovRegimeModel:
    """Hidden Markov Model for regime detection."""
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = None
        self.feature_history = deque(maxlen=200)  # Keep 200 data points
        self.regime_probabilities = np.ones(n_regimes) / n_regimes
        
    def update_features(self, features: np.ndarray):
        """Update feature history."""
        self.feature_history.append(features)
        
        # Retrain model periodically
        if len(self.feature_history) >= 50 and len(self.feature_history) % 10 == 0:
            self._fit_model()
    
    def _fit_model(self):
        """Fit Gaussian Mixture Model as proxy for HMM."""
        try:
            if len(self.feature_history) < 20:
                return
                
            X = np.array(list(self.feature_history))
            
            # Use Gaussian Mixture Model as simplified HMM
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                max_iter=50,
                random_state=42
            )
            self.model.fit(X)
            
            logger.debug(f"HMM model fitted with {len(X)} samples")
            
        except Exception as e:
            logger.warning(f"Failed to fit regime model: {e}")
    
    def get_current_regime(self, current_features: np.ndarray) -> Tuple[int, np.ndarray]:
        """Get current regime and probabilities."""
        if self.model is None:
            return 0, self.regime_probabilities
            
        try:
            # Predict regime
            regime = self.model.predict([current_features])[0]
            
            # Get probabilities
            probabilities = self.model.predict_proba([current_features])[0]
            
            return regime, probabilities
            
        except Exception as e:
            logger.warning(f"Failed to predict regime: {e}")
            return 0, self.regime_probabilities


class MarketRegimeDetector:
    """Main market regime detection system."""
    
    def __init__(self):
        self.cache = None
        
        # Components
        self.volatility_clustering = VolatilityClustering()
        self.hmm_model = HiddenMarkovRegimeModel()
        
        # Market data tracking
        self.market_metrics_history = deque(maxlen=100)
        self.regime_history = deque(maxlen=50)
        
        # Current state
        self.current_regime = MarketRegime.SIDEWAYS_LOW_VOL
        self.regime_confidence = 0.5
        self.regime_change_timestamp = datetime.utcnow()
        
        # Performance tracking
        self.regime_accuracy = 0.0
        self.regime_changes_detected = 0
        
    async def initialize(self):
        """Initialize the regime detector."""
        self.cache = get_trading_cache()
        logger.info("Market Regime Detector initialized")
    
    async def update_market_data(self, market_data: Dict[str, MarketData], 
                               indicators: Dict[str, Dict[str, float]]):
        """Update regime detection with new market data."""
        
        # Calculate market-wide metrics
        market_metrics = await self._calculate_market_metrics(market_data, indicators)
        self.market_metrics_history.append(market_metrics)
        
        # Update volatility clustering
        if len(self.market_metrics_history) >= 2:
            recent_metrics = list(self.market_metrics_history)[-2:]
            market_return = (recent_metrics[-1].market_volatility - recent_metrics[-2].market_volatility) / recent_metrics[-2].market_volatility
            self.volatility_clustering.update(market_return)
        
        # Create feature vector for regime detection
        features = self._create_feature_vector(market_metrics)
        self.hmm_model.update_features(features)
        
        # Detect current regime
        new_regime, regime_characteristics = await self._detect_regime(market_metrics, features)
        
        # Update regime if changed
        if new_regime != self.current_regime:
            await self._handle_regime_change(new_regime, regime_characteristics)
        
        # Update regime history
        self.regime_history.append((datetime.utcnow(), new_regime, regime_characteristics.probability))
        
        # Cache current regime information
        await self._cache_regime_info(regime_characteristics)
    
    async def _calculate_market_metrics(self, market_data: Dict[str, MarketData], 
                                      indicators: Dict[str, Dict[str, float]]) -> MarketMetrics:
        """Calculate market-wide metrics."""
        
        if not market_data:
            return MarketMetrics(
                timestamp=datetime.utcnow(),
                market_volatility=0.2,
                trend_strength=0.0,
                sector_dispersion=0.1,
                correlation_avg=0.5,
                volume_profile=1.0,
                breadth_indicator=0.5,
                credit_spreads=0.1,
                currency_volatility=0.15
            )
        
        # Calculate volatility across all symbols
        volatilities = []
        volumes = []
        returns = []
        
        for symbol, data in market_data.items():
            if symbol in indicators:
                symbol_indicators = indicators[symbol]
                volatilities.append(symbol_indicators.get('volatility', 0.2))
                volumes.append(symbol_indicators.get('volume_ratio', 1.0))
                
                # Calculate return if we have previous data
                if len(self.market_metrics_history) > 0:
                    prev_price = data.close  # Would get from previous data
                    current_return = 0.0  # Would calculate actual return
                    returns.append(current_return)
        
        # Aggregate metrics
        market_volatility = np.mean(volatilities) if volatilities else 0.2
        volume_profile = np.mean(volumes) if volumes else 1.0
        sector_dispersion = np.std(returns) if len(returns) > 5 else 0.1
        
        # Simple trend strength calculation
        trend_strength = 0.0
        if returns:
            trend_strength = np.mean(returns) / (np.std(returns) + 1e-6)
            trend_strength = np.clip(trend_strength, -1.0, 1.0)
        
        # Mock other metrics (would calculate from real data)
        correlation_avg = 0.6 if market_volatility > 0.3 else 0.4  # Higher correlation in high vol
        breadth_indicator = 0.7 if trend_strength > 0 else 0.3
        credit_spreads = market_volatility * 0.5  # Simplified relationship
        currency_volatility = market_volatility * 0.8
        
        return MarketMetrics(
            timestamp=datetime.utcnow(),
            market_volatility=market_volatility,
            trend_strength=trend_strength,
            sector_dispersion=sector_dispersion,
            correlation_avg=correlation_avg,
            volume_profile=volume_profile,
            breadth_indicator=breadth_indicator,
            credit_spreads=credit_spreads,
            currency_volatility=currency_volatility
        )
    
    def _create_feature_vector(self, metrics: MarketMetrics) -> np.ndarray:
        """Create feature vector for regime detection."""
        return np.array([
            metrics.market_volatility,
            metrics.trend_strength,
            metrics.sector_dispersion,
            metrics.correlation_avg,
            metrics.volume_profile,
            metrics.breadth_indicator,
            metrics.credit_spreads,
            metrics.currency_volatility
        ], dtype=np.float32)
    
    async def _detect_regime(self, metrics: MarketMetrics, 
                           features: np.ndarray) -> Tuple[MarketRegime, RegimeCharacteristics]:
        """Detect current market regime."""
        
        # Get volatility regime
        vol_regime, vol_confidence = self.volatility_clustering.detect_volatility_regime()
        
        # Get HMM regime
        hmm_regime, hmm_probabilities = self.hmm_model.get_current_regime(features)
        
        # Classify regime based on multiple factors
        volatility_level = metrics.market_volatility
        trend_strength = metrics.trend_strength
        correlation_breakdown = metrics.correlation_avg > 0.8  # High correlation = breakdown
        volume_surge = metrics.volume_profile > 1.5
        sector_rotation = metrics.sector_dispersion > 0.15
        
        # Determine regime
        if volatility_level > 0.4:  # High volatility
            if abs(trend_strength) > 0.3:
                if trend_strength > 0:
                    regime = MarketRegime.HIGH_VOLATILITY_BULL
                else:
                    regime = MarketRegime.HIGH_VOLATILITY_BEAR
            else:
                regime = MarketRegime.SIDEWAYS_HIGH_VOL
        else:  # Low volatility
            if abs(trend_strength) > 0.2:
                if trend_strength > 0:
                    regime = MarketRegime.LOW_VOLATILITY_BULL
                else:
                    regime = MarketRegime.LOW_VOLATILITY_BEAR
            else:
                regime = MarketRegime.SIDEWAYS_LOW_VOL
        
        # Check for crisis conditions
        if volatility_level > 0.6 and correlation_breakdown and volume_surge:
            regime = MarketRegime.CRISIS
        
        # Check for recovery conditions
        if (self.current_regime == MarketRegime.CRISIS and 
            volatility_level < 0.3 and not correlation_breakdown):
            regime = MarketRegime.RECOVERY
        
        # Calculate confidence based on feature consistency
        confidence = self._calculate_regime_confidence(metrics, vol_confidence, hmm_probabilities)
        
        # Estimate duration (simplified)
        duration_estimate = self._estimate_regime_duration(regime, metrics)
        
        characteristics = RegimeCharacteristics(
            regime=regime,
            volatility_level=volatility_level,
            trend_strength=trend_strength,
            correlation_breakdown=correlation_breakdown,
            volume_surge=volume_surge,
            sector_rotation=sector_rotation,
            probability=confidence,
            duration_estimate=duration_estimate
        )
        
        return regime, characteristics
    
    def _calculate_regime_confidence(self, metrics: MarketMetrics, 
                                   vol_confidence: float, 
                                   hmm_probabilities: np.ndarray) -> float:
        """Calculate confidence in regime classification."""
        
        # Base confidence from HMM probabilities
        hmm_confidence = np.max(hmm_probabilities)
        
        # Volatility clustering confidence
        vol_weight = 0.3
        
        # Feature consistency check
        feature_consistency = 0.7  # Would calculate based on feature stability
        
        # Combine confidences
        overall_confidence = (
            0.4 * hmm_confidence +
            0.3 * vol_confidence +
            0.3 * feature_consistency
        )
        
        return np.clip(overall_confidence, 0.1, 1.0)
    
    def _estimate_regime_duration(self, regime: MarketRegime, metrics: MarketMetrics) -> int:
        """Estimate how long the regime might last."""
        
        # Base durations (in days) for different regimes
        base_durations = {
            MarketRegime.LOW_VOLATILITY_BULL: 45,
            MarketRegime.HIGH_VOLATILITY_BULL: 20,
            MarketRegime.LOW_VOLATILITY_BEAR: 60,
            MarketRegime.HIGH_VOLATILITY_BEAR: 15,
            MarketRegime.SIDEWAYS_LOW_VOL: 30,
            MarketRegime.SIDEWAYS_HIGH_VOL: 25,
            MarketRegime.CRISIS: 10,
            MarketRegime.RECOVERY: 35
        }
        
        base_duration = base_durations.get(regime, 30)
        
        # Adjust based on volatility
        if metrics.market_volatility > 0.4:
            base_duration *= 0.7  # High volatility regimes don't last as long
        
        # Adjust based on trend strength
        if abs(metrics.trend_strength) > 0.5:
            base_duration *= 1.2  # Strong trends can persist longer
        
        return max(int(base_duration), 5)  # At least 5 days
    
    async def _handle_regime_change(self, new_regime: MarketRegime, 
                                  characteristics: RegimeCharacteristics):
        """Handle regime change event."""
        
        previous_regime = self.current_regime
        self.current_regime = new_regime
        self.regime_confidence = characteristics.probability
        self.regime_change_timestamp = datetime.utcnow()
        self.regime_changes_detected += 1
        
        logger.info(f"Market regime changed: {previous_regime.value} -> {new_regime.value} "
                   f"(confidence: {characteristics.probability:.2f})")
        
        # Get strategy recommendations
        strategy_allocations = characteristics.get_strategy_recommendations()
        
        # Cache regime change event
        if self.cache:
            regime_change_event = {
                'timestamp': datetime.utcnow().isoformat(),
                'previous_regime': previous_regime.value,
                'new_regime': new_regime.value,
                'confidence': characteristics.probability,
                'strategy_recommendations': strategy_allocations,
                'regime_characteristics': asdict(characteristics)
            }
            await self.cache.set_json("regime_change_event", regime_change_event, ttl=3600)
    
    async def _cache_regime_info(self, characteristics: RegimeCharacteristics):
        """Cache current regime information."""
        if self.cache:
            regime_info = {
                'regime': characteristics.regime.value,
                'confidence': characteristics.probability,
                'volatility_level': characteristics.volatility_level,
                'trend_strength': characteristics.trend_strength,
                'timestamp': datetime.utcnow().isoformat(),
                'strategy_recommendations': characteristics.get_strategy_recommendations(),
                'duration_estimate': characteristics.duration_estimate
            }
            await self.cache.set_json("current_market_regime", regime_info, ttl=60)
    
    async def get_current_regime(self) -> RegimeCharacteristics:
        """Get current market regime information."""
        
        # Create characteristics for current regime
        if len(self.market_metrics_history) > 0:
            latest_metrics = self.market_metrics_history[-1]
            
            return RegimeCharacteristics(
                regime=self.current_regime,
                volatility_level=latest_metrics.market_volatility,
                trend_strength=latest_metrics.trend_strength,
                correlation_breakdown=latest_metrics.correlation_avg > 0.8,
                volume_surge=latest_metrics.volume_profile > 1.5,
                sector_rotation=latest_metrics.sector_dispersion > 0.15,
                probability=self.regime_confidence,
                duration_estimate=30  # Default estimate
            )
        else:
            # Default regime
            return RegimeCharacteristics(
                regime=self.current_regime,
                volatility_level=0.2,
                trend_strength=0.0,
                correlation_breakdown=False,
                volume_surge=False,
                sector_rotation=False,
                probability=0.5,
                duration_estimate=30
            )
    
    async def get_strategy_allocations(self) -> Dict[str, float]:
        """Get recommended strategy allocations for current regime."""
        current_regime = await self.get_current_regime()
        return current_regime.get_strategy_recommendations()
    
    async def get_regime_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get regime history for analysis."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        history = []
        for timestamp, regime, confidence in self.regime_history:
            if timestamp >= cutoff_date:
                history.append({
                    'timestamp': timestamp.isoformat(),
                    'regime': regime.value,
                    'confidence': confidence
                })
        
        return history
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get regime detector performance metrics."""
        
        # Calculate regime stability
        regime_stability = 0.8  # Would calculate based on actual performance
        if len(self.regime_history) > 10:
            recent_regimes = [r[1] for r in list(self.regime_history)[-10:]]
            unique_regimes = len(set(recent_regimes))
            regime_stability = 1.0 - (unique_regimes / 10.0)
        
        return {
            'current_regime': self.current_regime.value,
            'regime_confidence': self.regime_confidence,
            'regime_changes_detected': self.regime_changes_detected,
            'regime_stability': regime_stability,
            'regime_accuracy': self.regime_accuracy,
            'model_features': len(self.hmm_model.feature_history),
            'days_since_regime_change': (datetime.utcnow() - self.regime_change_timestamp).days
        }


# Global regime detector instance
regime_detector: Optional[MarketRegimeDetector] = None


async def get_regime_detector() -> MarketRegimeDetector:
    """Get or create regime detector instance."""
    global regime_detector
    if regime_detector is None:
        regime_detector = MarketRegimeDetector()
        await regime_detector.initialize()
    return regime_detector