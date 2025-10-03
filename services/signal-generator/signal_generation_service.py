#!/usr/bin/env python3
"""Signal Generation Service - Trading signal generation and orchestration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from trading_common import MarketData, get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.messaging import get_message_consumer, get_message_producer
from trading_common.messaging import TradingSignalMessage
from technical_indicators import IndicatorResult
from trading_strategies import (
    TradingSignal, SignalType,
    MovingAverageCrossoverStrategy,
    RSIMeanReversionStrategy, 
    BreakoutStrategy,
    MomentumStrategy
)

logger = get_logger(__name__)
settings = get_settings()

# Kelly position sizing integrated directly into signal generation
# Using simplified Kelly criterion for position sizing

@dataclass
class PositionSizeRecommendation:
    """Position sizing recommendation."""
    position_percent: float
    position_dollars: float
    kelly_fraction: float
    confidence_level: str


@dataclass
class SignalConsensus:
    """Consensus analysis of multiple trading signals with Kelly sizing."""
    symbol: str
    timestamp: datetime
    signals: List[TradingSignal]
    consensus_signal: SignalType
    consensus_confidence: float  # 0-1
    consensus_strength: float  # 0-1
    recommended_action: str  # 'BUY', 'SELL', 'HOLD', 'CLOSE'
    position_size: float  # Fraction of portfolio (0-1) used downstream for sizing
    position_percent: float  # Percentage of portfolio
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    strategy_breakdown: Dict[str, int]  # Count of signals per strategy
    kelly_fraction: Optional[float] = None  # Kelly fraction used
    kelly_confidence: str = "UNKNOWN"  # Kelly confidence level
    sizing_reasoning: str = ""  # Position sizing explanation
    alternative_data_signals: List = None  # Alternative data signals
    alternative_data_score: float = 0.0  # 0-1, overall alternative data sentiment
    # Provenance (optional)
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    feature_vector_id: Optional[str] = None


class MarketRegimeDetector:
    """Detect market regimes using Hidden Markov Models and change point detection."""
    
    def __init__(self, n_states: int = 4):
        """Initialize regime detector with HMM.
        
        Args:
            n_states: Number of market regimes (default 4: Bull, Bear, Sideways, Volatile)
        """
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=100,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.regime_names = {
            0: "BEAR_MARKET",      # Low returns, high volatility
            1: "BULL_MARKET",      # High returns, low volatility
            2: "SIDEWAYS_MARKET",  # Near-zero returns, medium volatility
            3: "VOLATILE_MARKET"   # Variable returns, very high volatility
        }
        
    def prepare_features(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> np.ndarray:
        """Prepare feature matrix for HMM.
        
        Features include:
        - Log returns
        - Rolling volatility
        - Volume ratios (if available)
        - Price momentum
        """
        if len(prices) < 20:
            raise ValueError("Need at least 20 price points for regime detection")
        
        # Calculate log returns
        log_returns = np.diff(np.log(prices))
        
        # Calculate rolling volatility (5-period)
        volatility = []
        for i in range(len(log_returns)):
            if i < 4:
                vol = np.std(log_returns[:i+1]) if i > 0 else 0
            else:
                vol = np.std(log_returns[i-4:i+1])
            volatility.append(vol)
        volatility = np.array(volatility)
        
        # Calculate momentum (rate of change)
        momentum = []
        for i in range(len(prices) - 1):
            if i < 5:
                mom = 0
            else:
                mom = (prices[i] - prices[i-5]) / prices[i-5]
            momentum.append(mom)
        momentum = np.array(momentum)
        
        # Create feature matrix
        features = np.column_stack([log_returns, volatility, momentum])
        
        # Add volume features if available
        if volumes is not None and len(volumes) == len(prices):
            volume_ratios = []
            for i in range(len(volumes) - 1):
                if i < 10:
                    ratio = 1.0
                else:
                    avg_vol = np.mean(volumes[i-10:i])
                    ratio = volumes[i] / avg_vol if avg_vol > 0 else 1.0
                volume_ratios.append(ratio)
            features = np.column_stack([features, volume_ratios])
        
        return features
    
    def train(self, historical_prices: np.ndarray, historical_volumes: Optional[np.ndarray] = None):
        """Train HMM on historical data."""
        try:
            features = self.prepare_features(historical_prices, historical_volumes)
            
            # Normalize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train HMM
            self.model.fit(features_scaled)
            self.is_trained = True
            
            logger.info(f"Regime detector trained with {self.n_states} states")
            
        except Exception as e:
            logger.error(f"Failed to train regime detector: {e}")
            raise
    
    def detect_regime(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Tuple[int, float, Dict]:
        """Detect current market regime.
        
        Returns:
            Tuple of (regime_id, confidence, regime_info)
        """
        if not self.is_trained:
            # Auto-train if not trained
            if len(prices) >= 100:
                self.train(prices[:80], volumes[:80] if volumes is not None else None)
            else:
                # Return default regime if insufficient data
                return 2, 0.5, {"regime": "SIDEWAYS_MARKET", "confidence": "LOW"}
        
        try:
            features = self.prepare_features(prices, volumes)
            features_scaled = self.scaler.transform(features)
            
            # Predict regime sequence
            states = self.model.predict(features_scaled)
            current_regime = states[-1]
            
            # Calculate confidence (probability of current state)
            probabilities = self.model.predict_proba(features_scaled)
            confidence = probabilities[-1, current_regime]
            
            # Calculate regime statistics
            regime_info = {
                "regime": self.regime_names.get(current_regime, "UNKNOWN"),
                "confidence": "HIGH" if confidence > 0.7 else "MEDIUM" if confidence > 0.4 else "LOW",
                "probability": float(confidence),
                "regime_duration": self._calculate_regime_duration(states),
                "transition_probability": self._calculate_transition_probability(current_regime)
            }
            
            return current_regime, confidence, regime_info
            
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return 2, 0.5, {"regime": "SIDEWAYS_MARKET", "confidence": "LOW", "error": str(e)}
    
    def _calculate_regime_duration(self, states: np.ndarray) -> int:
        """Calculate how long we've been in current regime."""
        if len(states) == 0:
            return 0
        
        current_regime = states[-1]
        duration = 1
        
        for i in range(len(states) - 2, -1, -1):
            if states[i] == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _calculate_transition_probability(self, current_regime: int) -> Dict[str, float]:
        """Calculate probability of transitioning to other regimes."""
        if not self.is_trained:
            return {}
        
        transition_probs = {}
        trans_matrix = self.model.transmat_
        
        for next_regime in range(self.n_states):
            if next_regime != current_regime:
                prob = trans_matrix[current_regime, next_regime]
                regime_name = self.regime_names.get(next_regime, f"STATE_{next_regime}")
                transition_probs[f"to_{regime_name}"] = float(prob)
        
        return transition_probs
    
    def detect_change_points(self, prices: np.ndarray, sensitivity: float = 2.0) -> List[int]:
        """Detect significant change points in price series.
        
        Uses CUSUM (Cumulative Sum) algorithm for change point detection.
        """
        if len(prices) < 20:
            return []
        
        log_returns = np.diff(np.log(prices))
        mean_return = np.mean(log_returns)
        std_return = np.std(log_returns)
        
        if std_return == 0:
            return []
        
        # CUSUM algorithm
        threshold = sensitivity * std_return
        cusum_pos = np.zeros(len(log_returns))
        cusum_neg = np.zeros(len(log_returns))
        
        change_points = []
        
        for i in range(1, len(log_returns)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + log_returns[i] - mean_return - threshold/2)
            cusum_neg[i] = max(0, cusum_neg[i-1] - log_returns[i] + mean_return - threshold/2)
            
            if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
                change_points.append(i)
                cusum_pos[i] = 0
                cusum_neg[i] = 0
        
        return change_points


class SignalGenerationService:
    """Service for generating and orchestrating trading signals."""
    
    def __init__(self):
        self.consumer = None
        self.producer = None
        self.cache = None
        # Alternative data integrated into main data pipeline
        self.alternative_data_enabled = False
        
        # Initialize trading strategies
        self.strategies = {
            'ma_crossover': MovingAverageCrossoverStrategy(),
            'rsi_mean_reversion': RSIMeanReversionStrategy(),
            'breakout': BreakoutStrategy(), 
            'momentum': MomentumStrategy()
        }
        
        # Initialize regime detector
        self.regime_detector = MarketRegimeDetector(n_states=4)
        self.current_regimes = {}  # symbol -> regime_info
        
        self.is_running = False
        
        # Signal processing queue
        self.signal_queue = asyncio.Queue(maxsize=1000)
        
        # Performance metrics
        self.signals_generated = 0
        self.signals_processed = 0
        self.consensus_decisions = 0
        
        # Active signals tracking
        self.active_signals = {}  # symbol -> List[TradingSignal]
        
        # Price history for regime detection
        self.price_history = {}  # symbol -> list of prices
        self.volume_history = {}  # symbol -> list of volumes
        
    async def start(self):
        """Initialize and start signal generation service."""
        logger.info("Starting Signal Generation Service")
        
        try:
            # Initialize connections
            self.consumer = await get_message_consumer()
            self.producer = await get_message_producer()
            # Ensure we await the async cache factory; otherwise self.cache is a coroutine
            self.cache = await get_trading_cache()
            
            # Alternative data is now integrated through main data pipeline
            self.alternative_data_enabled = True
            logger.info("Alternative data enabled through main pipeline")
            
            # Subscribe to indicator analysis
            await self._setup_subscriptions()
            
            # Start processing tasks
            self.is_running = True
            
            # Create background tasks without blocking
            self.background_tasks = [
                asyncio.create_task(self._process_signal_queue()),
                asyncio.create_task(self._consume_messages()),
                asyncio.create_task(self._periodic_signal_review()),
                asyncio.create_task(self._cleanup_expired_signals())
            ]
            
            logger.info("Signal generation service started with 4 concurrent tasks")
            # Don't await gather - let tasks run in background
            
        except Exception as e:
            logger.error(f"Failed to start signal generation service: {e}")
            raise
    
    async def stop(self):
        """Stop signal generation service gracefully."""
        logger.info("Stopping Signal Generation Service")
        self.is_running = False
        
        # Cancel background tasks
        if hasattr(self, 'background_tasks'):
            for task in self.background_tasks:
                task.cancel()
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        if self.consumer:
            await self.consumer.close()
        if self.producer:
            await self.producer.close()
        
        logger.info("Signal Generation Service stopped")
    
    async def _setup_subscriptions(self):
        """Subscribe to indicator analysis streams."""
        try:
            await self.consumer.subscribe_indicator_analysis(
                self._handle_indicator_analysis_message,
                subscription_name="signal-generator-indicators"
            )
            logger.info("Subscribed to indicator analysis stream")
        except Exception as e:
            logger.warning(f"Subscription setup failed: {e}")
    
    async def _consume_messages(self):
        """Consume messages from subscribed topics."""
        try:
            await self.consumer.start_consuming()
        except Exception as e:
            logger.error(f"Message consumption failed: {e}")
    
    async def _handle_indicator_analysis_message(self, message):
        """Handle incoming indicator analysis for signal generation."""
        try:
            # Parse indicator analysis
            if hasattr(message, 'symbol'):
                analysis_data = {
                    'symbol': message.symbol,
                    'timestamp': message.timestamp,
                    'indicators': message.indicators,
                    'overall_signal': message.overall_signal,
                    'signal_strength': message.signal_strength,
                    'confidence': message.confidence
                }
            else:
                analysis_data = json.loads(message) if isinstance(message, str) else message
            
            # Add to signal processing queue
            await self.signal_queue.put(analysis_data)
            
        except Exception as e:
            logger.error(f"Failed to handle indicator analysis message: {e}")
    
    async def _process_signal_queue(self):
        """Process indicator analysis for signal generation."""
        while self.is_running:
            try:
                # Wait for indicator analysis
                analysis_data = await asyncio.wait_for(
                    self.signal_queue.get(),
                    timeout=1.0
                )
                
                symbol = analysis_data['symbol']
                
                # Generate signals from all strategies
                signals = []

                # Best-effort strategy evaluation; tolerate sync strategies and input mismatch
                for strategy_name, strategy in self.strategies.items():
                    try:
                        # Strategy APIs are synchronous in this service; call directly
                        sig = strategy.generate_signal(symbol, analysis_data)  # type: ignore[arg-type]
                        if sig and getattr(sig, 'signal_type', None) != SignalType.HOLD:
                            signals.append(sig)
                            logger.debug(
                                "Generated %s signal from %s for %s",
                                getattr(sig.signal_type, 'value', str(sig.signal_type)),
                                strategy_name,
                                symbol,
                            )
                    except Exception as e:
                        # Non-fatal: input shape may not satisfy strategy expectations (e.g., history length)
                        logger.debug("strategy.eval.skipped strategy=%s symbol=%s err=%s", strategy_name, symbol, e)

                # If no concrete strategy signals, derive a lightweight signal from analysis payload as fallback
                if not signals:
                    try:
                        fallback = self._build_signal_from_analysis(symbol, analysis_data)
                        if fallback:
                            signals.append(fallback)
                            logger.debug("fallback.analysis.signal symbol=%s action=%s", symbol, fallback.signal_type.value)
                    except Exception as e:
                        logger.debug("fallback.analysis.failed symbol=%s err=%s", symbol, e)
                
                if signals:
                    # Perform consensus analysis
                    consensus = await self._analyze_signal_consensus(symbol, signals)
                    
                    # Update active signals
                    self.active_signals[symbol] = signals
                    
                    # Cache and publish consensus
                    await self._cache_signal_consensus(consensus)
                    await self._publish_signal_consensus(consensus)
                    
                    self.consensus_decisions += 1
                
                self.signals_processed += 1
                self.signal_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Signal processing error: {e}")
    
    async def _update_price_history(self, symbol: str, price: float, volume: float = 0):
        """Update price history for regime detection."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.volume_history[symbol] = []
        
        self.price_history[symbol].append(price)
        self.volume_history[symbol].append(volume)
        
        # Keep only last 200 data points for efficiency
        if len(self.price_history[symbol]) > 200:
            self.price_history[symbol] = self.price_history[symbol][-200:]
            self.volume_history[symbol] = self.volume_history[symbol][-200:]
    
    async def _get_market_regime(self, symbol: str) -> Dict:
        """Get current market regime for symbol."""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 20:
            return {"regime": "UNKNOWN", "confidence": 0.0, "regime_adjustment": 1.0}
        
        try:
            prices = np.array(self.price_history[symbol])
            volumes = np.array(self.volume_history[symbol]) if self.volume_history[symbol] else None
            
            regime_id, confidence, regime_info = self.regime_detector.detect_regime(prices, volumes)
            
            # Calculate regime adjustment factor for position sizing
            regime_adjustment = 1.0
            if regime_info["regime"] == "BULL_MARKET":
                regime_adjustment = 1.2  # Increase position size in bull market
            elif regime_info["regime"] == "BEAR_MARKET":
                regime_adjustment = 0.7  # Reduce position size in bear market
            elif regime_info["regime"] == "VOLATILE_MARKET":
                regime_adjustment = 0.5  # Half position size in volatile market
            
            regime_info["regime_adjustment"] = regime_adjustment
            self.current_regimes[symbol] = regime_info
            
            return regime_info
            
        except Exception as e:
            logger.warning(f"Regime detection failed for {symbol}: {e}")
            return {"regime": "UNKNOWN", "confidence": 0.0, "regime_adjustment": 1.0}
    
    async def _analyze_signal_consensus(self, symbol: str, signals: List[TradingSignal]) -> SignalConsensus:
        """Analyze multiple signals to reach consensus."""
        
        # Get market regime for context-aware trading
        regime_info = await self._get_market_regime(symbol)
        regime_adjustment = regime_info.get("regime_adjustment", 1.0)
        
        # Get alternative data signals
        alternative_signals = []
        alternative_data_score = 0.0
        
        # Get alternative data score from cache if available
        if self.alternative_data_enabled and self.cache:
            try:
                alt_data_key = f"alternative_data:{symbol}"
                alt_data = await self.cache.get_json(alt_data_key)
                if alt_data:
                    # Extract sentiment and other alternative signals from cache
                    sentiment_score = alt_data.get('sentiment_score', 0.5)
                    news_sentiment = alt_data.get('news_sentiment', 0.5)
                    social_sentiment = alt_data.get('social_sentiment', 0.5)
                    
                    # Combine alternative data scores
                    alternative_data_score = (sentiment_score * 0.4 + news_sentiment * 0.3 + social_sentiment * 0.3)
                    alternative_data_score = max(0.0, min(1.0, alternative_data_score))  # Clamp to 0-1
                    logger.debug(f"Alternative data score for {symbol}: {alternative_data_score:.2f}")
            except Exception as e:
                logger.warning(f"Failed to get alternative data for {symbol}: {e}")
        
        # Count signals by type
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        
        # Strategy breakdown
        strategy_breakdown = {}
        for signal in signals:
            strategy_breakdown[signal.strategy_name] = strategy_breakdown.get(signal.strategy_name, 0) + 1
        
        # Determine consensus
        consensus_signal = SignalType.HOLD
        consensus_confidence = 0.0
        consensus_strength = 0.0
        recommended_action = "HOLD"
        position_size = 0.0
        risk_level = "LOW"
        
        # Weighted consensus based on signal confidence, strength, and alternative data
        if buy_signals and sell_signals:
            # Conflicting signals - analyze strength with alternative data boost
            buy_weight = sum(s.confidence * s.strength for s in buy_signals)
            sell_weight = sum(s.confidence * s.strength for s in sell_signals)
            
            # Apply alternative data influence (20% weight)
            if alternative_data_score > 0.6:  # Bullish alternative data
                buy_weight *= (1 + 0.2 * (alternative_data_score - 0.5))
            elif alternative_data_score < 0.4:  # Bearish alternative data
                sell_weight *= (1 + 0.2 * (0.5 - alternative_data_score))
            
            if buy_weight > sell_weight * 1.5:  # Require 50% stronger signal
                consensus_signal = SignalType.BUY
                consensus_confidence = min(buy_weight / len(buy_signals), 1.0)
                consensus_strength = sum(s.strength for s in buy_signals) / len(buy_signals)
                recommended_action = "BUY"
                
                # Calculate optimal position size using Kelly criterion
                position_size = await self._calculate_kelly_position_size(
                    symbol=symbol,
                    signal_confidence=consensus_confidence,
                    signal_strength=consensus_strength,
                    regime_adjustment=regime_adjustment
                )
                
                risk_level = "MEDIUM" if consensus_confidence > 0.7 else "HIGH"
            elif sell_weight > buy_weight * 1.5:
                consensus_signal = SignalType.SELL
                consensus_confidence = min(sell_weight / len(sell_signals), 1.0)
                consensus_strength = sum(s.strength for s in sell_signals) / len(sell_signals)
                recommended_action = "SELL"
                
                # Calculate optimal position size using Kelly criterion
                position_size = await self._calculate_kelly_position_size(
                    symbol=symbol,
                    signal_confidence=consensus_confidence,
                    signal_strength=consensus_strength,
                    regime_adjustment=regime_adjustment
                )
                
                risk_level = "MEDIUM" if consensus_confidence > 0.7 else "HIGH"
            else:
                # Too conflicted - hold
                recommended_action = "HOLD"
                risk_level = "HIGH"
        
        elif buy_signals and not sell_signals:
            # Clear buy consensus - boost with alternative data
            # Accept a single strong signal as actionable, otherwise require 2+
            if len(buy_signals) >= 2 or (
                len(buy_signals) == 1 and
                (buy_signals[0].confidence >= 0.7 and buy_signals[0].strength >= 0.7)
            ):
                consensus_signal = SignalType.BUY
                consensus_confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)
                consensus_strength = sum(s.strength for s in buy_signals) / len(buy_signals)
                
                # Boost confidence if alternative data agrees
                if alternative_data_score > 0.6:
                    consensus_confidence = min(consensus_confidence * 1.1, 1.0)  # 10% boost
                elif alternative_data_score < 0.4:  # Alternative data disagrees
                    consensus_confidence *= 0.9  # 10% reduction
                
                recommended_action = "BUY"
                
                # Calculate optimal position size using Kelly criterion
                # Boost for clear consensus with multiple confirming signals
                position_size = await self._calculate_kelly_position_size(
                    symbol=symbol,
                    signal_confidence=consensus_confidence,
                    signal_strength=consensus_strength,
                    regime_adjustment=regime_adjustment
                )
                # Boost size slightly for clear consensus
                position_size *= (1 + 0.1 * (len(buy_signals) - 2) / 3)  # Up to 10% boost
                position_size = min(position_size, 0.15)  # Cap at 15% for clear signals
                
                risk_level = "LOW" if len(buy_signals) >= 3 and consensus_confidence > 0.8 else "MEDIUM"
        
        elif sell_signals and not buy_signals:
            # Clear sell consensus - boost with alternative data
            # Accept a single strong signal as actionable, otherwise require 2+
            if len(sell_signals) >= 2 or (
                len(sell_signals) == 1 and
                (sell_signals[0].confidence >= 0.7 and sell_signals[0].strength >= 0.7)
            ):
                consensus_signal = SignalType.SELL
                consensus_confidence = sum(s.confidence for s in sell_signals) / len(sell_signals)
                consensus_strength = sum(s.strength for s in sell_signals) / len(sell_signals)
                
                # Boost confidence if alternative data agrees
                if alternative_data_score < 0.4:
                    consensus_confidence = min(consensus_confidence * 1.1, 1.0)  # 10% boost
                elif alternative_data_score > 0.6:  # Alternative data disagrees
                    consensus_confidence *= 0.9  # 10% reduction
                
                recommended_action = "SELL"
                
                # Calculate optimal position size using Kelly criterion
                # Boost for clear consensus with multiple confirming signals
                position_size = await self._calculate_kelly_position_size(
                    symbol=symbol,
                    signal_confidence=consensus_confidence,
                    signal_strength=consensus_strength,
                    regime_adjustment=regime_adjustment
                )
                # Boost size slightly for clear consensus
                position_size *= (1 + 0.1 * (len(sell_signals) - 2) / 3)  # Up to 10% boost
                position_size = min(position_size, 0.15)  # Cap at 15% for clear signals
                
                risk_level = "LOW" if len(sell_signals) >= 3 and consensus_confidence > 0.8 else "MEDIUM"
        
        return SignalConsensus(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            signals=signals,
            consensus_signal=consensus_signal,
            consensus_confidence=consensus_confidence,
            consensus_strength=consensus_strength,
            recommended_action=recommended_action,
            position_size=position_size,
            position_percent=position_size,
            risk_level=risk_level,
            strategy_breakdown=strategy_breakdown,
            alternative_data_signals=alternative_signals,
            alternative_data_score=alternative_data_score
        )

    # ---------------- Fallback helpers ---------------- #
    def _build_signal_from_analysis(self, symbol: str, analysis: Dict[str, Any]) -> Optional[TradingSignal]:
        """Construct a minimal TradingSignal from indicator-analysis payload.

        Uses overall_signal and confidence/strength fields when present.
        """
        try:
            overall = str(analysis.get('overall_signal') or '').upper()
            strength = float(analysis.get('signal_strength') or 0.0)
            confidence = float(analysis.get('confidence') or strength)
            if overall not in ('BUY', 'SELL'):
                return None
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY if overall == 'BUY' else SignalType.SELL,
                confidence=max(0.0, min(confidence, 1.0)),
                strength=max(0.0, min(strength, 1.0)),
                strategy_name='analysis_fallback',
                reasoning='derived from indicator-analysis payload'
            )
        except Exception:
            return None

    def _fallback_signal_from_features(self, symbol: str, features: Dict[str, Any]) -> Optional[TradingSignal]:
        """Construct a naive momentum signal from streaming features (market/news/social).

        If price momentum and sentiment align, emit a lightweight BUY/SELL with low confidence.
        """
        try:
            market = features.get('market') or {}
            price = market.get('close') or market.get('last') or 0.0
            news = features.get('news') or []
            social = features.get('social') or []
            # Sentiment proxy
            ns = [n.get('sentiment') for n in news if isinstance(n, dict) and n.get('sentiment') is not None]
            ss = [s.get('sentiment') for s in social if isinstance(s, dict) and s.get('sentiment') is not None]
            sent_vals = [float(x) for x in ns + ss if isinstance(x, (int, float))]
            sentiment = float(np.mean(sent_vals)) if sent_vals else 0.0
            # Very naive momentum proxy: rely on presence of increasing last values in cache if available later.
            # Without history, lean on sentiment only.
            if abs(sentiment) < 0.2:
                return None
            stype = SignalType.BUY if sentiment > 0 else SignalType.SELL
            conf = min(0.6, 0.3 + abs(sentiment) * 0.5)
            return TradingSignal(
                symbol=symbol,
                signal_type=stype,
                confidence=conf,
                strength=conf,
                strategy_name='features_fallback',
                reasoning=f'derived from streaming sentiment (avg={sentiment:.2f})',
                target_price=float(price) if price else None,
            )
        except Exception:
            return None
    
    async def _cache_signal_consensus(self, consensus: SignalConsensus):
        """Cache signal consensus results."""
        try:
            if self.cache:
                # Cache latest consensus
                cache_key = f"signal_consensus:{consensus.symbol}:latest"
                
                # Prepare data for caching
                cache_data = {
                    'symbol': consensus.symbol,
                    'timestamp': consensus.timestamp.isoformat(),
                    'consensus_signal': consensus.consensus_signal.value,
                    'consensus_confidence': consensus.consensus_confidence,
                    'consensus_strength': consensus.consensus_strength,
                    'recommended_action': consensus.recommended_action,
                    'position_size': consensus.position_size,
                    'risk_level': consensus.risk_level,
                    'strategy_breakdown': consensus.strategy_breakdown,
                    'signal_count': len(consensus.signals),
                    'model_name': getattr(consensus, 'model_name', None),
                    'model_version': getattr(consensus, 'model_version', None),
                    'feature_vector_id': getattr(consensus, 'feature_vector_id', None)
                }
                
                await self.cache.set_json(cache_key, cache_data, ttl=300)  # 5 minutes
                
                # Cache historical consensus
                historical_key = f"signal_consensus:{consensus.symbol}:{consensus.timestamp.strftime('%Y%m%d_%H%M')}"
                await self.cache.set_json(historical_key, cache_data, ttl=3600)  # 1 hour
                
        except Exception as e:
            logger.warning(f"Failed to cache signal consensus: {e}")
    
    async def _publish_signal_consensus(self, consensus: SignalConsensus):
        """Publish signal consensus to downstream services."""
        try:
            if self.producer:
                # Create consensus message
                ts = consensus.timestamp.isoformat()
                # Map consensus to a TradingSignalMessage; use simple fields and encode sizing in reasoning
                msg = TradingSignalMessage(
                    id=f"{consensus.symbol}-{int(consensus.timestamp.timestamp())}",
                    timestamp=ts,
                    symbol=consensus.symbol,
                    signal_type=consensus.consensus_signal.value.upper(),
                    confidence=float(consensus.consensus_confidence or 0.0),
                    target_price=0.0,
                    stop_loss=0.0,
                    take_profit=0.0,
                    strategy_name="consensus",
                    agent_id="signal-generator",
                    reasoning=(
                        f"action={consensus.recommended_action} size={consensus.position_size:.4f}"
                        f" strength={consensus.consensus_strength:.3f} risk={consensus.risk_level}"
                    )
                )
                try:
                    await self.producer.send_trading_signal(msg)
                    logger.info(
                        "signal.publish.ok symbol=%s action=%s conf=%.2f size=%.3f",
                        consensus.symbol,
                        consensus.recommended_action,
                        consensus.consensus_confidence or 0.0,
                        consensus.position_size or 0.0,
                    )
                except Exception as send_err:
                    logger.warning("signal.publish.failed symbol=%s err=%s", consensus.symbol, send_err)
                
                # Also log a compact JSON for auditing
                logger.debug(
                    "signal.consensus %s",
                    json.dumps({
                        'symbol': consensus.symbol,
                        'timestamp': ts,
                        'action': consensus.recommended_action,
                        'size': consensus.position_size,
                        'confidence': consensus.consensus_confidence,
                        'strength': consensus.consensus_strength,
                        'risk': consensus.risk_level,
                        'signals': len(consensus.signals),
                    })
                )
                
        except Exception as e:
            logger.warning(f"Failed to publish signal consensus: {e}")
    
    async def _periodic_signal_review(self):
        """Periodically review and update signal status."""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Review every 30 seconds
                
                # Review active signals for each symbol
                for symbol, signals in list(self.active_signals.items()):
                    # Check if signals are still valid
                    current_time = datetime.utcnow()
                    valid_signals = []
                    
                    for signal in signals:
                        # Remove signals older than 5 minutes
                        signal_age = (current_time - signal.timestamp).total_seconds()
                        if signal_age < 300:  # 5 minutes
                            valid_signals.append(signal)
                    
                    if valid_signals != signals:
                        self.active_signals[symbol] = valid_signals
                        logger.debug(f"Updated active signals for {symbol}: {len(valid_signals)} remaining")
                
            except Exception as e:
                logger.warning(f"Signal review error: {e}")
    
    async def _cleanup_expired_signals(self):
        """Clean up expired signals and cache entries."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
                # Remove symbols with no active signals
                expired_symbols = []
                for symbol, signals in self.active_signals.items():
                    if not signals:
                        expired_symbols.append(symbol)
                
                for symbol in expired_symbols:
                    del self.active_signals[symbol]
                    logger.debug(f"Cleaned up expired signals for {symbol}")
                
            except Exception as e:
                logger.warning(f"Signal cleanup error: {e}")
    
    async def get_signal_consensus(self, symbol: str) -> Optional[SignalConsensus]:
        """Get latest signal consensus for a symbol."""
        try:
            if self.cache:
                cache_key = f"signal_consensus:{symbol}:latest"
                cached_data = await self.cache.get_json(cache_key)
                
                if cached_data:
                    # Reconstruct SignalConsensus object (simplified)
                    return SignalConsensus(
                        symbol=cached_data['symbol'],
                        timestamp=datetime.fromisoformat(cached_data['timestamp']),
                        signals=[],  # Not cached for performance
                        consensus_signal=SignalType(cached_data['consensus_signal']),
                        consensus_confidence=cached_data['consensus_confidence'],
                        consensus_strength=cached_data['consensus_strength'],
                        recommended_action=cached_data['recommended_action'],
                        position_size=cached_data['position_size'],
                        risk_level=cached_data['risk_level'],
                        strategy_breakdown=cached_data['strategy_breakdown'],
                        model_name=cached_data.get('model_name'),
                        model_version=cached_data.get('model_version'),
                        feature_vector_id=cached_data.get('feature_vector_id')
                    )
            
        except Exception as e:
            logger.error(f"Failed to get signal consensus for {symbol}: {e}")
        
        return None
    
    async def generate_signals_for_symbol(self, symbol: str, indicator_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate signals for a symbol using all strategies."""
        signals = []
        
        try:
            for strategy_name, strategy in self.strategies.items():
                signal = await strategy.generate_signal(symbol, indicator_data)
                if signal:
                    signals.append(signal)
                    self.signals_generated += 1
            
        except Exception as e:
            logger.error(f"Failed to generate signals for {symbol}: {e}")
        
        return signals
    
    async def _calculate_kelly_position_size(self, symbol: str, signal_confidence: float, 
                                            signal_strength: float, regime_adjustment: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Kelly formula: f = (p*b - q) / b
        where:
            p = probability of winning (win rate)
            q = probability of losing (1 - p)
            b = ratio of win amount to loss amount
        """
        try:
            # Get historical performance metrics from cache if available
            if self.cache:
                perf_key = f"strategy_performance:{symbol}"
                performance = await self.cache.get_json(perf_key)
                
                if performance and performance.get('trade_count', 0) >= 20:
                    # Use actual historical performance
                    win_rate = performance.get('win_rate', 0.5)
                    avg_win = performance.get('avg_win_percent', 0.02)
                    avg_loss = abs(performance.get('avg_loss_percent', -0.01))
                    
                    # Calculate Kelly fraction
                    if avg_loss > 0:
                        win_loss_ratio = avg_win / avg_loss
                        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
                    else:
                        kelly_fraction = 0.0
                else:
                    # Use conservative estimates when insufficient history
                    win_rate = 0.50 + (signal_confidence * 0.1)  # 50-60% win rate based on confidence
                    win_loss_ratio = 1.5 + (signal_strength * 0.5)  # 1.5-2.0 win/loss ratio
                    kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            else:
                # Fallback to conservative estimates
                win_rate = 0.50 + (signal_confidence * 0.1)
                win_loss_ratio = 1.5 + (signal_strength * 0.5)
                kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            
            # Apply safety constraints
            kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            # Apply half-Kelly for additional safety
            kelly_fraction *= 0.5
            
            # Adjust for signal confidence and market regime
            adjusted_position = kelly_fraction * signal_confidence * regime_adjustment
            
            # Final position size constraints
            position_size = min(adjusted_position, 0.1)  # Maximum 10% per position
            position_size = max(position_size, 0.01)  # Minimum 1% per position
            
            logger.debug(f"Kelly position size for {symbol}: {position_size:.3f} "
                        f"(Kelly={kelly_fraction:.3f}, confidence={signal_confidence:.2f}, "
                        f"regime={regime_adjustment:.2f})")
            
            return position_size
            
        except Exception as e:
            logger.warning(f"Kelly calculation failed for {symbol}: {e}, using fallback")
            # Conservative fallback
            return min(0.02 * signal_confidence * regime_adjustment, 0.05)
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health status."""
        return {
            'service': 'signal_generation_service',
            'status': 'healthy' if self.is_running else 'stopped',
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {
                'signals_generated': self.signals_generated,
                'signals_processed': self.signals_processed,
                'consensus_decisions': self.consensus_decisions,
                'active_symbols': len(self.active_signals),
                'total_active_signals': sum(len(signals) for signals in self.active_signals.values())
            },
            'strategies': list(self.strategies.keys()),
            'connections': {
                'consumer': self.consumer is not None,
                'producer': self.producer is not None,
                'cache': self.cache is not None
            }
        }

    # ---------------- Metrics (Optional) ---------------- #
    try:  # defer import inside class scope for clarity
        from prometheus_client import Histogram as _Hist, Counter as _Ctr  # noqa: WPS433
        _STRATEGY_LAT = _Hist('strategy_signal_latency_seconds', 'Latency per strategy signal generation', ['strategy'], buckets=(0.005,0.01,0.02,0.05,0.1,0.25,0.5,1.0))  # noqa: N806
        _STRATEGY_FAIL = _Ctr('strategy_signal_failures_total', 'Total strategy signal generation failures', ['strategy'])  # noqa: N806
        _BATCH_FAIL = _Ctr('strategy_batch_failures_total', 'Total batched strategy evaluation failures')  # noqa: N806
    except Exception:  # noqa: BLE001
        _STRATEGY_LAT = None  # type: ignore
        _STRATEGY_FAIL = None  # type: ignore
        _BATCH_FAIL = None  # type: ignore

    async def generate_signals(self, composite_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signals for a composite payload (symbol->features).

        Returns dict: symbol -> consensus summary.
        """
        results: Dict[str, Any] = {}
        for symbol, features in composite_payload.items():
            symbol_signals = []
            for name, strategy in self.strategies.items():
                start = time.time()
                try:
                    # Strategies here are synchronous; call directly and handle TypeErrors gracefully
                    sig = None
                    try:
                        sig = strategy.generate_signal(symbol, features)  # type: ignore[arg-type]
                    except Exception as inner_e:  # noqa: BLE001
                        # Strategies may expect historical market data; ignore when unavailable
                        logger.debug("strategy.generate_signal.skipped strategy=%s symbol=%s err=%s", name, symbol, inner_e)
                        sig = None
                    if sig and getattr(sig, 'signal_type', None) != SignalType.HOLD:
                        symbol_signals.append(sig)
                    if self._STRATEGY_LAT:  # type: ignore[attr-defined]
                        try:
                            self._STRATEGY_LAT.labels(strategy=name).observe(max(0.0, time.time() - start))  # type: ignore[attr-defined]
                        except Exception:  # noqa: BLE001
                            pass
                except Exception as e:  # noqa: BLE001
                    if self._STRATEGY_FAIL:  # type: ignore[attr-defined]
                        try:
                            self._STRATEGY_FAIL.labels(strategy=name).inc()  # type: ignore[attr-defined]
                        except Exception:  # noqa: BLE001
                            pass
                    logger.debug(f"strategy.generate_signal.failed strategy={name} symbol={symbol} err={e}")
            # Fallback: derive a simple signal from streaming features when strategies provide none
            if not symbol_signals:
                try:
                    fallback = self._fallback_signal_from_features(symbol, features)
                    if fallback:
                        symbol_signals.append(fallback)
                        logger.debug("fallback.features.signal symbol=%s action=%s", symbol, fallback.signal_type.value)
                except Exception as e:  # noqa: BLE001
                    logger.debug("fallback.features.failed symbol=%s err=%s", symbol, e)
            if symbol_signals:
                try:
                    consensus = await self._analyze_signal_consensus(symbol, symbol_signals)
                    # Persist and publish so downstream (execution/risk) receive the trading-signals
                    try:
                        await self._cache_signal_consensus(consensus)
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        await self._publish_signal_consensus(consensus)
                    except Exception:  # noqa: BLE001
                        pass
                    results[symbol] = {
                        'consensus_signal': consensus.consensus_signal.value,
                        'confidence': consensus.consensus_confidence,
                        'strength': consensus.consensus_strength,
                        'recommended_action': consensus.recommended_action,
                        'position_size': consensus.position_size,
                        'risk_level': consensus.risk_level,
                        'strategy_breakdown': consensus.strategy_breakdown,
                        'generated_at': consensus.timestamp.isoformat()
                    }
                except Exception as e:  # noqa: BLE001
                    # Batch-level failure for this symbol
                    if self._BATCH_FAIL:  # type: ignore[attr-defined]
                        try:
                            self._BATCH_FAIL.inc()  # type: ignore[attr-defined]
                        except Exception:  # noqa: BLE001
                            pass
                    logger.debug(f"consensus.compute.failed symbol={symbol} err={e}")
        return results


# Global service instance
signal_service: Optional[SignalGenerationService] = None


async def get_signal_service() -> SignalGenerationService:
    """Get or create signal generation service instance."""
    global signal_service
    if signal_service is None:
        signal_service = SignalGenerationService()
    return signal_service