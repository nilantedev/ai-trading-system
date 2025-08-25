#!/usr/bin/env python3
"""Signal Generation Service - Trading signal generation and orchestration."""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from trading_common import MarketData, get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.messaging import get_message_consumer, get_message_producer
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

# Import Kelly position sizing
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'portfolio-management'))
    from kelly_position_sizer import calculate_optimal_position_size, PositionSizeRecommendation
except ImportError:
    logger.warning("Kelly position sizer not available - using fallback sizing")

# Import alternative data collector
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data-ingestion'))
    from alternative_data_collector import get_alternative_data_collector, AlternativeDataSignal
except ImportError:
    logger.warning("Alternative data collector not available - using technical signals only")


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
    position_size: float  # Dollar amount (Kelly-optimized)
    position_percent: float  # Percentage of portfolio
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    strategy_breakdown: Dict[str, int]  # Count of signals per strategy
    kelly_fraction: Optional[float] = None  # Kelly fraction used
    kelly_confidence: str = "UNKNOWN"  # Kelly confidence level
    sizing_reasoning: str = ""  # Position sizing explanation
    alternative_data_signals: List = None  # Alternative data signals
    alternative_data_score: float = 0.0  # 0-1, overall alternative data sentiment


class SignalGenerationService:
    """Service for generating and orchestrating trading signals."""
    
    def __init__(self):
        self.consumer = None
        self.producer = None
        self.cache = None
        self.alternative_data_collector = None
        
        # Initialize trading strategies
        self.strategies = {
            'ma_crossover': MovingAverageCrossoverStrategy(),
            'rsi_mean_reversion': RSIMeanReversionStrategy(),
            'breakout': BreakoutStrategy(), 
            'momentum': MomentumStrategy()
        }
        
        self.is_running = False
        
        # Signal processing queue
        self.signal_queue = asyncio.Queue(maxsize=1000)
        
        # Performance metrics
        self.signals_generated = 0
        self.signals_processed = 0
        self.consensus_decisions = 0
        
        # Active signals tracking
        self.active_signals = {}  # symbol -> List[TradingSignal]
        
    async def start(self):
        """Initialize and start signal generation service."""
        logger.info("Starting Signal Generation Service")
        
        try:
            # Initialize connections
            self.consumer = await get_message_consumer()
            self.producer = await get_message_producer()
            self.cache = get_trading_cache()
            
            # Initialize alternative data collector
            try:
                self.alternative_data_collector = await get_alternative_data_collector()
                logger.info("Alternative data collector initialized")
            except Exception as e:
                logger.warning(f"Alternative data collector not available: {e}")
            
            # Subscribe to indicator analysis
            await self._setup_subscriptions()
            
            # Start processing tasks
            self.is_running = True
            
            tasks = [
                asyncio.create_task(self._process_signal_queue()),
                asyncio.create_task(self._consume_messages()),
                asyncio.create_task(self._periodic_signal_review()),
                asyncio.create_task(self._cleanup_expired_signals())
            ]
            
            logger.info("Signal generation service started with 4 concurrent tasks")
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Failed to start signal generation service: {e}")
            raise
    
    async def stop(self):
        """Stop signal generation service gracefully."""
        logger.info("Stopping Signal Generation Service")
        self.is_running = False
        
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
                
                for strategy_name, strategy in self.strategies.items():
                    try:
                        signal = await strategy.generate_signal(symbol, analysis_data)
                        if signal and signal.signal_type != SignalType.HOLD:
                            signals.append(signal)
                            logger.debug(f"Generated {signal.signal_type.value} signal from {strategy_name} for {symbol}")
                    except Exception as e:
                        logger.warning(f"Strategy {strategy_name} failed for {symbol}: {e}")
                
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
    
    async def _analyze_signal_consensus(self, symbol: str, signals: List[TradingSignal]) -> SignalConsensus:
        """Analyze multiple signals to reach consensus."""
        
        # Get alternative data signals
        alternative_signals = []
        alternative_data_score = 0.0
        
        if self.alternative_data_collector:
            try:
                alt_data = await self.alternative_data_collector.get_comprehensive_alternative_data(symbol)
                if alt_data and alt_data.signals:
                    alternative_signals = alt_data.signals
                    # Calculate overall alternative data sentiment score
                    bullish_count = sum(1 for s in alt_data.signals if s.signal_type == 'BUY')
                    bearish_count = sum(1 for s in alt_data.signals if s.signal_type == 'SELL')
                    total_signals = len(alt_data.signals)
                    
                    if total_signals > 0:
                        alternative_data_score = (bullish_count - bearish_count) / total_signals
                        alternative_data_score = (alternative_data_score + 1) / 2  # Normalize to 0-1
                        logger.debug(f"Alternative data score for {symbol}: {alternative_data_score:.2f} ({bullish_count} bullish, {bearish_count} bearish)")
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
                
                # Calculate Kelly-optimized position size
                try:
                    kelly_rec = await calculate_optimal_position_size(
                        symbol=symbol,
                        signal_confidence=consensus_confidence,
                        signal_strength=consensus_strength,
                        portfolio_value=100000,  # Default portfolio value - should come from portfolio service
                        current_price=1.0,  # Would get from market data
                        win_rate=0.55,  # Conservative estimate - should come from strategy performance
                        avg_win=0.02,  # 2% average win
                        avg_loss=-0.01  # 1% average loss
                    )
                    position_size = kelly_rec.position_dollars / 100000  # Convert to percentage
                except Exception as e:
                    logger.warning(f"Kelly sizing failed for {symbol}, using fallback: {e}")
                    position_size = min(consensus_confidence * 0.2, 0.1)  # Max 10% position fallback
                
                risk_level = "MEDIUM" if consensus_confidence > 0.7 else "HIGH"
            elif sell_weight > buy_weight * 1.5:
                consensus_signal = SignalType.SELL
                consensus_confidence = min(sell_weight / len(sell_signals), 1.0)
                consensus_strength = sum(s.strength for s in sell_signals) / len(sell_signals)
                recommended_action = "SELL"
                
                # Calculate Kelly-optimized position size
                try:
                    kelly_rec = await calculate_optimal_position_size(
                        symbol=symbol,
                        signal_confidence=consensus_confidence,
                        signal_strength=consensus_strength,
                        portfolio_value=100000,
                        current_price=1.0,
                        win_rate=0.55,
                        avg_win=0.02,
                        avg_loss=-0.01
                    )
                    position_size = kelly_rec.position_dollars / 100000
                except Exception as e:
                    logger.warning(f"Kelly sizing failed for {symbol}, using fallback: {e}")
                    position_size = min(consensus_confidence * 0.2, 0.1)
                
                risk_level = "MEDIUM" if consensus_confidence > 0.7 else "HIGH"
            else:
                # Too conflicted - hold
                recommended_action = "HOLD"
                risk_level = "HIGH"
        
        elif buy_signals and not sell_signals:
            # Clear buy consensus - boost with alternative data
            if len(buy_signals) >= 2:  # Require at least 2 confirming strategies
                consensus_signal = SignalType.BUY
                consensus_confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)
                consensus_strength = sum(s.strength for s in buy_signals) / len(buy_signals)
                
                # Boost confidence if alternative data agrees
                if alternative_data_score > 0.6:
                    consensus_confidence = min(consensus_confidence * 1.1, 1.0)  # 10% boost
                elif alternative_data_score < 0.4:  # Alternative data disagrees
                    consensus_confidence *= 0.9  # 10% reduction
                
                recommended_action = "BUY"
                
                # Calculate Kelly-optimized position size
                try:
                    kelly_rec = await calculate_optimal_position_size(
                        symbol=symbol,
                        signal_confidence=consensus_confidence,
                        signal_strength=consensus_strength,
                        portfolio_value=100000,
                        current_price=1.0,
                        win_rate=0.60,  # Higher win rate for clear consensus
                        avg_win=0.025,  # Slightly higher expected win
                        avg_loss=-0.01
                    )
                    position_size = kelly_rec.position_dollars / 100000
                except Exception as e:
                    logger.warning(f"Kelly sizing failed for {symbol}, using fallback: {e}")
                    position_size = min(consensus_confidence * len(buy_signals) * 0.05, 0.15)
                
                risk_level = "LOW" if len(buy_signals) >= 3 and consensus_confidence > 0.8 else "MEDIUM"
        
        elif sell_signals and not buy_signals:
            # Clear sell consensus - boost with alternative data
            if len(sell_signals) >= 2:
                consensus_signal = SignalType.SELL
                consensus_confidence = sum(s.confidence for s in sell_signals) / len(sell_signals)
                consensus_strength = sum(s.strength for s in sell_signals) / len(sell_signals)
                
                # Boost confidence if alternative data agrees
                if alternative_data_score < 0.4:
                    consensus_confidence = min(consensus_confidence * 1.1, 1.0)  # 10% boost
                elif alternative_data_score > 0.6:  # Alternative data disagrees
                    consensus_confidence *= 0.9  # 10% reduction
                
                recommended_action = "SELL"
                
                # Calculate Kelly-optimized position size
                try:
                    kelly_rec = await calculate_optimal_position_size(
                        symbol=symbol,
                        signal_confidence=consensus_confidence,
                        signal_strength=consensus_strength,
                        portfolio_value=100000,
                        current_price=1.0,
                        win_rate=0.60,  # Higher win rate for clear consensus
                        avg_win=0.025,
                        avg_loss=-0.01
                    )
                    position_size = kelly_rec.position_dollars / 100000
                except Exception as e:
                    logger.warning(f"Kelly sizing failed for {symbol}, using fallback: {e}")
                    position_size = min(consensus_confidence * len(sell_signals) * 0.05, 0.15)
                
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
            risk_level=risk_level,
            strategy_breakdown=strategy_breakdown,
            alternative_data_signals=alternative_signals,
            alternative_data_score=alternative_data_score
        )
    
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
                    'signal_count': len(consensus.signals)
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
                message_data = {
                    'symbol': consensus.symbol,
                    'timestamp': consensus.timestamp.isoformat(),
                    'consensus_signal': consensus.consensus_signal.value,
                    'recommended_action': consensus.recommended_action,
                    'position_size': consensus.position_size,
                    'confidence': consensus.consensus_confidence,
                    'strength': consensus.consensus_strength,
                    'risk_level': consensus.risk_level,
                    'strategy_count': len(consensus.strategy_breakdown),
                    'total_signals': len(consensus.signals)
                }
                
                # Would publish to trading signals topic
                logger.debug(f"Publishing signal consensus for {consensus.symbol}: {consensus.recommended_action}")
                
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
                        strategy_breakdown=cached_data['strategy_breakdown']
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


# Global service instance
signal_service: Optional[SignalGenerationService] = None


async def get_signal_service() -> SignalGenerationService:
    """Get or create signal generation service instance."""
    global signal_service
    if signal_service is None:
        signal_service = SignalGenerationService()
    return signal_service