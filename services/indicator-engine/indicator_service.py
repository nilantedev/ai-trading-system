#!/usr/bin/env python3
"""Indicator Service - Technical indicator calculation service."""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from trading_common import MarketData, get_settings, get_logger
from trading_common.cache import get_trading_cache
from trading_common.messaging import get_message_consumer, get_message_producer
from technical_indicators import TechnicalIndicators, IndicatorResult

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class IndicatorAnalysis:
    """Complete indicator analysis for a symbol."""
    symbol: str
    timestamp: datetime
    indicators: Dict[str, IndicatorResult]
    overall_signal: str  # 'BUY', 'SELL', 'HOLD'
    signal_strength: float  # 0-1
    confidence: float  # 0-1
    analysis_summary: str


class IndicatorService:
    """Service for calculating and managing technical indicators."""
    
    def __init__(self):
        self.consumer = None
        self.producer = None
        self.cache = None
        
        self.indicators_engine = TechnicalIndicators()
        self.is_running = False
        
        # Analysis queue
        self.analysis_queue = asyncio.Queue(maxsize=500)
        
        # Performance metrics
        self.calculations_performed = 0
        self.calculation_errors = 0
        
    async def start(self):
        """Initialize and start indicator service."""
        logger.info("Starting Indicator Service")
        
        try:
            # Initialize connections
            self.consumer = await get_message_consumer()
            self.producer = await get_message_producer()
            self.cache = get_trading_cache()
            
            # Subscribe to processed market data
            await self._setup_subscriptions()
            
            # Start processing tasks
            self.is_running = True
            
            tasks = [
                asyncio.create_task(self._process_analysis_queue()),
                asyncio.create_task(self._consume_messages()),
                asyncio.create_task(self._periodic_analysis())
            ]
            
            logger.info("Indicator service started with 3 concurrent tasks")
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Failed to start indicator service: {e}")
            raise
    
    async def stop(self):
        """Stop indicator service gracefully."""
        logger.info("Stopping Indicator Service")
        self.is_running = False
        
        if self.consumer:
            await self.consumer.close()
        if self.producer:
            await self.producer.close()
        
        logger.info("Indicator Service stopped")
    
    async def _setup_subscriptions(self):
        """Subscribe to market data streams."""
        try:
            await self.consumer.subscribe_market_data(
                self._handle_market_data_message,
                subscription_name="indicator-service-market"
            )
            logger.info("Subscribed to market data stream")
        except Exception as e:
            logger.warning(f"Subscription setup failed: {e}")
    
    async def _consume_messages(self):
        """Consume messages from subscribed topics."""
        try:
            await self.consumer.start_consuming()
        except Exception as e:
            logger.error(f"Message consumption failed: {e}")
    
    async def _handle_market_data_message(self, message):
        """Handle incoming market data for indicator calculation."""
        try:
            # Parse market data
            if hasattr(message, 'symbol'):
                market_data = MarketData(
                    symbol=message.symbol,
                    timestamp=datetime.fromisoformat(message.timestamp),
                    open=message.open,
                    high=message.high,
                    low=message.low,
                    close=message.close,
                    volume=message.volume,
                    timeframe="1min",
                    data_source=message.data_source if hasattr(message, 'data_source') else "stream"
                )
            else:
                data = json.loads(message) if isinstance(message, str) else message
                market_data = MarketData(**data)
            
            # Add to analysis queue
            await self.analysis_queue.put(market_data)
            
        except Exception as e:
            logger.error(f"Failed to handle market data message: {e}")
            self.calculation_errors += 1
    
    async def _process_analysis_queue(self):
        """Process market data for indicator analysis."""
        while self.is_running:
            try:
                # Wait for market data
                market_data = await asyncio.wait_for(
                    self.analysis_queue.get(),
                    timeout=1.0
                )
                
                # Update indicators engine with new data
                self.indicators_engine.update_data(market_data.symbol, market_data)
                
                # Calculate all indicators
                indicators = self.indicators_engine.calculate_all_indicators(market_data.symbol)
                
                if indicators:
                    # Perform comprehensive analysis
                    analysis = await self._perform_indicator_analysis(market_data.symbol, indicators)
                    
                    # Cache analysis results
                    await self._cache_analysis(analysis)
                    
                    # Publish analysis
                    await self._publish_analysis(analysis)
                
                self.calculations_performed += 1
                self.analysis_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Indicator analysis error: {e}")
                self.calculation_errors += 1
    
    async def _perform_indicator_analysis(self, symbol: str, indicators: Dict[str, IndicatorResult]) -> IndicatorAnalysis:
        """Perform comprehensive analysis of all indicators."""
        
        # Collect all signals and strengths
        buy_signals = []
        sell_signals = []
        signal_strengths = []
        
        for indicator in indicators.values():
            if indicator.signal == "BUY":
                buy_signals.append(indicator)
                signal_strengths.append(indicator.strength)
            elif indicator.signal == "SELL":
                sell_signals.append(indicator)
                signal_strengths.append(indicator.strength)
        
        # Determine overall signal
        overall_signal = "HOLD"
        signal_strength = 0.0
        confidence = 0.0
        
        if len(buy_signals) > len(sell_signals):
            if len(buy_signals) >= 3:  # Require at least 3 confirming indicators
                overall_signal = "BUY"
                signal_strength = sum(s.strength for s in buy_signals) / len(buy_signals)
                confidence = min(len(buy_signals) / 10.0, 1.0)  # More indicators = higher confidence
        elif len(sell_signals) > len(buy_signals):
            if len(sell_signals) >= 3:
                overall_signal = "SELL" 
                signal_strength = sum(s.strength for s in sell_signals) / len(sell_signals)
                confidence = min(len(sell_signals) / 10.0, 1.0)
        
        # Generate analysis summary
        analysis_summary = self._generate_analysis_summary(symbol, indicators, overall_signal, buy_signals, sell_signals)
        
        return IndicatorAnalysis(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            indicators=indicators,
            overall_signal=overall_signal,
            signal_strength=signal_strength,
            confidence=confidence,
            analysis_summary=analysis_summary
        )
    
    def _generate_analysis_summary(self, symbol: str, indicators: Dict[str, IndicatorResult], 
                                 overall_signal: str, buy_signals: List[IndicatorResult], 
                                 sell_signals: List[IndicatorResult]) -> str:
        """Generate human-readable analysis summary."""
        
        summary_parts = [f"Technical Analysis for {symbol}:"]
        
        # Overall signal
        summary_parts.append(f"Overall Signal: {overall_signal}")
        
        # Key indicators
        key_indicators = []
        
        # Moving averages
        if 'sma_20' in indicators and 'sma_50' in indicators:
            sma_20 = indicators['sma_20'].value
            sma_50 = indicators['sma_50'].value
            ma_trend = "bullish" if sma_20 > sma_50 else "bearish"
            key_indicators.append(f"MA trend is {ma_trend}")
        
        # RSI
        if 'rsi' in indicators:
            rsi_val = indicators['rsi'].value
            if rsi_val <= 30:
                key_indicators.append("RSI indicates oversold conditions")
            elif rsi_val >= 70:
                key_indicators.append("RSI indicates overbought conditions")
            else:
                key_indicators.append(f"RSI is neutral at {rsi_val:.1f}")
        
        # MACD
        if 'macd' in indicators:
            macd_signal = indicators['macd'].signal
            if macd_signal in ['BUY', 'SELL']:
                key_indicators.append(f"MACD shows {macd_signal.lower()} signal")
        
        # Volume
        if 'volume_ratio' in indicators:
            vol_ratio = indicators['volume_ratio'].value
            if vol_ratio > 2.0:
                key_indicators.append("High volume activity detected")
            elif vol_ratio < 0.5:
                key_indicators.append("Low volume activity")
        
        if key_indicators:
            summary_parts.append("Key Observations: " + ", ".join(key_indicators))
        
        # Signal summary
        if buy_signals:
            buy_names = [s.name for s in buy_signals]
            summary_parts.append(f"Bullish indicators: {', '.join(buy_names)}")
        
        if sell_signals:
            sell_names = [s.name for s in sell_signals]
            summary_parts.append(f"Bearish indicators: {', '.join(sell_names)}")
        
        return ". ".join(summary_parts)
    
    async def _cache_analysis(self, analysis: IndicatorAnalysis):
        """Cache indicator analysis results."""
        try:
            if self.cache:
                # Cache latest analysis
                cache_key = f"indicator_analysis:{analysis.symbol}:latest"
                
                # Prepare data for caching
                cache_data = {
                    'symbol': analysis.symbol,
                    'timestamp': analysis.timestamp.isoformat(),
                    'overall_signal': analysis.overall_signal,
                    'signal_strength': analysis.signal_strength,
                    'confidence': analysis.confidence,
                    'analysis_summary': analysis.analysis_summary,
                    'indicators': {}
                }
                
                # Add indicator values
                for name, indicator in analysis.indicators.items():
                    cache_data['indicators'][name] = {
                        'value': indicator.value,
                        'signal': indicator.signal,
                        'strength': indicator.strength,
                        'parameters': indicator.parameters
                    }
                
                await self.cache.set_json(cache_key, cache_data, ttl=300)  # 5 minutes
                
                # Also cache historical analysis
                historical_key = f"indicator_analysis:{analysis.symbol}:{analysis.timestamp.strftime('%Y%m%d_%H%M')}"
                await self.cache.set_json(historical_key, cache_data, ttl=3600)  # 1 hour
                
        except Exception as e:
            logger.warning(f"Failed to cache analysis: {e}")
    
    async def _publish_analysis(self, analysis: IndicatorAnalysis):
        """Publish analysis results to downstream services."""
        try:
            if self.producer:
                # Create analysis message
                message_data = {
                    'symbol': analysis.symbol,
                    'timestamp': analysis.timestamp.isoformat(),
                    'overall_signal': analysis.overall_signal,
                    'signal_strength': analysis.signal_strength,
                    'confidence': analysis.confidence,
                    'analysis_summary': analysis.analysis_summary,
                    'indicator_count': len(analysis.indicators)
                }
                
                # Would publish to indicator analysis topic
                logger.debug(f"Publishing indicator analysis for {analysis.symbol}: {analysis.overall_signal}")
                
        except Exception as e:
            logger.warning(f"Failed to publish analysis: {e}")
    
    async def _periodic_analysis(self):
        """Perform periodic analysis on tracked symbols."""
        tracked_symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'GOOGL']  # Default symbols
        
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                for symbol in tracked_symbols:
                    # Check if we have enough data for analysis
                    if symbol in self.indicators_engine.price_data:
                        if len(self.indicators_engine.price_data[symbol]) >= 10:
                            # Calculate indicators for this symbol
                            indicators = self.indicators_engine.calculate_all_indicators(symbol)
                            
                            if indicators:
                                analysis = await self._perform_indicator_analysis(symbol, indicators)
                                await self._cache_analysis(analysis)
                                
                                logger.debug(f"Periodic analysis for {symbol}: {analysis.overall_signal}")
                
            except Exception as e:
                logger.warning(f"Periodic analysis error: {e}")
    
    async def get_indicator_analysis(self, symbol: str) -> Optional[IndicatorAnalysis]:
        """Get latest indicator analysis for a symbol."""
        try:
            if self.cache:
                cache_key = f"indicator_analysis:{symbol}:latest"
                cached_data = await self.cache.get_json(cache_key)
                
                if cached_data:
                    # Reconstruct IndicatorAnalysis object
                    indicators = {}
                    for name, data in cached_data.get('indicators', {}).items():
                        indicators[name] = IndicatorResult(
                            name=name,
                            value=data['value'],
                            signal=data.get('signal'),
                            strength=data.get('strength', 0.0),
                            parameters=data.get('parameters', {})
                        )
                    
                    return IndicatorAnalysis(
                        symbol=cached_data['symbol'],
                        timestamp=datetime.fromisoformat(cached_data['timestamp']),
                        indicators=indicators,
                        overall_signal=cached_data['overall_signal'],
                        signal_strength=cached_data['signal_strength'],
                        confidence=cached_data['confidence'],
                        analysis_summary=cached_data['analysis_summary']
                    )
            
        except Exception as e:
            logger.error(f"Failed to get indicator analysis for {symbol}: {e}")
        
        return None
    
    async def calculate_indicators_for_symbol(self, symbol: str, market_data_history: List[MarketData]) -> Dict[str, IndicatorResult]:
        """Calculate indicators for a symbol given historical data."""
        try:
            # Update engine with historical data
            for data in market_data_history:
                self.indicators_engine.update_data(symbol, data)
            
            # Calculate all indicators
            indicators = self.indicators_engine.calculate_all_indicators(symbol)
            self.calculations_performed += 1
            
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators for {symbol}: {e}")
            self.calculation_errors += 1
            return {}
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health status."""
        return {
            'service': 'indicator_service',
            'status': 'healthy' if self.is_running else 'stopped',
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {
                'calculations_performed': self.calculations_performed,
                'calculation_errors': self.calculation_errors,
                'success_rate': (self.calculations_performed / max(self.calculations_performed + self.calculation_errors, 1)) * 100,
                'analysis_queue_size': self.analysis_queue.qsize()
            },
            'tracked_symbols': list(self.indicators_engine.price_data.keys()),
            'connections': {
                'consumer': self.consumer is not None,
                'producer': self.producer is not None,
                'cache': self.cache is not None
            }
        }


# Global service instance
indicator_service: Optional[IndicatorService] = None


async def get_indicator_service() -> IndicatorService:
    """Get or create indicator service instance."""
    global indicator_service
    if indicator_service is None:
        indicator_service = IndicatorService()
    return indicator_service